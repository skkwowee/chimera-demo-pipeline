"""Stages 2+3 combined: VOD -> audio -> timestamped transcript, in one pass.

These two stages share a machine (ffmpeg + a GPU, i.e. the RunPod worker) and
the audio between them is large + transient, so we never persist it as a
separate artifact. One streaming pass:

    youtube VOD --(yt-dlp + ffmpeg)--> 16kHz mono wav in a tempdir
                --(faster-whisper)--> [(word, t_start_sec, t_end_sec), ...]
                --> discard the wav, keep only the (small) transcript

The transcript timestamps are in VOD TIME (seconds from the start of the
video). Converting VOD time -> demo ticks is stage 4 (alignment), a separate
module — this one just produces clean timestamped words.

Storage discipline (pipeline-wide): the wav lives only inside a
`tempfile.TemporaryDirectory`, removed on exit (success OR error). For a full
map VOD the wav is ~40 MB at 16 kHz mono; the kept transcript is ~tens of KB.

Requirements (NOT available in the WSL dev env — this runs on the pod):
    - ffmpeg            (apt install ffmpeg)
    - yt-dlp            (pip; already a dep)
    - faster-whisper    (pip install faster-whisper) + CUDA for GPU decode

Public API:
    transcribe_vod(video_id, model=..., ...) -> Transcript
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# faster-whisper is imported lazily inside transcribe_vod so this module can be
# imported (and the rest of the pipeline used) on machines without it.


@dataclass
class Word:
    text: str
    start: float          # VOD seconds
    end: float
    prob: float = 1.0     # ASR confidence; low-prob words are weak anchors


@dataclass
class Segment:
    text: str
    start: float
    end: float
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    video_id: str
    language: str
    duration: float                 # VOD seconds (from ASR or audio probe)
    segments: list[Segment]
    model: str

    def words(self) -> list[Word]:
        return [w for s in self.segments for w in s.words]

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "language": self.language,
            "duration": self.duration,
            "model": self.model,
            "segments": [
                {"text": s.text, "start": s.start, "end": s.end,
                 "words": [{"t": w.text, "s": round(w.start, 3),
                            "e": round(w.end, 3), "p": round(w.prob, 3)}
                           for w in s.words]}
                for s in self.segments
            ],
        }


# ---------------------------------------------------------------------------
# stage 2 — audio extraction (yt-dlp pulls the stream, ffmpeg transcodes)
# ---------------------------------------------------------------------------
def extract_audio(video_id: str, dest_wav: Path,
                  sample_rate: int = 16000) -> Path:
    """Download VOD audio-only and transcode to mono `sample_rate` wav.

    yt-dlp handles the YouTube stream; its `-x` post-processor invokes ffmpeg.
    We force 16 kHz mono PCM — exactly what whisper expects, so no second
    resample later.
    """
    dest_wav.parent.mkdir(parents=True, exist_ok=True)
    # yt-dlp writes <out>.wav; pass the path without forcing extension juggling
    out_tmpl = str(dest_wav.with_suffix(""))  # yt-dlp appends .wav
    cmd = [
        "yt-dlp",
        "-f", "bestaudio/best",
        "-x", "--audio-format", "wav",
        "--postprocessor-args", f"ffmpeg:-ac 1 -ar {sample_rate}",
        "--no-warnings", "--no-playlist",
        "-o", out_tmpl + ".%(ext)s",
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        raise RuntimeError(
            f"audio extract failed for {video_id} (rc={proc.returncode}):\n"
            f"{proc.stderr[-600:]}"
        )
    if not dest_wav.exists():
        # yt-dlp may have produced a differently-suffixed file; find it
        cands = list(dest_wav.parent.glob(dest_wav.stem + "*.wav"))
        if not cands:
            raise RuntimeError(f"no wav produced for {video_id}")
        cands[0].rename(dest_wav)
    return dest_wav


# ---------------------------------------------------------------------------
# stage 3 — ASR (faster-whisper, word-level timestamps)
# ---------------------------------------------------------------------------
def transcribe_audio(wav_path: Path, model_name: str = "large-v3",
                     device: str = "cuda", compute_type: str = "float16",
                     language: str | None = None,
                     vad_filter: bool = True) -> Transcript:
    """Run faster-whisper on a wav, returning word-timestamped segments.

    language=None lets whisper auto-detect (broadcasts are multilingual:
    EN official, plus RU/PT/TR community casts). Pass e.g. "en" to force.
    vad_filter drops long silences (desk segments, pauses) — fewer spurious
    words to align later.
    """
    from faster_whisper import WhisperModel  # lazy import (pod-only dep)

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    seg_iter, info = model.transcribe(
        str(wav_path), language=language, word_timestamps=True,
        vad_filter=vad_filter,
    )
    segments: list[Segment] = []
    for s in seg_iter:
        words = [Word(text=w.word.strip(), start=w.start, end=w.end,
                      prob=getattr(w, "probability", 1.0) or 1.0)
                 for w in (s.words or []) if w.word.strip()]
        segments.append(Segment(text=s.text.strip(), start=s.start,
                                end=s.end, words=words))
    return Transcript(
        video_id="", language=info.language,
        duration=float(getattr(info, "duration", 0.0) or 0.0),
        segments=segments, model=model_name,
    )


# ---------------------------------------------------------------------------
# combined: stage 2 + 3 in one streaming pass
# ---------------------------------------------------------------------------
def transcribe_vod(video_id: str, model_name: str = "large-v3",
                   device: str = "cuda", compute_type: str = "float16",
                   language: str | None = None,
                   sample_rate: int = 16000) -> Transcript:
    """VOD -> timestamped transcript, audio discarded.

    The wav lives only inside the TemporaryDirectory — extracted, transcribed,
    then deleted when the `with` block exits (even on exception). Keeps the
    pipeline's zero-persistent-storage discipline.
    """
    with tempfile.TemporaryDirectory(prefix="chimera_vod_") as tmpd:
        wav = Path(tmpd) / f"{video_id}.wav"
        extract_audio(video_id, wav, sample_rate=sample_rate)
        tr = transcribe_audio(wav, model_name=model_name, device=device,
                              compute_type=compute_type, language=language)
        tr.video_id = video_id
        return tr
