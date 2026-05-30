
# import os
# import tempfile



# # ─── Whisper model size recommendations ────────────────────────────
# # tiny   → fastest, lowest accuracy  (~75MB)
# # base   → good balance for CPU      (~150MB)  ← RECOMMENDED for CPU
# # small  → better accuracy           (~460MB)
# # medium → high accuracy, slow CPU   (~1.5GB)
# DEFAULT_WHISPER_MODEL = "tiny"


# def _load_whisper(model_size: str = DEFAULT_WHISPER_MODEL):
#     """Load whisper model (cached after first call)."""
#     try:
#         import whisper
#     except ImportError:
#         raise ImportError(
#             "openai-whisper not installed. Run: pip install openai-whisper torch"
#         )
#     return whisper.load_model(model_size)


# def transcribe_audio(audio_path: str, model_size: str = DEFAULT_WHISPER_MODEL,
#                      language: str = None) -> str:
#     """
#     Transcribe an audio file to text using local Whisper model.

#     Args:
#         audio_path:  Path to audio file (.wav, .mp3, .m4a, .flac, etc.)
#         model_size:  Whisper model size — 'tiny', 'base', 'small', 'medium'
#         language:    ISO language code (e.g. 'hi' for Hindi). None = auto-detect.

#     Returns:
#         Transcribed text string.
#     """
#     if not os.path.exists(audio_path):
#         raise FileNotFoundError(f"Audio file not found: {audio_path}")

#     print(f"[Audio] Loading Whisper model ({model_size})...")
#     model = _load_whisper(model_size)

#     print(f"[Audio] Transcribing: {audio_path}")
#     options = {"fp16": False}   # fp16=False required for CPU
#     if language:
#         options["language"] = language

#     result = model.transcribe(audio_path, **options)
#     text   = result["text"].strip()

#     print(f"[Audio] Detected language: {result.get('language', 'unknown')}")
#     print(f"[Audio] Transcript: {text[:100]}{'...' if len(text)>100 else ''}")
#     return text


# def transcribe_video(video_path: str, model_size: str = DEFAULT_WHISPER_MODEL,
#                      language: str = None) -> str:
#     """
#     Extract audio from video and transcribe using local Whisper model.

#     Args:
#         video_path:  Path to video file (.mp4, .avi, .mov, .mkv, etc.)
#         model_size:  Whisper model size
#         language:    ISO language code or None for auto-detect

#     Returns:
#         Transcribed text string.
#     """
#     try:
#         from moviepy import VideoFileClip
#     except ImportError:
#         raise ImportError(
#             "moviepy not installed. Run: pip install moviepy"
#         )

#     if not os.path.exists(video_path):
#         raise FileNotFoundError(f"Video file not found: {video_path}")

#     print(f"[Video] Extracting audio from: {video_path}")

#     # Extract audio to a temporary WAV file
#     with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
#         tmp_path = tmp.name

#     try:
#         clip = VideoFileClip(video_path)
#         if clip.audio is None:
#             raise ValueError("Video file has no audio track.")
#         clip.audio.write_audiofile(tmp_path, verbose=False, logger=None)
#         clip.close()
#         print(f"[Video] Audio extracted → transcribing...")
#         text = transcribe_audio(tmp_path, model_size=model_size, language=language)
#     finally:
#         if os.path.exists(tmp_path):
#             os.remove(tmp_path)

#     return text


# def process_multimodal_input(input_path: str,
#                               model_size: str = DEFAULT_WHISPER_MODEL) -> str:
#     """
#     Auto-detect input type (text file / audio / video) and return text.
#     This is the single entry point for the multimodal pipeline.

#     For text input, just pass the text string directly to predict() — no need
#     to call this function.
#     """
#     ext = os.path.splitext(input_path)[1].lower()

#     audio_exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}
#     video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
#     text_exts  = {".txt", ".text"}

#     if ext in audio_exts:
#         return transcribe_audio(input_path, model_size=model_size)
#     elif ext in video_exts:
#         return transcribe_video(input_path, model_size=model_size)
#     elif ext in text_exts:
#         with open(input_path, "r", encoding="utf-8") as f:
#             return f.read().strip()
#     else:
#         raise ValueError(
#             f"Unsupported file type: {ext}\n"
#             f"Supported: audio={audio_exts}, video={video_exts}, text={text_exts}"
#         )


# # ─── Integration example ─────────────────────────────────────────────
# if __name__ == "__main__":
#     import sys
#     sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
#     from src.inference import predict

#     # Example usage:
#     # text = transcribe_audio("complaint.wav")
#     # result = predict(text)

#     print("Multimodal module ready.")
#     print("Phase 2 dependencies: pip install openai-whisper moviepy torch")
#     print("\nUsage:")
#     print('  from src.multimodal import transcribe_audio, transcribe_video')
#     print('  text = transcribe_audio("complaint.wav")')
#     print('  from src.inference import predict')
#     print('  result = predict(text)')
