"""
app.py
------
Gradio web interface for the Complaint Auto-Routing System.
Supports: Text | Audio | Video

Run: python app.py
Opens at: http://localhost:7860
"""

import gradio as gr
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

# Create one executor (reused across all calls)
executor = ThreadPoolExecutor(max_workers=1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from inference import predict

# ─── Priority display helpers ────────────────────────────────────────
PRIORITY_EMOJI = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}

DOMAIN_LABELS = {
    "infrastructure":  "🏗️  Infrastructure & Roads",
    "utilities":       "🔌  Utilities",
    "sanitation":      "🧹  Sanitation & Hygiene",
    "water_supply":    "💧  Water Supply",
    "electricity":     "⚡  Electricity",
    "healthcare":      "🏥  Healthcare",
    "transport":       "🚌  Transport",
    "noise_pollution": "🔊  Noise Pollution",
    "taxation":        "🧾  Taxation",
    "housing":         "🏠  Housing",
}

# ─── Whisper loader (singleton) ──────────────────────────────────────
_whisper_model = None

def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        print("[Whisper] Loading model (base — CPU optimised)...")
        _whisper_model = whisper.load_model("tiny") # tiny is ~75MB and fast on CPU; switch to "base" for better accuracy if you have RAM to spare
        print("[Whisper] Model loaded.")
    return _whisper_model


# ─── Core result formatter ───────────────────────────────────────────
def format_results(result):
    """Turn predict() output into 4 markdown strings for the UI."""

    # Officer
    o = result["officer"]
    domain_label = DOMAIN_LABELS.get(o["domain"], o["domain"])
    officer_md = f"""
### 👤 Assigned Officer

| Field | Value |
|-------|-------|
| **Name** | {o['name']} |
| **Officer ID** | `{o['id']}` |
| **Domain** | {domain_label} |
| **Routing Confidence** | {result['confidence']['officer']*100:.1f}% |
"""

    # Priority
    pri  = result["priority"]
    dist = result["priority_dist"]
    priority_md = f"""
### {PRIORITY_EMOJI.get(pri,'⚪')} Priority: **{pri}**

Confidence: **{result['confidence']['priority']*100:.1f}%**

| Priority | Probability |
|----------|-------------|
| 🔴 High   | {dist.get('High',0)*100:.1f}% |
| 🟡 Medium | {dist.get('Medium',0)*100:.1f}% |
| 🟢 Low    | {dist.get('Low',0)*100:.1f}% |
"""

    # ETA
    eta = result["eta_days"]
    if eta <= 3:
        note = "⚡ Urgent — requires immediate action"
    elif eta <= 7:
        note = "📅 Expected within a week"
    elif eta <= 14:
        note = "🗓️  Expected within 2 weeks"
    else:
        note = "📋 Complex issue — extended resolution time"

    eta_md = f"""
### ⏱️ Estimated Resolution

# **{eta} days**

{note}
"""

    # Similar complaints
    similar = result["similar"]
    if similar:
        rows = "\n".join([
            f"| `{s['complaint_id']}` | {s['text'][:65]}{'...' if len(s['text'])>65 else ''} "
            f"| {PRIORITY_EMOJI.get(s['priority'],'')} {s['priority']} "
            f"| {s['eta_days']}d | {s['status']} | {s['similarity']*100:.1f}% |"
            for s in similar
        ])
        similar_md = f"""
### 🔍 Similar Past Complaints

| ID | Complaint | Priority | ETA | Status | Similarity |
|----|-----------|----------|-----|--------|------------|
{rows}

> *Semantic search via sentence embeddings + FAISS vector index*
"""
    else:
        similar_md = "### 🔍 Similar Complaints\n\nNo similar complaints found."

    return officer_md, priority_md, eta_md, similar_md


# ─── Tab 1: Text ─────────────────────────────────────────────────────
def process_text(complaint_text):
    if not complaint_text or len(complaint_text.strip()) < 10:
        return ("⚠️ Please enter at least 10 characters.", "", "", "")
    try:
        result = predict(complaint_text.strip())
        return format_results(result)
    except Exception as e:
        return (f"❌ Error: {e}", "", "", "")


# ─── Tab 2: Audio ─────────────────────────────────────────────────────
def process_audio(audio_path):
    """
    audio_path: file path string from Gradio Audio component.
    Pipeline: audio file → Whisper transcription → predict() → results
    """
    if audio_path is None:
        return ("⚠️ Please upload or record an audio file.", "", "", "", "")

    try:
        # Transcribe using local Whisper
        w = get_whisper()
        print(f"[Audio] Transcribing: {audio_path}")
        # Submit to background thread — main thread stays free
        future = executor.submit(w.transcribe, audio_path, fp16=False) # fp16=False for CPU
        # Wait for result (Gradio waits here, but page stays alive)
        result_w = future.result()   
        transcript = result_w["text"].strip()
        lang = result_w.get("language", "unknown")
        print(f"[Audio] Language: {lang} | Transcript: {transcript[:80]}")

        if not transcript:
            return ("⚠️ Could not transcribe audio. Please try again.", "", "", "", "")

        # Run complaint analysis on transcript
        result = predict(transcript)
        officer_md, priority_md, eta_md, similar_md = format_results(result)

        # Prepend transcript so user can verify what was heard
        transcript_md = f"""
### 🎙️ Transcribed Text
> **Detected language:** `{lang}`

*"{transcript}"*

---
"""
        return transcript_md + officer_md, priority_md, eta_md, similar_md, transcript

    except Exception as e:
        return (f"❌ Error during audio processing: {e}", "", "", "", "")


# ─── Tab 3: Video ─────────────────────────────────────────────────────
def process_video(video_path):
    """
    video_path: file path string from Gradio Video component.
    Pipeline: video → moviepy extracts audio → Whisper → predict() → results
    """
    if video_path is None:
        return ("⚠️ Please upload a video file.", "", "", "", "")

    tmp_audio = None
    try:
        from moviepy import VideoFileClip

        # Extract audio from video to a temp WAV file
        print(f"[Video] Extracting audio from: {video_path}")
        tmp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        clip = VideoFileClip(video_path)

        if clip.audio is None:
            clip.close()
            return ("⚠️ This video has no audio track.", "", "", "", "")

        clip.audio.write_audiofile(tmp_audio, verbose=False, logger=None)
        clip.close()
        print(f"[Video] Audio extracted → {tmp_audio}")

        # Transcribe the extracted audio
        w = get_whisper()
        result_w = w.transcribe(tmp_audio, fp16=False)
        transcript = result_w["text"].strip()
        lang = result_w.get("language", "unknown")
        print(f"[Video] Language: {lang} | Transcript: {transcript[:80]}")

        if not transcript:
            return ("⚠️ Could not transcribe video audio. Please try again.", "", "", "", "")

        # Run complaint analysis
        result = predict(transcript)
        officer_md, priority_md, eta_md, similar_md = format_results(result)

        transcript_md = f"""
### 🎬 Transcribed from Video
> **Detected language:** `{lang}`

*"{transcript}"*

---
"""
        return transcript_md + officer_md, priority_md, eta_md, similar_md, transcript

    except Exception as e:
        return (f"❌ Error during video processing: {e}", "", "", "", "")

    finally:
        # Always clean up temp audio file
        if tmp_audio and os.path.exists(tmp_audio):
            os.remove(tmp_audio)


# ─── Shared result columns ────────────────────────────────────────────
def results_columns():
    """Returns the 4 output markdown components used in every tab."""
    with gr.Row():
        officer_out  = gr.Markdown()
        priority_out = gr.Markdown()
    with gr.Row():
        eta_out = gr.Markdown()
    similar_out = gr.Markdown()
    return officer_out, priority_out, eta_out, similar_out


# ─── Text samples ─────────────────────────────────────────────────────
TEXT_SAMPLES = [
    ["No water supply in our area for the past 5 days. The pipeline near Gandhi Nagar is leaking heavily."],
    ["Power outage in Sector 14 since yesterday. Medicines in fridge are spoiling. Old people are suffering badly."],
    ["Garbage has not been collected from our street for 10 days. Unbearable smell and mosquitoes everywhere."],
    ["The road on MG Road has large potholes causing accidents. Multiple vehicles damaged this week."],
    ["My property tax bill is three times higher than last year without any change."],
    ["Construction work continues all night violating noise pollution norms. No sleep for 5 days."],
    # Multilingual
    ["हमारे क्षेत्र में 5 दिनों से पानी की आपूर्ति नहीं है। पाइपलाइन में बड़ी लीक है।"],
    ["আমাদের এলাকায় ৫ দিন ধরে বিদ্যুৎ নেই। বয়স্ক রোগীরা কষ্ট পাচ্ছেন।"],
]


# ─── BUILD UI ─────────────────────────────────────────────────────────
with gr.Blocks(
    title="Complaint Auto-Routing System",
    theme=gr.themes.Soft(primary_hue="blue"),
    css="""
    .tab-header { font-size: 16px; font-weight: bold; }
    footer { display: none !important; }
    """,
) as demo:

    gr.HTML("""
    <div style='text-align:center; padding: 10px 0 5px 0'>
        <h1 style='margin-bottom:4px'>🏛️ Complaint Auto-Routing System</h1>
        <p style='color:#666; font-size:14px; margin:0'>
            AI/ML-powered • Fully Offline • Multilingual • No External APIs
        </p>
    </div>
    """)

    # ── Three input tabs ──────────────────────────────────────────────
    with gr.Tabs():

        # ── TAB 1: TEXT ──────────────────────────────────────────────
        with gr.Tab("📝  Text"):
            with gr.Row():
                with gr.Column(scale=5):
                    text_input = gr.Textbox(
                        label="Enter Your Complaint",
                        placeholder="Describe your complaint in detail... (any language supported)",
                        lines=5,
                    )
                    with gr.Row():
                        text_submit = gr.Button("🚀 Analyse", variant="primary", scale=3)
                        text_clear  = gr.Button("🗑️ Clear", scale=1)
                    gr.Examples(
                        examples=TEXT_SAMPLES,
                        inputs=text_input,
                        label="💡 Sample Complaints",
                    )

                with gr.Column(scale=5):
                    gr.Markdown("### 📊 Analysis Results")
                    t_officer, t_priority, t_eta, t_similar = results_columns()

            text_submit.click(
                fn=process_text,
                inputs=[text_input],
                outputs=[t_officer, t_priority, t_eta, t_similar],
            )
            text_clear.click(
                fn=lambda: ("", "", "", "", ""),
                outputs=[text_input, t_officer, t_priority, t_eta, t_similar],
            )

        # ── TAB 2: AUDIO ─────────────────────────────────────────────
        with gr.Tab("🎙️  Audio"):
            gr.Markdown("""
> **How it works:** Upload an audio file or record directly.
> Whisper (local, offline) transcribes it → complaint is analysed automatically.
> Supports any language Whisper understands (100+ languages).
""")
            with gr.Row():
                with gr.Column(scale=5):
                    audio_input = gr.Audio(
                        label="Upload Audio or Record",
                        type="filepath",          # gives us the file path
                        sources=["upload", "microphone"],
                    )
                    audio_submit = gr.Button("🚀 Transcribe & Analyse", variant="primary")
                    gr.Markdown("""
**Supported formats:** WAV, MP3, M4A, FLAC, OGG

⚡ *First run downloads Whisper base model (~150MB). Subsequent runs are instant.*
""")

                with gr.Column(scale=5):
                    gr.Markdown("### 📊 Analysis Results")
                    a_officer, a_priority, a_eta, a_similar = results_columns()

            # Extra box showing the transcript so user can verify
            a_transcript = gr.Textbox(
                label="📝 Transcribed Text (what the model heard)",
                interactive=False,
                lines=3,
            )

            audio_submit.click(
                fn=process_audio,
                inputs=[audio_input],
                outputs=[a_officer, a_priority, a_eta, a_similar, a_transcript],
            )

        # ── TAB 3: VIDEO ─────────────────────────────────────────────
        with gr.Tab("🎬  Video"):
            gr.Markdown("""
> **How it works:** Upload a video containing a spoken complaint.
> moviepy extracts the audio → Whisper transcribes it → complaint is analysed.
> The video track is discarded — only the audio is used.
""")
            with gr.Row():
                with gr.Column(scale=5):
                    video_input = gr.Video(
                        label="Upload Video",
                        sources=["upload"],
                    )
                    video_submit = gr.Button("🚀 Extract Audio & Analyse", variant="primary")
                    gr.Markdown("""
**Supported formats:** MP4, AVI, MOV, MKV, WEBM

⚡ *Audio is extracted locally using moviepy — no upload to any server.*
""")

                with gr.Column(scale=5):
                    gr.Markdown("### 📊 Analysis Results")
                    v_officer, v_priority, v_eta, v_similar = results_columns()

            v_transcript = gr.Textbox(
                label="📝 Transcribed Text (extracted from video audio)",
                interactive=False,
                lines=3,
            )

            video_submit.click(
                fn=process_video,
                inputs=[video_input],
                outputs=[v_officer, v_priority, v_eta, v_similar, v_transcript],
            )

    # ── Footer ────────────────────────────────────────────────────────
    gr.HTML("""
    <hr style='margin-top:20px'>
    <div style='text-align:center; color:#999; font-size:12px; padding:8px 0'>
        <b>Text:</b> sentence-transformers (multilingual) + RandomForest + FAISS &nbsp;|&nbsp;
        <b>Audio/Video:</b> openai-whisper (local) + moviepy &nbsp;|&nbsp;
        <b>All models run offline — zero external API calls</b>
    </div>
    """)


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Complaint Auto-Routing System — Phase 2")
    print("  Modes: Text | Audio | Video")
    print("  Starting at: http://localhost:7860")
    print("="*55 + "\n")
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
