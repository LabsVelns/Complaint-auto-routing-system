# 🏛️ Complaint Auto-Routing System

> AI/ML-driven complaint routing with officer assignment, priority prediction, ETA estimation, and semantic similarity search. Fully offline — no external APIs.

---

## 📋 Problem Overview

Users submit complaints in text format. The system automatically:
- **Routes** the complaint to the most suitable officer (ML classification)
- **Predicts priority** — High / Medium / Low (ML classification)
- **Estimates resolution time** — ETA in days (ML regression)
- **Finds similar past complaints** — semantic vector search (FAISS)

All decisions come entirely from complaint content. No department selection, routing hints, or metadata provided in input.

---

## 🏗️ Architecture

```
Complaint Text (any language)
         │
         ▼
┌─────────────────────────────────┐
│  Sentence Transformer           │
│  paraphrase-multilingual-       │
│  MiniLM-L12-v2  (384-dim)       │
│  ✅ Offline  ✅ 50+ languages   │
└────────────┬────────────────────┘
             │  embedding vector (384-dim, L2-normalised)
    ┌────────┼─────────────────────────────┐
    │        │                             │
    ▼        ▼                             ▼
┌───────┐ ┌──────────┐  ┌──────────┐  ┌──────────────┐
│Officer│ │Priority  │  │  ETA     │  │ FAISS Index  │
│Random │ │Random    │  │  Random  │  │ IndexFlatIP  │
│Forest │ │Forest    │  │  Forest  │  │ cosine sim   │
│(10 cls│ │(3 class) │  │Regressor │  │ historical   │
│)      │ │          │  │          │  │ complaints   │
└───┬───┘ └────┬─────┘  └────┬─────┘  └──────┬───────┘
    │          │              │               │
    ▼          ▼              ▼               ▼
Officer     Priority       ETA days     Top-K similar
assigned    H/M/L          (int)        complaints
```

---

## 🧠 Model Decisions & Justification

### Embedding Model: `paraphrase-multilingual-MiniLM-L12-v2`

| Property | Value |
|----------|-------|
| Languages | 50+ (Hindi, Bengali, Tamil, Telugu, etc.) |
| Embedding dim | 384 |
| Model size | ~120MB |
| Inference speed | ~500 sentences/sec on CPU |
| Offline | ✅ Downloaded once, runs locally |

**Why this model:** Handles multilingual input (a hard requirement) while being fast enough on CPU. The `MiniLM` architecture is distilled from larger models — same quality at 5× speed.

### Classifier: `RandomForestClassifier`

- `n_estimators=300` — enough trees for stability
- `class_weight="balanced"` — handles unequal class distribution in officer routing
- Works well on embedding features without hyperparameter tuning
- Fast inference (~1ms per prediction on CPU)

**Alternatives considered:**
- SVM with RBF kernel — similar accuracy, slower training
- Logistic Regression — faster, slightly lower accuracy on imbalanced classes
- Fine-tuned transformer — higher accuracy, but requires GPU and much more data

### Regressor: `RandomForestRegressor`
- Predicts continuous ETA values directly
- Non-linear, captures domain-specific resolution patterns
- MAE reported in days

### Similarity Search: `FAISS IndexFlatIP`
- Inner product on L2-normalised vectors = cosine similarity
- Exact search (no approximation) — acceptable for historical dataset size
- For production scale (>1M complaints), switch to `IndexIVFFlat` with quantization

---

## 📁 Project Structure

```
complaint_system/
├── src/
│   ├── data_generator.py    # Synthetic dataset generation
│   ├── train.py             # Full training pipeline
│   └── inference.py         # Prediction engine (singleton model loader)
├── data/
│   ├── complaints_train.csv     # Generated training data (1200 records)
│   ├── historical_complaints.csv # FAISS source data (500 records)
│   └── officers.json             # Officer roster
├── models/
│   ├── officer_classifier.pkl    # Trained routing model
│   ├── priority_classifier.pkl   # Trained priority model
│   ├── eta_regressor.pkl         # Trained ETA model
│   ├── le_officer.pkl            # Label encoder (officers)
│   ├── le_priority.pkl           # Label encoder (priority)
│   ├── similarity.index          # FAISS index
│   └── metrics.json              # Training evaluation report
├── app.py                    # Gradio web interface
├── cli.py                    # Command-line interface
└── requirements.txt
```

---

## 🚀 Setup & Running

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate data + train all models
```bash
cd complaint_system
python src/train.py
```
This will:
- Generate 1200 synthetic training complaints (10 domains × 120 each)
- Generate 500 historical complaints for FAISS
- Train officer routing, priority, and ETA models
- Build FAISS similarity index
- Save everything to `models/`

Expected time: ~3-5 minutes on CPU

### 3a. Launch web UI
```bash
python app.py
# Open: http://localhost:7860
```

### 3b. Use CLI
```bash
# Interactive mode
python cli.py

# Single complaint
python cli.py --text "No water supply for 5 days in our area"

# JSON output
python cli.py --text "Power outage since yesterday" --json

# Run evaluation
python cli.py --eval
```

---

## 📊 Evaluation Metrics

After training, metrics are saved to `models/metrics.json`:

| Task | Metric | Target |
|------|--------|--------|
| Officer Routing | Accuracy, F1 (weighted) | >0.85 |
| Priority Classification | Accuracy, F1 (weighted) | >0.80 |
| ETA Prediction | MAE (days) | <3 days |
| Similarity Retrieval | Recall@5 | Qualitative |

---

## 🌍 Multilingual Support

The `paraphrase-multilingual-MiniLM-L12-v2` model supports 50+ languages out of the box. No translation step needed — the model maps all languages into the same semantic embedding space.

**Tested languages:** English, Hindi (हिंदी), Bengali (বাংলা), Tamil, Telugu, Marathi

---

## 🔮 Phase 2: Audio & Video Extension

Architecture is designed for easy multimodal extension:

```python
# Audio pipeline (add to inference.py)
import whisper
audio_model = whisper.load_model("base")   # ~150MB, CPU-compatible
result = audio_model.transcribe("complaint.wav")
text = result["text"]
# → feed into existing predict(text)

# Video pipeline
from moviepy.editor import VideoFileClip
clip = VideoFileClip("complaint.mp4")
clip.audio.write_audiofile("temp.wav")
# → feed temp.wav into whisper
```

No architecture changes needed — just add an audio/video → text preprocessing step before the embedding model.

---

## ⚙️ Trade-offs & Limitations

| Trade-off | Decision | Reasoning |
|-----------|----------|-----------|
| Synthetic data | Generated in code | No real complaint dataset available; domain-specific templates ensure semantic variety |
| RandomForest vs fine-tuned LLM | RandomForest | CPU-only constraint; RF on good embeddings performs comparably for classification |
| FAISS flat index vs IVF | Flat (exact) | Dataset size (500 records) makes exact search fast; switch to IVF for >100K records |
| Single embedding model | One model for all tasks | Simpler pipeline; the multilingual model handles all 4 tasks from same representation |

---

## 👨‍💻 Author

**Harshad Mane**  
AI/ML Engineer  
GitHub: github.com/LabsVelns
