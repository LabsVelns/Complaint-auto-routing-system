"""
inference.py
------------
Loads all trained models and provides a single predict() function.

Returns:
    {
        "officer":      { "id": "OFF005", "name": "Deepak Singh", "domain": "electricity" },
        "priority":     "High",
        "eta_days":     3,
        "similar":      [ { "complaint_id": ..., "text": ..., "similarity": 0.93, ... }, ... ],
        "confidence":   { "officer": 0.87, "priority": 0.91 }
    }
"""

import os
import json
import numpy as np
import pandas as pd
import faiss
import joblib
from sentence_transformers import SentenceTransformer

# ─── Paths ───────────────────────────────────────────────────────────
MODELS_DIR = "models"
DATA_DIR   = "data"

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# ─── Singleton loader (load once, reuse) ────────────────────────────
_cache = {}

def _load_all():
    if _cache:
        return _cache

    print("[Inference] Loading models (first call only)...")

    _cache["embed_model"]    = SentenceTransformer(EMBEDDING_MODEL)
    _cache["clf_officer"]    = joblib.load(os.path.join(MODELS_DIR, "officer_classifier.pkl"))
    _cache["clf_priority"]   = joblib.load(os.path.join(MODELS_DIR, "priority_classifier.pkl"))
    _cache["reg_eta"]        = joblib.load(os.path.join(MODELS_DIR, "eta_regressor.pkl"))
    _cache["le_officer"]     = joblib.load(os.path.join(MODELS_DIR, "le_officer.pkl"))
    _cache["le_priority"]    = joblib.load(os.path.join(MODELS_DIR, "le_priority.pkl"))
    _cache["faiss_index"]    = faiss.read_index(os.path.join(MODELS_DIR, "similarity.index"))

    # Historical complaints for displaying similar results
    _cache["hist_df"]        = pd.read_csv(os.path.join(DATA_DIR, "historical_complaints.csv"))

    # Officer roster
    with open(os.path.join(DATA_DIR, "officers.json")) as f:
        _cache["officers"]   = json.load(f)

    print("[Inference] All models loaded.")
    return _cache


def _embed(text: str, m) -> np.ndarray:
    """Embed a single text, return normalised float32 vector shaped (1, dim)."""
    vec = m.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vec.astype(np.float32)


def predict(complaint_text: str, top_k_similar: int = 5) -> dict:
    """
    Main inference function.

    Args:
        complaint_text:  Raw complaint string (any language supported by multilingual model)
        top_k_similar:   Number of similar past complaints to retrieve

    Returns:
        dict with officer, priority, eta_days, similar, confidence
    """
    c = _load_all()

    # ── Step 1: Embed the complaint ──────────────────────────────────
    vec = _embed(complaint_text, c["embed_model"])

    # ── Step 2: Officer routing ──────────────────────────────────────
    officer_proba   = c["clf_officer"].predict_proba(vec)[0]
    officer_idx     = int(np.argmax(officer_proba))
    officer_conf    = float(officer_proba[officer_idx])
    officer_id      = c["le_officer"].inverse_transform([officer_idx])[0]
    officer_info    = c["officers"].get(officer_id, {})

    # ── Step 3: Priority prediction ──────────────────────────────────
    priority_proba  = c["clf_priority"].predict_proba(vec)[0]
    priority_idx    = int(np.argmax(priority_proba))
    priority_conf   = float(priority_proba[priority_idx])
    priority_label  = c["le_priority"].inverse_transform([priority_idx])[0]

    # All class probabilities for display
    priority_classes = c["le_priority"].classes_
    priority_dist    = {cls: round(float(p), 3)
                        for cls, p in zip(priority_classes, priority_proba)}

    # ── Step 4: ETA prediction ───────────────────────────────────────
    eta_raw  = float(c["reg_eta"].predict(vec)[0])
    eta_days = max(1, round(eta_raw))

    # ── Step 5: Similarity search ────────────────────────────────────
    distances, indices = c["faiss_index"].search(vec, top_k_similar)
    # distances are inner products (cosine similarity since vecs are normalised)
    similar = []
    hist_df = c["hist_df"]
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        row = hist_df.iloc[idx]
        similar.append({
            "complaint_id":  row.get("complaint_id", f"HIST{idx}"),
            "text":          row["complaint_text"],
            "priority":      row["priority"],
            "officer_id":    row["officer_id"],
            "eta_days":      int(row["eta_days"]),
            "status":        row.get("status", "Resolved"),
            "similarity":    round(float(dist), 4),
        })

    # ── Assemble result ──────────────────────────────────────────────
    result = {
        "officer": {
            "id":     officer_id,
            "name":   officer_info.get("name", officer_id),
            "domain": officer_info.get("domain", ""),
        },
        "priority":       priority_label,
        "priority_dist":  priority_dist,
        "eta_days":       eta_days,
        "similar":        similar,
        "confidence": {
            "officer":  round(officer_conf,  3),
            "priority": round(priority_conf, 3),
        },
    }
    return result


# ─── CLI quick test ──────────────────────────────────────────────────
if __name__ == "__main__":
    test_complaints = [
        "There is no water supply in our area for the past 5 days. The pipeline near Gandhi Nagar is leaking heavily.",
        "Power outage in Sector 14 since yesterday. Medicines in fridge are spoiling. Old people are suffering.",
        "Garbage has not been collected from our street for 10 days. Unbearable smell and mosquitoes everywhere.",
        "The road on MG Road has large potholes causing accidents. Multiple vehicles have been damaged.",
        "My property tax bill is three times higher than last year without any change in property.",
    ]

    for text in test_complaints:
        print("\n" + "="*60)
        print(f"COMPLAINT: {text[:80]}...")
        result = predict(text)
        print(f"  Officer  : {result['officer']['name']} ({result['officer']['id']}) — domain: {result['officer']['domain']}")
        print(f"  Priority : {result['priority']} (conf: {result['confidence']['priority']})")
        print(f"  ETA      : {result['eta_days']} days")
        print(f"  Similar  : {len(result['similar'])} complaints found")
        if result['similar']:
            top = result['similar'][0]
            print(f"    Top match (sim={top['similarity']}): {top['text'][:70]}...")
