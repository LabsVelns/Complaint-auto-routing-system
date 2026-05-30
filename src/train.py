import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
import faiss

from sklearn.ensemble          import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.svm               import SVC
from sklearn.preprocessing     import LabelEncoder
from sklearn.model_selection   import train_test_split, cross_val_score
from sklearn.metrics           import classification_report, accuracy_score, f1_score, mean_absolute_error
from sentence_transformers     import SentenceTransformer

# ─── Config ─────────────────────────────────────────────────────────
EMBEDDING_MODEL  = "paraphrase-multilingual-MiniLM-L12-v2"   # multilingual, CPU-fast
DATA_PATH        = "data/complaints_train.csv"
HIST_PATH        = "data/historical_complaints.csv"
MODELS_DIR       = "models"
DATA_DIR         = "data"


def load_data():
    print("[1/7] Loading data...")
    df   = pd.read_csv(DATA_PATH)
    hist = pd.read_csv(HIST_PATH)
    print(f"  Train: {len(df)} complaints | Historical: {len(hist)} complaints")
    return df, hist


def generate_embeddings(model, texts, desc=""):
    print(f"  Encoding {len(texts)} texts [{desc}] ...")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalised for cosine sim via dot product
    )
    print(f"  Done in {time.time()-t0:.1f}s — shape: {embeddings.shape}")
    return embeddings


def train_classifier(X_train, y_train, X_test, y_test, label, le):
    print(f"\n[Training] {label} classifier")

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="weighted")

    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 (wtd) : {f1:.4f}")
    print(classification_report(y_test, y_pred,
                                 target_names=le.classes_,
                                 zero_division=0))
    return clf, {"accuracy": round(acc,4), "f1_weighted": round(f1,4)}


def train_eta_regressor(X_train, y_train, X_test, y_test):
    print("\n[Training] ETA regressor")

    reg = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    print(f"  MAE: {mae:.2f} days")
    return reg, {"mae_days": round(mae, 2)}


def build_faiss_index(embeddings):
    """
    Build a flat L2 index.
    Since embeddings are L2-normalised, inner product == cosine similarity.
    We use IndexFlatIP (inner product) for cosine similarity search.
    """
    print("\n[6/7] Building FAISS similarity index...")
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner product on normalised vecs = cosine sim
    index.add(embeddings.astype(np.float32))
    print(f"  Index built: {index.ntotal} vectors, dim={dim}")
    return index


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR,   exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────
    df, hist = load_data()

    # ── Load embedding model ────────────────────────────────────────
    print(f"\n[2/7] Loading sentence transformer: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  Model loaded. Embedding dim: {model.get_sentence_embedding_dimension()}")

    # ── Encode training texts ───────────────────────────────────────
    print("\n[3/7] Generating embeddings for training data...")
    X = generate_embeddings(model, df["complaint_text"].tolist(), "train")

    # Save embeddings cache (speeds up re-runs)
    np.save(os.path.join(DATA_DIR, "train_embeddings.npy"), X)
    print(f"  Embeddings cached to data/train_embeddings.npy")

    # ── Label encoding ──────────────────────────────────────────────
    le_officer  = LabelEncoder()
    le_priority = LabelEncoder()

    y_officer  = le_officer.fit_transform(df["officer_id"])
    y_priority = le_priority.fit_transform(df["priority"])
    y_eta      = df["eta_days"].values

    # ── Train/test split ────────────────────────────────────────────
    X_tr, X_te, yo_tr, yo_te = train_test_split(X, y_officer,  test_size=0.2, random_state=42, stratify=y_officer)
    _,    _,    yp_tr, yp_te = train_test_split(X, y_priority, test_size=0.2, random_state=42, stratify=y_priority)
    _,    _,    ye_tr, ye_te = train_test_split(X, y_eta,       test_size=0.2, random_state=42)

    print(f"\n  Train size: {len(X_tr)} | Test size: {len(X_te)}")

    # ── Train models ────────────────────────────────────────────────
    print("\n[4/7] Training ML models...")

    clf_officer,  metrics_officer  = train_classifier(X_tr, yo_tr, X_te, yo_te, "Officer Routing",   le_officer)
    clf_priority, metrics_priority = train_classifier(X_tr, yp_tr, X_te, yp_te, "Priority",          le_priority)
    reg_eta,      metrics_eta      = train_eta_regressor(X_tr, ye_tr, X_te, ye_te)

    # ── Build FAISS index ───────────────────────────────────────────
    print("\n[5/7] Encoding historical complaints for FAISS...")
    hist_embeddings = generate_embeddings(model, hist["complaint_text"].tolist(), "historical")
    np.save(os.path.join(DATA_DIR, "hist_embeddings.npy"), hist_embeddings)

    faiss_index = build_faiss_index(hist_embeddings.astype(np.float32))

    # ── Save everything ─────────────────────────────────────────────
    print("\n[7/7] Saving models and artefacts...")

    joblib.dump(clf_officer,  os.path.join(MODELS_DIR, "officer_classifier.pkl"))
    joblib.dump(clf_priority, os.path.join(MODELS_DIR, "priority_classifier.pkl"))
    joblib.dump(reg_eta,      os.path.join(MODELS_DIR, "eta_regressor.pkl"))
    joblib.dump(le_officer,   os.path.join(MODELS_DIR, "le_officer.pkl"))
    joblib.dump(le_priority,  os.path.join(MODELS_DIR, "le_priority.pkl"))
    faiss.write_index(faiss_index, os.path.join(MODELS_DIR, "similarity.index"))

    # Save metrics report
    metrics = {
        "officer_routing": metrics_officer,
        "priority_classification": metrics_priority,
        "eta_prediction": metrics_eta,
        "embedding_model": EMBEDDING_MODEL,
        "training_samples": len(df),
        "historical_samples": len(hist),
    }
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "="*55)
    print("  TRAINING COMPLETE")
    print("="*55)
    print(f"  Officer Routing  — Accuracy: {metrics_officer['accuracy']}  F1: {metrics_officer['f1_weighted']}")
    print(f"  Priority         — Accuracy: {metrics_priority['accuracy']}  F1: {metrics_priority['f1_weighted']}")
    print(f"  ETA Prediction   — MAE: {metrics_eta['mae_days']} days")
    print(f"  FAISS Index      — {faiss_index.ntotal} vectors indexed")
    print("\n  Saved to: models/")
    print("="*55)


if __name__ == "__main__":
    # Run data generation first if files don't exist
    if not os.path.exists(DATA_PATH):
        print("Training data not found. Generating...")
        sys.path.insert(0, os.path.dirname(__file__))
        from data_generator import generate_dataset, generate_historical_complaints, OFFICER_ROSTER
        import json
        os.makedirs("data", exist_ok=True)
        generate_dataset(120).to_csv(DATA_PATH, index=False)
        generate_historical_complaints(500).to_csv(HIST_PATH, index=False)
        with open("data/officers.json", "w") as f:
            json.dump(OFFICER_ROSTER, f, indent=2)
        print("Data generated.\n")

    main()
