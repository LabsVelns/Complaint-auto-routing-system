"""
cli.py
------
Command-line interface for the Complaint Auto-Routing System.
Usage:
    python cli.py
    python cli.py --text "Your complaint here"
    python cli.py --eval          # runs evaluation on test set
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from inference import predict


PRIORITY_COLOR = {"High": "\033[91m", "Medium": "\033[93m", "Low": "\033[92m"}
RESET = "\033[0m"
BOLD  = "\033[1m"
CYAN  = "\033[96m"
GRAY  = "\033[90m"


def print_result(text, result):
    print(f"\n{'='*65}")
    print(f"{BOLD}COMPLAINT:{RESET}")
    print(f"  {text[:120]}{'...' if len(text)>120 else ''}")
    print(f"{'─'*65}")

    # Officer
    o = result["officer"]
    print(f"{BOLD}OFFICER ASSIGNED:{RESET}")
    print(f"  Name      : {CYAN}{o['name']}{RESET}")
    print(f"  ID        : {o['id']}")
    print(f"  Domain    : {o['domain']}")
    print(f"  Confidence: {result['confidence']['officer']*100:.1f}%")

    # Priority
    pri   = result["priority"]
    color = PRIORITY_COLOR.get(pri, "")
    print(f"\n{BOLD}PRIORITY:{RESET}  {color}{BOLD}{pri}{RESET}  "
          f"(confidence: {result['confidence']['priority']*100:.1f}%)")
    dist = result["priority_dist"]
    print(f"  Distribution: High={dist.get('High',0)*100:.1f}%  "
          f"Medium={dist.get('Medium',0)*100:.1f}%  "
          f"Low={dist.get('Low',0)*100:.1f}%")

    # ETA
    print(f"\n{BOLD}ETA:{RESET}  {result['eta_days']} days")

    # Similar
    similar = result["similar"]
    print(f"\n{BOLD}SIMILAR PAST COMPLAINTS ({len(similar)} found):{RESET}")
    for i, s in enumerate(similar, 1):
        sim_pct = s["similarity"] * 100
        print(f"  [{i}] {GRAY}{s['complaint_id']}{RESET} "
              f"(sim={sim_pct:.1f}%, {s['priority']}, {s['eta_days']}d, {s['status']})")
        print(f"       {s['text'][:80]}{'...' if len(s['text'])>80 else ''}")

    print(f"{'='*65}")


def interactive_mode():
    print(f"\n{BOLD}Complaint Auto-Routing System — Interactive CLI{RESET}")
    print("Type your complaint and press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            text = input(f"{CYAN}Enter complaint:{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if text.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            break

        if len(text) < 5:
            print("  ⚠  Complaint too short. Please provide more detail.")
            continue

        result = predict(text)
        print_result(text, result)


def run_evaluation():
    """
    Run inference on a sample of the training data and report metrics.
    This gives a quick sanity check on model performance.
    """
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

    print(f"\n{BOLD}Running evaluation on training data sample...{RESET}")

    df = pd.read_csv("data/complaints_train.csv").sample(100, random_state=99)
    texts      = df["complaint_text"].tolist()
    true_off   = df["officer_id"].tolist()
    true_pri   = df["priority"].tolist()
    true_eta   = df["eta_days"].tolist()

    pred_off, pred_pri, pred_eta = [], [], []

    for i, text in enumerate(texts):
        r = predict(text)
        pred_off.append(r["officer"]["id"])
        pred_pri.append(r["priority"])
        pred_eta.append(r["eta_days"])
        if (i+1) % 20 == 0:
            print(f"  {i+1}/100 processed...")

    acc_off = accuracy_score(true_off, pred_off)
    acc_pri = accuracy_score(true_pri, pred_pri)
    f1_off  = f1_score(true_off, pred_off, average="weighted", zero_division=0)
    f1_pri  = f1_score(true_pri, pred_pri, average="weighted", zero_division=0)
    mae_eta = mean_absolute_error(true_eta, pred_eta)

    print(f"\n{'='*50}")
    print(f"{BOLD}EVALUATION RESULTS (n=100){RESET}")
    print(f"{'─'*50}")
    print(f"  Officer Routing  — Acc: {acc_off:.3f}  F1: {f1_off:.3f}")
    print(f"  Priority         — Acc: {acc_pri:.3f}  F1: {f1_pri:.3f}")
    print(f"  ETA Prediction   — MAE: {mae_eta:.2f} days")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Complaint Auto-Routing System CLI")
    parser.add_argument("--text",  type=str,  help="Single complaint text to process")
    parser.add_argument("--eval",  action="store_true", help="Run evaluation on test sample")
    parser.add_argument("--json",  action="store_true", help="Output result as JSON")
    args = parser.parse_args()

    if args.eval:
        run_evaluation()
    elif args.text:
        result = predict(args.text)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print_result(args.text, result)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
