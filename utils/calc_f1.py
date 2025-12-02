import json
import re
import numpy as np
from sklearn import metrics

# --- PATH TO YOUR PREDICTIONS FILE ---
SAMPLE_SIZE = 200
PRED_FILE = f"/home/ezrah/CS598_DLH/project/SleepQA/DPR-main/results/reader_{SAMPLE_SIZE}_predictions.json"

# --- HELPER FUNCTIONS ---
def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in {'a', 'an', 'the'}]
    return tokens

def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction)
    truth_tokens = normalize_text(truth)
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    if len(common_tokens) == 0:
        return 0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    return 2 * (prec * rec) / (prec + rec)

def main():
    print(f"Reading predictions from: {PRED_FILE}")
    
    try:
        with open(PRED_FILE, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: File not found.")
        return

    f1_scores = []
    em_scores = []
    
    print(f"Loaded {len(data)} entries. Calculating scores...")

    for i, entry in enumerate(data):
        # --- FIX: Handle Nested Structure ---
        # The file has a list 'predictions', each with a 'prediction' object containing 'text'
        prediction_text = ""
        
        if "predictions" in entry and len(entry["predictions"]) > 0:
            # Take the top-1 prediction (usually the first one)
            top_pred = entry["predictions"][0]
            if "prediction" in top_pred and "text" in top_pred["prediction"]:
                prediction_text = top_pred["prediction"]["text"]
        
        # Fallback: Check if it's at the root (just in case format changes)
        if not prediction_text and "prediction" in entry:
             if isinstance(entry["prediction"], str):
                 prediction_text = entry["prediction"]

        # 2. Get Gold Answers
        answers = entry.get("gold_answers", [])
        if not answers:
            answers = entry.get("answers", []) # Fallback

        if not answers:
            continue

        # 3. Score against ALL answers and take the MAX
        best_f1 = 0
        best_em = 0
        
        for ans in answers:
            # F1
            f1 = compute_f1(prediction_text, ans)
            if f1 > best_f1: best_f1 = f1
            
            # EM
            if " ".join(normalize_text(prediction_text)) == " ".join(normalize_text(ans)):
                best_em = 1

        f1_scores.append(best_f1)
        em_scores.append(best_em)

    if len(f1_scores) > 0:
        print(f"\n--------------------------------------------------")
        print(f"Evaluated {len(f1_scores)} questions.")
        print(f"Exact Match (EM): {100 * sum(em_scores) / len(em_scores):.2f}%")
        print(f"F1 Score:         {100 * sum(f1_scores) / len(f1_scores):.2f}%")
        print(f"--------------------------------------------------")
    else:
        print("Error: No valid entries found.")

if __name__ == "__main__":
    main()