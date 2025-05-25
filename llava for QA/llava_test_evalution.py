import json
import os
from tqdm import tqdm
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def normalize_text(text):
    return text.strip().lower().replace('"', '').replace("'", '')

def compute_bleu(reference, prediction):
    smoothie = SmoothingFunction().method4
    return [
        sentence_bleu([reference], prediction, weights=(1, 0, 0, 0), smoothing_function=smoothie),
        sentence_bleu([reference], prediction, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie),
        sentence_bleu([reference], prediction, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie),
        sentence_bleu([reference], prediction, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie),
    ]

def compute_rouge(reference, prediction, rouge_metric):
    return rouge_metric.get_scores(prediction, reference)[0]['rouge-l']['f']

def load_predictions(pred_file):
    predictions = {}
    with open(pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            j = json.loads(line)
            predictions[j['question_id']] = normalize_text(j['text'])
    return predictions

def load_ground_truth(gt_file):
    ground_truth = {}
    with open(gt_file, 'r', encoding='utf-8') as f:
        for line in f:
            j = json.loads(line)
            ground_truth[j['qid']] = [normalize_text(ans) for ans in j['answers']]
    return ground_truth

def evaluate_webqa(pred_path, gt_path):
    predictions = load_predictions(pred_path)
    ground_truth = load_ground_truth(gt_path)

    total = 0
    em_count = 0
    bleu1, bleu2, bleu3, bleu4 = 0, 0, 0, 0
    rouge_l = 0

    rouge_metric = Rouge()

    for qid in tqdm(ground_truth):
        if qid not in predictions:
            continue

        total += 1
        pred = predictions[qid]
        refs = ground_truth[qid]

        em = any(pred == ref for ref in refs)
        if em:
            em_count += 1

        
        bleu_scores = [compute_bleu(ref, pred) for ref in refs]
        best_bleu = list(map(max, zip(*bleu_scores)))  

        bleu1 += best_bleu[0]
        bleu2 += best_bleu[1]
        bleu3 += best_bleu[2]
        bleu4 += best_bleu[3]

        
        rouge_l_scores = [compute_rouge(ref, pred, rouge_metric) for ref in refs]
        rouge_l += max(rouge_l_scores)

    print("\n=== WebQA Evaluation Result ===")
    print(f"Total samples evaluated: {total}")
    print(f"Exact Match (EM): {em_count / total * 100:.2f}")
    print(f"BLEU-1: {bleu1 / total * 100:.2f}")
    print(f"BLEU-2: {bleu2 / total * 100:.2f}")
    print(f"BLEU-3: {bleu3 / total * 100:.2f}")
    print(f"BLEU-4: {bleu4 / total * 100:.2f}")
    print(f"ROUGE-L: {rouge_l / total * 100:.2f}")

if __name__ == "__main__":
    pred_file = "results/llava_webqa_outputs/webqa_llava_output.jsonl"
    gt_file = "results/llava_webqa_outputs/processed_webqa.jsonl"
    evaluate_webqa(pred_file, gt_file)