from bert_score import BERTScorer
import evaluate
import json


def eval_rouge_bert(dec, ref):
    rouge_metric = evaluate.load("./rouge/rouge.py")
    metric_bert_score = BERTScorer(lang="en")

    result_rouge = rouge_metric.compute(predictions=dec, references=ref)
    result_rouge = {k: round(v * 100, 4) for k, v in result_rouge.items()}
    print(result_rouge)

    result_bert_score = {}
    precision, recall, f1 = metric_bert_score.score(cands=dec, refs=ref)
    result_bert_score["BERTScore Precision"] = round(precision.mean().item(), 4)
    result_bert_score["BERTScore Recall"] = round(recall.mean().item(), 4)
    result_bert_score["BERTScore F1"] = round(f1.mean().item(), 4)
    print(result_bert_score)

    # return result_rouge

    return result_rouge, result_bert_score

can_path = '%s/%s.txt' % ('./result', 'ext_bert_attack_candidate')
gold_path = '%s/%s.txt' % ('./result', 'ext_bert_attack_gold')

pred_all = []
gold_all = []
with open(can_path, 'r') as save_pred:
    for line_pred in save_pred:
        # print('pred1',line_pred)
        # line = " ".join(line)
        pred_all.append(line_pred)

with open(gold_path, 'r') as save_gold:
    for line_gold in save_gold:
        # print('gold1', line_gold)
        gold_all.append(line_gold)


print('\n')
print('Start evaluating ROUGE score and BERT score !!!')
rouge, bert_score = eval_rouge_bert(pred_all, gold_all)
result = {'rouge': rouge, 'bert_score': bert_score}
print(result)