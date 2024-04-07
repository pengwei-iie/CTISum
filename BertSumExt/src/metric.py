import evaluate
from bert_score import BERTScorer


def format_to_metric(can_path, gold_path):
	rouge_metric = evaluate.load("./PreSumm/src/rouge/rouge.py")
	metric_bert_score = BERTScorer(lang="en")
	try:
		candidate = []
		gold = []
		with open(can_path, 'r') as open_can:
			for line in open_can:
				output_string = line.replace('<q>', '\n')
				candidate.append(output_string)

		with open(gold_path, 'r') as open_gold:
			for line in open_gold:
				output_string = line.replace('<q>', '\n')
				gold.append(output_string)

		result_rouge = rouge_metric.compute(predictions=candidate,references = gold)
		result_rouge = {k: round(v * 100, 4) for k, v in result_rouge.items()}
		print(result_rouge)

		result_bert_score = {}
		precision, recall, f1 = metric_bert_score.score(cands=candidate, refs=gold)
		result_bert_score["BERTScore Precision"] = round(precision.mean().item(), 4)
		result_bert_score["BERTScore Recall"] = round(recall.mean().item(), 4)
		result_bert_score["BERTScore F1"] = round(f1.mean().item(), 4)
		print(result_bert_score)

	except FileNotFoundError:
		print(f"Error: File not found at path: {can_path or gold_path}")
	except Exception as e:
		print(f"An unexpected error occurred: {e}")


can_path = './results/ext_bert_general_step10_candidate.txt'
gold_path = './results/ext_bert_general_step10_gold.txt'

format_to_metric(can_path, gold_path)


