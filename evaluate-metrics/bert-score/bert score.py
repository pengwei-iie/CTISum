from bert_score import score
import json

def calculate_bert_score(reference_texts, candidate_texts):
    P, R, F1 = score(reference_texts, candidate_texts, lang='en', verbose=False)
    bert_scores = {'precision': round(P.mean().item(), 4), 'recall': round(R.mean().item(), 4), 'f1': round(F1.mean().item(), 4)}
    return bert_scores


# 读取 JSON 文件
with open('result.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
# 获取特定字段的值
all_pre = []
all_ref= []
for item in data:
    prediction = item['prediction']
    reference = item['reference']

    all_pre.append(prediction)
    all_ref.append(reference)

# Calculate BERT score
bert_scores = calculate_bert_score(all_ref, all_pre)
print("BERT scores:", bert_scores)


