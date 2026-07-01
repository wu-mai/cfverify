#!/usr/bin/env python3
"""Prepare 30 2WikiMultihopQA questions for CF-Verify cross-dataset validation.

Output format matches HotpotQA 30-question format for consistency.
"""
import pyarrow as pa
import pyarrow.ipc as ipc
import json
import random
import ast

random.seed(42)  # 固定种子

arrow_path = '/home/tylg-jsjxy-3/.cache/huggingface/datasets/voidful___2_wiki_multihop_qa/default/0.0.0/16852fde9d85cba158cf7e6517e7a3f9415a28c0/2_wiki_multihop_qa-validation.arrow'
with open(arrow_path, 'rb') as f:
    reader = ipc.open_stream(f)
    table = reader.read_all()

all_rows = table.to_pylist()
print(f"Total 2WikiMultihopQA validation: {len(all_rows)}")

# 平衡 15 compositional + 15 comparison
compositional = [r for r in all_rows if r['type'] == 'compositional']
comparison = [r for r in all_rows if r['type'] == 'comparison']

random.shuffle(compositional)
random.shuffle(comparison)
selected = compositional[:15] + comparison[:15]
random.shuffle(selected)
print(f"Selected: {len(selected)} (comp={sum(1 for r in selected if r['type']=='compositional')}, comp={sum(1 for r in selected if r['type']=='comparison')})")

# 转换为统一格式 (与 HotpotQA 一致)
output = []
for i, q in enumerate(selected, 1):
    # 解析 context
    docs = []
    for doc_idx, ctx_item in enumerate(q['context']):
        title = ctx_item[0]
        # sentences 是 JSON 字符串
        sentences_raw = ctx_item[1]
        try:
            sentences = json.loads(sentences_raw)
        except:
            sentences = ast.literal_eval(sentences_raw)
        docs.append({
            'doc_idx': doc_idx,
            'title': title,
            'sentences': sentences
        })

    # 计算 gold 句子的全局索引
    sf = q['supporting_facts']  # [[title, sent_id], ...]
    gold_indices = []
    for sf_title, sf_sent_id in sf:
        sf_sent_id = int(sf_sent_id)
        # 在 docs 中找匹配 title (注意 title 可能带引号)
        sf_title_clean = sf_title.strip('"').strip("'")
        for doc in docs:
            doc_title_clean = doc['title'].strip('"').strip("'")
            if doc_title_clean == sf_title_clean or doc_title_clean == sf_title:
                # 计算全局句索引
                prev_sents = sum(len(d['sentences']) for d in docs if d['doc_idx'] < doc['doc_idx'])
                global_idx = prev_sents + sf_sent_id
                gold_indices.append(global_idx)
                break

    output.append({
        'question_id': i,
        'id': q['_id'],
        'question': q['question'],
        'answer': q['answer'],
        'type': q['type'],
        'documents': docs,
        'gold_sentence_indices': sorted(gold_indices),
    })

# 保存
with open('results/2wiki_30_questions.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

# 统计
print(f"\n✓ Saved 2wiki_30_questions.json")
print(f"  Avg gold sentences per Q: {sum(len(q['gold_sentence_indices']) for q in output)/len(output):.1f}")

# 检查 gold 标注是否成功
fail = 0
for q in output:
    if not q['gold_sentence_indices']:
        fail += 1
        print(f"  ⚠ Q{q['question_id']}: no gold found - {q['question'][:50]}")
print(f"  Questions without gold (need to fix): {fail}")