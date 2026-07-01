#!/usr/bin/env python3
"""
从 HotpotQA 真实数据集准备 30 个问题
用 supporting_facts 作为 gold ground truth
"""

import json
import pyarrow as pa
import pyarrow.ipc as ipc

# 加载 HotpotQA validation
with open('/home/tylg-jsjxy-3/.cache/huggingface/datasets/hotpotqa___hotpot_qa/distractor/0.0.0/1908d6afbbead072334abe2965f91bd2709910ab/hotpot_qa-validation.arrow', 'rb') as f:
    reader = ipc.open_stream(f)
    table = reader.read_all()

print(f"HotpotQA validation: {len(table)} questions")

# 随机选 30 个问题（不同 type / level 平衡）
import random
random.seed(42)

# 先按 type 分组
all_rows = table.to_pylist()
by_type = {}
for r in all_rows:
    by_type.setdefault(r['type'], []).append(r)

print(f"\nTypes: {[(t, len(rs)) for t, rs in by_type.items()]}")

# 从每个 type 中按比例采样
# 30 个问题, 平衡采样
samples_per_type = {
    'comparison': 10,
    'bridge': 10,
    'lookup': 5,  # 实际可能没有
    'arith': 5,   # 实际可能没有
}

# 简化: 直接取前 30 个
# 但确保是 hard 难度
hard_questions = [r for r in all_rows if r['level'] == 'hard']
print(f"\nHard questions: {len(hard_questions)}")

# 选前 30 个 hard 级别问题, 但平衡 type
selected = []
type_count = {'comparison': 0, 'bridge': 0}
for r in hard_questions:
    if r['type'] in type_count:
        if type_count[r['type']] < 15:
            selected.append(r)
            type_count[r['type']] += 1
    if len(selected) >= 30:
        break

print(f"\nSelected {len(selected)} questions")
print(f"  Type distribution: {type_count}")
print(f"\nFirst 3 samples:")
for r in selected[:3]:
    print(f"  Q ({r['type']}): {r['question']}")
    print(f"    Answer: {r['answer']}")
    print(f"    Supporting facts: {r['supporting_facts']}")
    print()

# 保存 30 个问题为 JSON
output_data = []
for i, q in enumerate(selected, 1):
    # 转换格式
    docs = []
    titles = q['context']['title']
    sentences_per_doc = q['context']['sentences']
    for doc_idx, (title, sents) in enumerate(zip(titles, sentences_per_doc)):
        docs.append({
            'doc_idx': doc_idx,
            'title': title,
            'sentences': sents
        })

    # 找 supporting facts 的全局 sentence index
    sf_titles = q['supporting_facts']['title']
    sf_sent_ids = q['supporting_facts']['sent_id']
    gold_local = []
    for sf_title, sf_sent_id in zip(sf_titles, sf_sent_ids):
        # 在 docs 中找匹配的 title
        for doc in docs:
            if doc['title'] == sf_title:
                # 计算全局句索引
                # 先算前面所有 doc 的总句数
                prev_sents = sum(len(d['sentences']) for d in docs if d['doc_idx'] < doc['doc_idx'])
                global_idx = prev_sents + sf_sent_id
                gold_local.append(global_idx)
                break

    output_data.append({
        'question_id': i,
        'id': q['id'],
        'question': q['question'],
        'answer': q['answer'],
        'type': q['type'],
        'level': q['level'],
        'documents': docs,
        'gold_sentence_indices': gold_local,  # 0-based global sentence index
    })

# 保存
with open('results/hotpotqa_30_real_questions.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"\n✓ Saved {len(output_data)} questions to results/hotpotqa_30_real_questions.json")
print(f"  Average gold sentences per question: {sum(len(q['gold_sentence_indices']) for q in output_data) / len(output_data):.1f}")
