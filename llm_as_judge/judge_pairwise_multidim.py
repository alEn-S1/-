#!/usr/bin/env python3
"""
多维度Pairwise Comparison评测脚本
基于Qwen3-235B Judge API进行多维度评分
输出格式符合用户指定的JSON模板
"""

import json
import requests
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
from datetime import datetime

# ==================== API配置 ====================

JUDGE_API_URL = "https://aimpapi.midea.com/t-aigc/aimp-qwen3-235b-a22b/v1/chat/completions"
JUDGE_API_KEY = "msk-c3e6e836acff79160513e563d6d288d3e24b9605349a03bb9564e9a7b3bafefe"

# 数据路径
BASE_MODEL_FILE = "/data/private/LLaMA-Factory/generated_predictions-qwen3-4b-templateqwen3.jsonl"
TUNED_MODEL_FILE = "/data/private/LLaMA-Factory/generated_predictions-qwen3-4b-lora-newtemplateqwen3.jsonl"

# 评测参数
NUM_SAMPLES = 100   # 测试数量，None表示全部
MAX_WORKERS = 5     # API并发数
OUTPUT_DIR = "/data/private/outputs"
PAIRWISE_DATA_FILE = f"{OUTPUT_DIR}/pairwise_comparison_data.jsonl"
EVALUATION_RESULTS_FILE = f"{OUTPUT_DIR}/evaluation_results.jsonl"
SUMMARY_FILE = f"{OUTPUT_DIR}/evaluation_summary.json"

# ==================== 评测维度定义 ====================

EVALUATION_DIMENSIONS = {
    "correctness": {
        "name": "正确性",
        "weight": 0.30,
        "description": "回答的事实准确性、逻辑推理的正确性，是否有错误信息"
    },
    "relevance": {
        "name": "相关性",
        "weight": 0.15,
        "description": "回答是否切题，是否准确理解并回应了用户的问题"
    },
    "completeness": {
        "name": "完整性",
        "weight": 0.15,
        "description": "回答是否全面，是否涵盖了关键信息，有无遗漏重要内容"
    },
    "reasoning": {
        "name": "推理能力",
        "weight": 0.20,
        "description": "思考过程的逻辑性、推理链条的清晰度、解释的深度"
    },
    "clarity": {
        "name": "表达清晰度",
        "weight": 0.10,
        "description": "语言是否简洁明了、表述是否自然流畅、结构是否清晰"
    },
    "helpfulness": {
        "name": "有用性",
        "weight": 0.10,
        "description": "回答对用户的实际帮助程度、是否易于理解和应用"
    }
}

# ==================== 工具函数 ====================

def extract_think_and_answer(text: str) -> Tuple[str, str]:
    """提取thinking和answer"""
    if '<think>' in text and '</think>' in text:
        parts = text.split('</think>')
        thinking = parts[0].split('<think>')[-1].strip()
        answer = parts[-1].strip()
        return thinking, answer
    return "", text.strip()

def call_judge_api_pairwise(
    prompt: str,
    response_a: str,
    response_b: str,
    label: str,
    dimension_key: str
) -> Dict:
    """
    调用Judge API进行Pairwise对比评分
    
    返回: {"score": 1-5, "reason": "..."}
    - 1分: A明显更优
    - 2分: A略优
    - 3分: 相当
    - 4分: B略优
    - 5分: B明显更优
    """
    dim_config = EVALUATION_DIMENSIONS[dimension_key]
    
    # 提取thinking和answer
    think_a, ans_a = extract_think_and_answer(response_a)
    think_b, ans_b = extract_think_and_answer(response_b)
    
    judge_prompt = f"""# 任务说明
你是一个专业的模型回答评测员。请对两个AI模型的回答进行Pairwise比较，并在【{dim_config['name']}】这个维度上进行评分。

## 评分维度
**{dim_config['name']}**: {dim_config['description']}

## 问题
{prompt}

## 参考答案
{label}

## 模型A的回答
{response_a}

## 模型B的回答
{response_b}

## 评分标准
请在【{dim_config['name']}】维度上，对模型B相对于模型A的表现打分（1-5分）：
- **5分**: B明显优于A（B在该维度上有显著优势）
- **4分**: B略优于A（B在该维度上稍好）
- **3分**: A和B持平（两者在该维度上表现相当）
- **2分**: A略优于B（A在该维度上稍好）
- **1分**: A明显优于B（A在该维度上有显著优势）

## 输出格式
请严格按照以下JSON格式输出（不要包含任何额外文字）：
{{
  "score": <1-5的整数>,
  "reason": "<简短的评分理由，50字以内>"
}}"""

    try:
        response = requests.post(
            JUDGE_API_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {JUDGE_API_KEY}"
            },
            json={
                "model": "/model/qwen3-235b-a22b",
                "messages": [{"role": "user", "content": judge_prompt}],
                "temperature": 0.1,
                "stream": False
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content'].strip()
            
            # 尝试解析JSON
            try:
                # 提取JSON内容
                if '```json' in content:
                    content = content.split('```json')[1].split('```')[0].strip()
                elif '```' in content:
                    content = content.split('```')[1].split('```')[0].strip()
                
                parsed = json.loads(content)
                score = parsed.get('score', 3)
                reason = parsed.get('reason', content[:100])
                
                # 验证分数范围
                if not isinstance(score, int) or score < 1 or score > 5:
                    score = 3
                
                return {"score": score, "reason": reason}
            except:
                # 如果解析失败，尝试从文本中提取
                return {"score": 3, "reason": content[:100]}
        else:
            return {"score": 3, "reason": "API调用失败"}
    except Exception as e:
        print(f"评分失败: {e}")
        return {"score": 3, "reason": f"异常: {str(e)[:50]}"}

# ==================== 主流程 ====================

print("="*70)
print("🎯 多维度Pairwise Comparison评测")
print("="*70)

# 0. 创建输出目录
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"📁 输出目录: {OUTPUT_DIR}")

# 1. 加载数据
print("\n📖 加载数据...")
with open(BASE_MODEL_FILE, 'r', encoding='utf-8') as f:
    base_data = [json.loads(line) for line in f]

with open(TUNED_MODEL_FILE, 'r', encoding='utf-8') as f:
    tuned_data = [json.loads(line) for line in f]

if NUM_SAMPLES:
    base_data = base_data[:NUM_SAMPLES]
    tuned_data = tuned_data[:NUM_SAMPLES]

print(f"✅ 加载了 {len(base_data)} 对样本")
assert len(base_data) == len(tuned_data), "数据集长度不一致！"

# 2. 构建Pairwise Comparison数据
print("\n📊 构建Pairwise对比数据...")
pairwise_data = []

for i, (base_item, tuned_item) in enumerate(zip(base_data, tuned_data)):
    pairwise_record = {
        "question_id": f"qwen3_eval_{i:04d}",
        "question": base_item['prompt'],
        "answer_a": base_item['predict'],  # 原始模型
        "answer_b": tuned_item['predict'], # 微调模型
        "reference_answer": base_item['label']
    }
    pairwise_data.append(pairwise_record)

print(f"✅ 构建了 {len(pairwise_data)} 个对比样本")

# 保存pairwise数据
print(f"\n💾 保存Pairwise数据到: {PAIRWISE_DATA_FILE}")
with open(PAIRWISE_DATA_FILE, 'w', encoding='utf-8') as f:
    for item in pairwise_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
print("✅ Pairwise数据已保存")

# 3. 多维度评分
print(f"\n🎯 开始多维度评分 (并发数={MAX_WORKERS})...")
print(f"📋 评测维度: {', '.join([v['name'] for v in EVALUATION_DIMENSIONS.values()])}")

# 准备所有评分任务
tasks = []
for i, item in enumerate(pairwise_data):
    for dim_key in EVALUATION_DIMENSIONS.keys():
        tasks.append({
            'index': i,
            'dimension': dim_key,
            'prompt': item['question'],
            'response_a': item['answer_a'],
            'response_b': item['answer_b'],
            'label': item['reference_answer']
        })

total_tasks = len(tasks)
print(f"📞 总计 {total_tasks} 个评分任务 ({len(pairwise_data)}样本 × {len(EVALUATION_DIMENSIONS)}维度)")

# 并发调用Judge API
results_map = {}  # (index, dimension) -> {"score": ..., "reason": ...}

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {
        executor.submit(
            call_judge_api_pairwise,
            task['prompt'],
            task['response_a'],
            task['response_b'],
            task['label'],
            task['dimension']
        ): task
        for task in tasks
    }
    
    for future in tqdm(as_completed(futures), total=total_tasks, desc="评分进度"):
        task = futures[future]
        try:
            result = future.result()
            key = (task['index'], task['dimension'])
            results_map[key] = result
        except Exception as e:
            print(f"\n任务失败: {e}")
            key = (task['index'], task['dimension'])
            results_map[key] = {"score": 3, "reason": "异常"}

# 4. 汇总结果并生成符合模板的输出
print("\n📊 汇总多维度评分...")

evaluation_results = []
for i, item in enumerate(pairwise_data):
    # 收集各维度评分
    criteria = {}
    weighted_score = 0
    
    for dim_key, dim_config in EVALUATION_DIMENSIONS.items():
        key = (i, dim_key)
        score_data = results_map.get(key, {"score": 3, "reason": ""})
        score = score_data['score']
        reason = score_data['reason']
        
        criteria[dim_key] = {
            "score": score,
            "weight": dim_config['weight'],
            "comment": reason
        }
        weighted_score += score * dim_config['weight']
    
    # 判断优先选择
    if weighted_score > 3.2:
        preferred = "B"
        final_reason = "模型B（微调模型）在多个维度上表现更优"
    elif weighted_score < 2.8:
        preferred = "A"
        final_reason = "模型A（原始模型）在多个维度上表现更优"
    else:
        preferred = "Tie"
        final_reason = "两个模型表现相当，各有优势"
    
    # 按照用户模板格式构建结果
    result = {
        "question_id": pairwise_data[i]['question_id'],
        "question": pairwise_data[i]['question'],
        "answer_a": pairwise_data[i]['answer_a'],
        "answer_b": pairwise_data[i]['answer_b'],
        "evaluation": {
            "criteria": criteria,
            "overall_score": round(weighted_score, 2),
            "preferred": preferred,
            "final_reason": final_reason
        },
        "judge_model": "qwen3-235b-a22b",
        "evaluation_time": datetime.now().isoformat() + "Z"
    }
    
    evaluation_results.append(result)

# 5. 保存详细评测结果（按用户模板格式）
print(f"\n💾 保存详细评测结果到: {EVALUATION_RESULTS_FILE}")
with open(EVALUATION_RESULTS_FILE, 'w', encoding='utf-8') as f:
    for result in evaluation_results:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')
print("✅ 详细评测结果已保存")

# 6. 计算汇总统计
print("\n📊 计算汇总统计...")

# 统计胜率
a_wins = sum(1 for r in evaluation_results if r['evaluation']['preferred'] == 'A')
b_wins = sum(1 for r in evaluation_results if r['evaluation']['preferred'] == 'B')
ties = sum(1 for r in evaluation_results if r['evaluation']['preferred'] == 'Tie')
total_cases = len(evaluation_results)

# 计算各维度平均分
# 将pairwise分数转换为各模型的绝对分数
model_a_scores = []
model_b_scores = []
model_a_dim_scores = {dim_key: [] for dim_key in EVALUATION_DIMENSIONS.keys()}
model_b_dim_scores = {dim_key: [] for dim_key in EVALUATION_DIMENSIONS.keys()}

for result in evaluation_results:
    overall_score = result['evaluation']['overall_score']
    
    for dim_key in EVALUATION_DIMENSIONS.keys():
        score = result['evaluation']['criteria'][dim_key]['score']
        # 将pairwise分数(1-5)转换为绝对分数
        # score=1(A强), 2(A略优), 3(平), 4(B略优), 5(B强)
        if score == 1:
            model_a_dim_scores[dim_key].append(5.0)
            model_b_dim_scores[dim_key].append(2.0)
        elif score == 2:
            model_a_dim_scores[dim_key].append(4.5)
            model_b_dim_scores[dim_key].append(3.0)
        elif score == 3:
            model_a_dim_scores[dim_key].append(4.0)
            model_b_dim_scores[dim_key].append(4.0)
        elif score == 4:
            model_a_dim_scores[dim_key].append(3.0)
            model_b_dim_scores[dim_key].append(4.5)
        else:  # score == 5
            model_a_dim_scores[dim_key].append(2.0)
            model_b_dim_scores[dim_key].append(5.0)
    
    # 整体分数
    if overall_score < 2.5:
        model_a_scores.append(5.0)
        model_b_scores.append(2.5)
    elif overall_score < 2.9:
        model_a_scores.append(4.5)
        model_b_scores.append(3.5)
    elif overall_score <= 3.1:
        model_a_scores.append(4.0)
        model_b_scores.append(4.0)
    elif overall_score <= 3.5:
        model_a_scores.append(3.5)
        model_b_scores.append(4.5)
    else:
        model_a_scores.append(2.5)
        model_b_scores.append(5.0)

# 计算平均值
model_a_per_dim_avg = {
    dim_key: round(sum(scores) / len(scores), 2)
    for dim_key, scores in model_a_dim_scores.items()
}
model_b_per_dim_avg = {
    dim_key: round(sum(scores) / len(scores), 2)
    for dim_key, scores in model_b_dim_scores.items()
}

model_a_avg_overall = round(sum(model_a_scores) / len(model_a_scores), 2)
model_b_avg_overall = round(sum(model_b_scores) / len(model_b_scores), 2)

# 按用户模板格式构建汇总报告
evaluation_summary = {
    "evaluation_summary": {
        "total_cases": total_cases,
        "model_A": {
            "model_name": "Qwen3-4B (原始模型)",
            "model_path": BASE_MODEL_FILE,
            "win_rate": round(a_wins / total_cases, 3),
            "avg_overall_score": model_a_avg_overall,
            "per_dimension_avg": model_a_per_dim_avg
        },
        "model_B": {
            "model_name": "Qwen3-4B-LoRA (微调模型)",
            "model_path": TUNED_MODEL_FILE,
            "win_rate": round(b_wins / total_cases, 3),
            "avg_overall_score": model_b_avg_overall,
            "per_dimension_avg": model_b_per_dim_avg
        },
        "tie_rate": round(ties / total_cases, 3),
        "judge_model": "qwen3-235b-a22b"
    }
}

# 保存汇总报告
print(f"\n💾 保存汇总报告到: {SUMMARY_FILE}")
with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
    json.dump(evaluation_summary, f, ensure_ascii=False, indent=2)
print("✅ 汇总报告已保存")

# 7. 打印统计结果
print("\n" + "="*70)
print("📈 评测结果汇总")
print("="*70)

print(f"\n📊 总评测样本数: {total_cases}")

print("\n【胜率统计】")
print(f"  模型A（原始）胜: {a_wins}/{total_cases} ({a_wins/total_cases*100:.1f}%)")
print(f"  模型B（微调）胜: {b_wins}/{total_cases} ({b_wins/total_cases*100:.1f}%)")
print(f"  平局:           {ties}/{total_cases} ({ties/total_cases*100:.1f}%)")

print("\n【各维度平均分对比】(1-5分制)")
print(f"{'维度':<15} {'模型A':<10} {'模型B':<10} {'差值':<10}")
print("-" * 50)
for dim_key, dim_config in EVALUATION_DIMENSIONS.items():
    a_score = model_a_per_dim_avg[dim_key]
    b_score = model_b_per_dim_avg[dim_key]
    diff = b_score - a_score
    print(f"{dim_config['name']:<15} {a_score:<10.2f} {b_score:<10.2f} {diff:+.2f}")

print("\n【整体平均分】")
print(f"  模型A: {model_a_avg_overall:.2f}")
print(f"  模型B: {model_b_avg_overall:.2f}")

if b_wins > a_wins * 1.5:
    conclusion = "✅ 微调模型(B)表现明显优于原始模型(A)"
elif b_wins > a_wins:
    conclusion = "↗ 微调模型(B)表现略优于原始模型(A)"
elif a_wins > b_wins * 1.5:
    conclusion = "❌ 原始模型(A)表现明显优于微调模型(B)"
elif a_wins > b_wins:
    conclusion = "↘ 原始模型(A)表现略优于微调模型(B)"
else:
    conclusion = "➡ 两个模型整体表现相当"

print(f"\n【结论】: {conclusion}")

print("\n" + "="*70)
print("🎉 评测完成！")
print("="*70)
print(f"\n📁 输出文件:")
print(f"  1. Pairwise数据:    {PAIRWISE_DATA_FILE}")
print(f"  2. 详细评测结果:    {EVALUATION_RESULTS_FILE}")
print(f"  3. 汇总报告:        {SUMMARY_FILE}")
