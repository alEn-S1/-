#!/usr/bin/env python3
"""
多维度Pairwise Comparison评测脚本
使用Qwen3-235B作为Judge，对比原始模型和微调模型的生成质量
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests
import pandas as pd
from typing import List, Dict, Tuple

# ==================== 配置 ====================

API_URL = "https://aimpapi.midea.com/t-aigc/aimp-qwen3-235b-a22b/v1/chat/completions"
API_KEY = "msk-c3e6e836acff79160513e563d6d288d3e24b9605349a03bb9564e9a7b3bafefe"

# 数据路径
BASE_MODEL_FILE = "/data/private/LLaMA-Factory/generated_predictions-qwen3-4b-templateqwen3.jsonl"
TUNED_MODEL_FILE = "/data/private/LLaMA-Factory/generated_predictions-qwen3-4b-lora-newtemplateqwen3.jsonl"

# 评测参数
NUM_SAMPLES = 10  # 测试数量，None表示全部
MAX_WORKERS = 10   # API并发数
OUTPUT_FILE = "/data/private/pairwise_evaluation_results.csv"

# ==================== 评测维度定义 ====================

EVALUATION_DIMENSIONS = {
    "task_completion": {
        "name": "任务完成度",
        "weight": 0.25,
        "description": "是否准确理解并完成了用户的任务要求，是否遗漏关键信息"
    },
    "language_quality": {
        "name": "语言自然度",
        "weight": 0.20,
        "description": "中文表达是否自然流畅，是否符合母语者的表达习惯，有无语病"
    },
    "correctness": {
        "name": "内容正确性",
        "weight": 0.25,
        "description": "回答的事实准确性，逻辑推理的正确性，是否有错误信息"
    },
    "explanation_depth": {
        "name": "解释深度",
        "weight": 0.15,
        "description": "思考过程是否深入，是否提供了充分的推理步骤和背景知识"
    },
    "readability": {
        "name": "可读性",
        "weight": 0.15,
        "description": "结构是否清晰，格式是否规范，是否易于理解"
    }
}

# ==================== 工具函数 ====================

def extract_think_and_answer(text: str) -> Tuple[str, str]:
    """提取thinking和answer"""
    think_pattern = r'<think>(.*?)</think>'
    match = re.search(think_pattern, text, re.DOTALL)
    
    if match:
        thinking = match.group(1).strip()
        answer = text[match.end():].strip()
        return thinking, answer
    else:
        return "", text.strip()

def call_judge_api_pairwise(
    prompt: str,
    response_a: str,
    response_b: str,
    label: str,
    dimension: str
) -> Dict:
    """
    调用Judge API进行Pairwise Comparison
    返回每个维度的对比结果
    """
    dim_config = EVALUATION_DIMENSIONS[dimension]
    
    user_content = f"""你是一个专业的AI模型评估专家。现在需要你对比两个AI模型对同一问题的回答质量。

【评测维度】: {dim_config['name']}
【维度说明】: {dim_config['description']}

【用户问题】:
{prompt}

【参考答案】:
{label[:800]}...

【模型A的回答】:
{response_a[:1500]}...

【模型B的回答】:
{response_b[:1500]}...

【评分要求】:
请从"{dim_config['name']}"这个维度，对比模型A和模型B的回答质量。

评分规则（1-5分）：
- 5分：B明显优于A（显著更好）
- 4分：B略优于A（稍好一些）
- 3分：A和B相当（难分伯仲）
- 2分：A略优于B（稍好一些）
- 1分：A明显优于B（显著更好）

请按以下JSON格式输出（只输出JSON，不要其他内容）：
{{
    "score": <1-5的整数>,
    "reason": "<简短说明理由，50字以内>"
}}"""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": "/model/qwen3-235b-a22b",
        "messages": [{"role": "user", "content": user_content}],
        "stream": False,
        "temperature": 0.1
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=90)
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content'].strip()
            
            # 提取JSON
            json_match = re.search(r'\{[^}]*"score"[^}]*\}', content, re.DOTALL)
            if json_match:
                score_data = json.loads(json_match.group())
                return {
                    "score": int(score_data.get("score", 3)),
                    "reason": score_data.get("reason", "")
                }
            else:
                # 如果没有JSON，尝试提取数字
                score_match = re.search(r'[1-5]', content)
                if score_match:
                    return {"score": int(score_match.group()), "reason": content[:100]}
                return {"score": 3, "reason": "解析失败"}
        else:
            print(f"API错误: {response.status_code}")
            return {"score": 3, "reason": "API调用失败"}
    except Exception as e:
        print(f"评分失败: {e}")
        return {"score": 3, "reason": f"异常: {str(e)[:50]}"}

# ==================== 主流程 ====================

print("="*70)
print("🎯 多维度Pairwise Comparison评测")
print("="*70)

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
    pairwise_data.append({
        'index': i,
        'prompt': base_item['prompt'],
        'response_a': base_item['predict'],  # 原始模型 (A)
        'response_b': tuned_item['predict'], # 微调模型 (B)
        'label': base_item['label']
    })

print(f"✅ 构建了 {len(pairwise_data)} 个对比样本")

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
            'prompt': item['prompt'],
            'response_a': item['response_a'],
            'response_b': item['response_b'],
            'label': item['label']
        })

total_tasks = len(tasks)
print(f"📞 总计 {total_tasks} 个评分任务 ({len(pairwise_data)}样本 × {len(EVALUATION_DIMENSIONS)}维度)")

# 并发调用Judge API
results_map = {}
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

# 4. 汇总结果
print("\n📊 汇总多维度评分...")

final_results = []
for i, item in enumerate(pairwise_data):
    result = {
        'index': i,
        'prompt': item['prompt'][:100] + "...",
    }
    
    # 各维度得分
    dimension_scores = {}
    weighted_score = 0
    
    for dim_key, dim_config in EVALUATION_DIMENSIONS.items():
        key = (i, dim_key)
        score_data = results_map.get(key, {"score": 3, "reason": ""})
        score = score_data['score']
        reason = score_data['reason']
        
        dimension_scores[dim_key] = score
        weighted_score += score * dim_config['weight']
        
        result[f"{dim_config['name']}_分数"] = score
        result[f"{dim_config['name']}_理由"] = reason
    
    # 计算加权总分和胜负
    result['加权总分'] = round(weighted_score, 2)
    
    # 判断胜负（3分为平，>3为B优，<3为A优）
    if weighted_score > 3.2:
        result['综合判断'] = "微调模型(B)更优"
    elif weighted_score < 2.8:
        result['综合判断'] = "原始模型(A)更优"
    else:
        result['综合判断'] = "两者相当"
    
    final_results.append(result)

# 5. 生成统计报告
print("\n" + "="*70)
print("📈 评测结果汇总")
print("="*70)

# 各维度平均分
print("\n【各维度平均分】(1-5分，3为持平)")
for dim_key, dim_config in EVALUATION_DIMENSIONS.items():
    avg_score = sum(r[f"{dim_config['name']}_分数"] for r in final_results) / len(final_results)
    
    if avg_score > 3.2:
        trend = "✅ B(微调)明显更优"
    elif avg_score > 3.05:
        trend = "↗ B(微调)略优"
    elif avg_score < 2.8:
        trend = "❌ A(原始)明显更优"
    elif avg_score < 2.95:
        trend = "↘ A(原始)略优"
    else:
        trend = "➡ 相当"
    
    print(f"  {dim_config['name']:<12} {avg_score:.2f}  {trend}")

# 综合胜率
print("\n【综合胜率统计】")
b_win = sum(1 for r in final_results if r['综合判断'] == "微调模型(B)更优")
tie = sum(1 for r in final_results if r['综合判断'] == "两者相当")
a_win = sum(1 for r in final_results if r['综合判断'] == "原始模型(A)更优")

print(f"  微调模型(B)胜: {b_win}/{len(final_results)} ({b_win/len(final_results)*100:.1f}%)")
print(f"  两者相当:       {tie}/{len(final_results)} ({tie/len(final_results)*100:.1f}%)")
print(f"  原始模型(A)胜: {a_win}/{len(final_results)} ({a_win/len(final_results)*100:.1f}%)")

# 加权总分分布
avg_weighted_score = sum(r['加权总分'] for r in final_results) / len(final_results)
print(f"\n【加权总分】: {avg_weighted_score:.2f} (>3表示B优，<3表示A优)")

if avg_weighted_score > 3.2:
    conclusion = "✅ 微调模型整体表现明显优于原始模型"
elif avg_weighted_score > 3.05:
    conclusion = "↗ 微调模型整体表现略优于原始模型"
elif avg_weighted_score < 2.8:
    conclusion = "❌ 原始模型整体表现明显优于微调模型"
elif avg_weighted_score < 2.95:
    conclusion = "↘ 原始模型整体表现略优于微调模型"
else:
    conclusion = "➡ 两个模型整体表现相当"

print(f"\n【结论】: {conclusion}")

# 6. 保存详细结果
df = pd.DataFrame(final_results)
df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"\n✅ 详细结果已保存: {OUTPUT_FILE}")

# 7. 导出汇总JSON
summary = {
    "样本数量": len(final_results),
    "评测维度": {k: v['name'] for k, v in EVALUATION_DIMENSIONS.items()},
    "各维度平均分": {
        EVALUATION_DIMENSIONS[k]['name']: round(
            sum(r[f"{EVALUATION_DIMENSIONS[k]['name']}_分数"] for r in final_results) / len(final_results), 
            2
        )
        for k in EVALUATION_DIMENSIONS.keys()
    },
    "综合胜率": {
        "微调模型(B)胜": f"{b_win}/{len(final_results)} ({b_win/len(final_results)*100:.1f}%)",
        "两者相当": f"{tie}/{len(final_results)} ({tie/len(final_results)*100:.1f}%)",
        "原始模型(A)胜": f"{a_win}/{len(final_results)} ({a_win/len(final_results)*100:.1f}%)"
    },
    "加权总分": round(avg_weighted_score, 2),
    "结论": conclusion
}

summary_file = OUTPUT_FILE.replace('.csv', '_summary.json')
with open(summary_file, 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"✅ 汇总报告已保存: {summary_file}")

print("\n🎉 评测完成！")
print("="*70)

