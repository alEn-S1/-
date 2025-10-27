#!/usr/bin/env python3
"""
Chain-of-Thought模型评测脚本
评估LoRA微调后模型在生成<think>标签、thinking质量和答案质量方面的表现
"""

import json
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import argparse
from pathlib import Path
import numpy as np
from vllm import LLM, SamplingParams
from tqdm import tqdm
import pandas as pd
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class EvaluationResult:
    """单条数据的评测结果（仅enable_thinking=False）"""
    index: int
    instruction: str
    
    # 原始模型结果 (enable_thinking=False)
    base_has_think: bool
    base_think_length: int
    base_answer_length: int
    base_response: str
    
    # 微调模型结果 (enable_thinking=False)
    tuned_has_think: bool
    tuned_think_length: int
    tuned_answer_length: int
    tuned_response: str
    
    # 质量评分(后续填充，可选)
    base_think_score: float = 0.0
    base_answer_score: float = 0.0
    tuned_think_score: float = 0.0
    tuned_answer_score: float = 0.0


class CoTEvaluator:
    """Chain-of-Thought评测器"""
    
    def __init__(
        self,
        base_model_path: str,
        tuned_model_path: str,
        test_data_path: str,
        judge_api_url: str = None,
        judge_api_key: str = None,
        max_model_len: int = 2048,
        tensor_parallel_size: int = 1,
        num_samples: int = None,
        max_workers: int = 15,
    ):
        self.base_model_path = base_model_path
        self.tuned_model_path = tuned_model_path
        self.test_data_path = test_data_path
        self.judge_api_url = judge_api_url
        self.judge_api_key = judge_api_key
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.max_workers = max_workers
        
        # 加载测试数据
        print(f"📖 加载测试数据: {test_data_path}")
        with open(test_data_path, 'r', encoding='utf-8') as f:
            self.test_data = json.load(f)
        
        # 限制样本数量
        if num_samples:
            self.test_data = self.test_data[:num_samples]
            print(f"✅ 使用前 {len(self.test_data)} 条测试数据")
        else:
            print(f"✅ 加载了 {len(self.test_data)} 条测试数据")
        
    def extract_think_and_answer(self, text: str) -> Tuple[str, str]:
        """
        从生成文本中提取thinking部分和answer部分
        
        Returns:
            (thinking_content, answer_content)
        """
        # 匹配<think>标签
        think_pattern = r'<think>(.*?)</think>'
        match = re.search(think_pattern, text, re.DOTALL)
        
        if match:
            thinking = match.group(1).strip()
            # answer是</think>之后的内容
            answer = text[match.end():].strip()
            return thinking, answer
        else:
            # 没有<think>标签,全部算作answer
            return "", text.strip()
    
    def generate_responses(
        self, 
        model_path: str, 
        messages_list: List[List[Dict]],
        enable_thinking: bool = False
    ) -> List[str]:
        """
        使用vllm批量生成响应，通过tokenizer正确传递enable_thinking参数
        
        Args:
            model_path: 模型路径
            messages_list: 消息列表，格式为 [[{"role": "user", "content": "..."}], ...]
            enable_thinking: 是否启用thinking模式
            
        Returns:
            生成的响应列表
        """
        from transformers import AutoTokenizer
        
        thinking_status = "启用" if enable_thinking else "禁用"
        print(f"\n🚀 加载模型: {model_path} (enable_thinking={enable_thinking})")
        
        # 加载tokenizer
        print("📝 加载tokenizer并应用chat_template...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        
        # 使用tokenizer.apply_chat_template并传递enable_thinking参数
        prompts = []
        for messages in messages_list:
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    chat_template_kwargs={"enable_thinking": enable_thinking}  # ✅ 关键参数
                )
                prompts.append(prompt)
            except Exception as e:
                print(f"⚠️  apply_chat_template失败: {e}")
                # 降级到手动构建prompt
                user_content = messages[0]["content"]
                prompt = f"<|im_start|>system\n你是一个乐于助人的AI助手。<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
                prompts.append(prompt)
        
        print(f"✅ 生成了 {len(prompts)} 个prompts (enable_thinking={enable_thinking})")
        
        # 加载vllm模型
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            max_model_len=self.max_model_len,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=0.85,
        )
        
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=2048,
            stop=["</s>", "<|endoftext|>", "<|im_end|>"],
        )
        
        print(f"🔄 开始批量推理 ({len(prompts)} 条数据, enable_thinking={thinking_status})...")
        outputs = llm.generate(prompts, sampling_params)
        
        # 提取生成的文本
        responses = [output.outputs[0].text for output in outputs]
        
        # 释放模型和tokenizer
        del llm
        del tokenizer
        import torch
        torch.cuda.empty_cache()
        
        return responses
    
    def prepare_messages(self) -> List[List[Dict]]:
        """
        准备推理用的messages（OpenAI格式）
        返回格式: [[{"role": "user", "content": "..."}], ...]
        """
        messages_list = []
        for item in self.test_data:
            instruction = item['instruction']
            input_text = item.get('input', '')
            
            # 构建user content
            if input_text:
                user_content = f"{instruction}\n\n{input_text}"
            else:
                user_content = instruction
            
            # OpenAI messages格式
            messages = [
                {"role": "user", "content": user_content}
            ]
            
            messages_list.append(messages)
        
        return messages_list
    
    def call_judge_api(
        self, 
        instruction: str, 
        thinking: str, 
        answer: str,
        ground_truth: str,
        eval_type: str
    ) -> float:
        """
        调用Judge API评估质量
        
        Args:
            instruction: 问题
            thinking: 思维过程
            answer: 答案
            ground_truth: 标准答案
            eval_type: 评估类型 ("thinking" 或 "answer")
            
        Returns:
            评分 (0-10)
        """
        if eval_type == "thinking":
            prompt = f"""你是一个专业的思维过程评估专家。请评估以下思维过程(thinking)的质量。

评分标准(0-10分):
- 逻辑连贯性: 思维过程是否有清晰的推理链条
- 完整性: 是否涵盖了解决问题的关键步骤
- 深度: 思考是否深入,有无考虑多个角度
- 结构性: 思维组织是否清晰,易于理解

问题: {instruction}

思维过程:
{thinking if thinking else "[无思维过程]"}

标准答案供参考:
{ground_truth}

请只输出一个0-10之间的分数,不要有其他内容。"""
        else:  # answer
            prompt = f"""你是一个专业的答案质量评估专家。请评估以下答案的质量。

评分标准(0-10分):
- 准确性: 答案是否正确回答了问题
- 完整性: 答案是否全面,有无遗漏关键信息
- 清晰度: 表达是否清晰易懂
- 专业性: 是否使用恰当的专业术语和表达方式

问题: {instruction}

生成的答案:
{answer if answer else "[无答案]"}

标准答案供参考:
{ground_truth}

请只输出一个0-10之间的分数,不要有其他内容。"""
        
        # 调用API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.judge_api_key}"
        }
        
        payload = {
            "model": "/model/qwen3-235b-a22b",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": False}
        }
        
        try:
            response = requests.post(
                self.judge_api_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            # 解析流式响应
            full_response = ""
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data_str = line[6:]
                        if data_str.strip() == '[DONE]':
                            break
                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                full_response += content
                        except json.JSONDecodeError:
                            continue
            
            # 提取分数
            score_match = re.search(r'(\d+\.?\d*)', full_response.strip())
            if score_match:
                score = float(score_match.group(1))
                score = min(max(score, 0), 10)  # 限制在0-10之间
                return score
            else:
                print(f"⚠️  无法从响应中提取分数: {full_response[:100]}")
                return 5.0
                
        except Exception as e:
            print(f"⚠️  API调用失败: {e}")
            return 5.0
    
    def basic_evaluation(self) -> List[EvaluationResult]:
        """
        基础评测: 生成响应并提取thinking/answer
        只测试enable_thinking=False的情况（看是否自发生成<think>）
        """
        # 准备messages列表
        messages_list = self.prepare_messages()
        
        # 1. 原始模型 - enable_thinking=False
        print("\n" + "="*70)
        print("📊 评测原始模型 (enable_thinking=False)")
        print("="*70)
        base_responses = self.generate_responses(
            self.base_model_path, messages_list, enable_thinking=False
        )
        
        # 2. 微调模型 - enable_thinking=False
        print("\n" + "="*70)
        print("📊 评测微调模型 (enable_thinking=False)")
        print("="*70)
        tuned_responses = self.generate_responses(
            self.tuned_model_path, messages_list, enable_thinking=False
        )
        
        # 分析结果
        print("\n🔍 分析生成结果...")
        results = []
        for idx, item in enumerate(self.test_data):
            # 提取thinking和answer
            base_think, base_ans = self.extract_think_and_answer(base_responses[idx])
            tuned_think, tuned_ans = self.extract_think_and_answer(tuned_responses[idx])
            
            result = EvaluationResult(
                index=idx,
                instruction=item['instruction'],
                # Base
                base_has_think=bool(base_think),
                base_think_length=len(base_think),
                base_answer_length=len(base_ans),
                base_response=base_responses[idx],
                # Tuned
                tuned_has_think=bool(tuned_think),
                tuned_think_length=len(tuned_think),
                tuned_answer_length=len(tuned_ans),
                tuned_response=tuned_responses[idx],
            )
            results.append(result)
        
        return results
    
    def judge_quality(self, results: List[EvaluationResult]) -> List[EvaluationResult]:
        """
        使用judge API评估thinking和answer的质量 (支持并发调用)
        """
        if not self.judge_api_url or not self.judge_api_key:
            print("⚠️  未配置Judge API,跳过质量评分")
            return results
        
        print("\n" + "="*60)
        print("🎯 使用Judge API评估质量 (并发数: {})".format(self.max_workers))
        print("="*60)
        print(f"Judge API: {self.judge_api_url}")
        
        total_calls = len(results) * 4  # 每条数据4个评分(2个模型×2个维度)
        estimated_time = total_calls / self.max_workers * 2 / 60  # 估算时间
        print(f"📞 需要调用API {total_calls} 次，预计耗时 {estimated_time:.1f} 分钟...")
        
        # 准备所有评分任务
        tasks = []
        for idx, result in enumerate(results):
            instruction = result.instruction
            ground_truth = self.test_data[idx]['output']
            
            # 提取thinking和answer
            base_think, base_ans = self.extract_think_and_answer(result.base_response)
            tuned_think, tuned_ans = self.extract_think_and_answer(result.tuned_response)
            
            # 创建4个评分任务
            tasks.extend([
                # Base - thinking
                {
                    'idx': idx,
                    'type': 'base_think',
                    'instruction': instruction,
                    'thinking': base_think,
                    'answer': base_ans,
                    'ground_truth': ground_truth,
                    'eval_type': 'thinking'
                },
                # Base - answer
                {
                    'idx': idx,
                    'type': 'base_answer',
                    'instruction': instruction,
                    'thinking': base_think,
                    'answer': base_ans,
                    'ground_truth': ground_truth,
                    'eval_type': 'answer'
                },
                # Tuned - thinking
                {
                    'idx': idx,
                    'type': 'tuned_think',
                    'instruction': instruction,
                    'thinking': tuned_think,
                    'answer': tuned_ans,
                    'ground_truth': ground_truth,
                    'eval_type': 'thinking'
                },
                # Tuned - answer
                {
                    'idx': idx,
                    'type': 'tuned_answer',
                    'instruction': instruction,
                    'thinking': tuned_think,
                    'answer': tuned_ans,
                    'ground_truth': ground_truth,
                    'eval_type': 'answer'
                },
            ])
        
        # 并发调用API
        print(f"🚀 开始并发调用API (最大并发数: {self.max_workers})...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(
                    self.call_judge_api,
                    task['instruction'],
                    task['thinking'],
                    task['answer'],
                    task['ground_truth'],
                    task['eval_type']
                ): task
                for task in tasks
            }
            
            # 使用tqdm显示进度
            with tqdm(total=len(tasks), desc="评分进度") as pbar:
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        score = future.result()
                        idx = task['idx']
                        score_type = task['type']
                        
                        # 更新结果
                        if score_type == 'base_think':
                            results[idx].base_think_score = score
                        elif score_type == 'base_answer':
                            results[idx].base_answer_score = score
                        elif score_type == 'tuned_think':
                            results[idx].tuned_think_score = score
                        elif score_type == 'tuned_answer':
                            results[idx].tuned_answer_score = score
                        
                    except Exception as e:
                        print(f"\n⚠️  评分任务失败: {e}")
                    
                    pbar.update(1)
        
        print("✅ 所有评分任务完成!")
        return results
    
    def generate_report(self, results: List[EvaluationResult], output_dir: str):
        """
        生成评测报告
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*60)
        print("📝 生成评测报告")
        print("="*60)
        
        # 1. 统计数据
        n = len(results)
        
        # <think>标签生成率统计
        base_think_rate = sum(r.base_has_think for r in results) / n * 100
        tuned_think_rate = sum(r.tuned_has_think for r in results) / n * 100
        
        # 平均thinking长度
        def safe_mean(values):
            filtered = [v for v in values if v > 0]
            return np.mean(filtered) if filtered else 0
        
        base_avg_len = safe_mean([r.base_think_length for r in results])
        tuned_avg_len = safe_mean([r.tuned_think_length for r in results])
        
        # 检查是否有评分
        has_scores = results[0].base_think_score > 0
        
        # 2. 生成统计报告
        report = {
            "总体统计": {
                "测试样本数": n,
                "评测模式": "enable_thinking=False（测试自发生成能力）",
                "原始模型(Qwen3-4B)": {
                    "<think>标签生成率": f"{base_think_rate:.1f}%",
                    "平均thinking长度": f"{base_avg_len:.0f} 字符",
                },
                "微调模型(Qwen3-4B-LoRA)": {
                    "<think>标签生成率": f"{tuned_think_rate:.1f}%",
                    "平均thinking长度": f"{tuned_avg_len:.0f} 字符",
                },
                "关键发现": {
                    "生成率提升": f"{tuned_think_rate - base_think_rate:+.1f}%",
                    "结论": "✅ 微调成功！学会了自发生成CoT" if tuned_think_rate > 80 else "⚠️ 微调效果不明显，生成率偏低"
                }
            }
        }
        
        if has_scores:
            # 计算平均评分
            report["质量评分(0-10分)"] = {
                "原始模型": {
                    "Thinking质量": f"{np.mean([r.base_think_score for r in results]):.2f}",
                    "Answer质量": f"{np.mean([r.base_answer_score for r in results]):.2f}",
                },
                "微调模型": {
                    "Thinking质量": f"{np.mean([r.tuned_think_score for r in results]):.2f}",
                    "Answer质量": f"{np.mean([r.tuned_answer_score for r in results]):.2f}",
                },
                "质量提升": {
                    "Thinking": f"{np.mean([r.tuned_think_score for r in results]) - np.mean([r.base_think_score for r in results]):+.2f}",
                    "Answer": f"{np.mean([r.tuned_answer_score for r in results]) - np.mean([r.base_answer_score for r in results]):+.2f}",
                }
            }
        
        # 3. 保存JSON报告
        report_json_path = output_path / "evaluation_report.json"
        with open(report_json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"✅ JSON报告已保存: {report_json_path}")
        
        # 4. 保存详细结果到CSV
        df_data = []
        for r in results:
            row = {
                "序号": r.index,
                "问题": r.instruction[:50] + "..." if len(r.instruction) > 50 else r.instruction,
                "Base_有<think>": "✓" if r.base_has_think else "✗",
                "Base_Think长度": r.base_think_length,
                "Tuned_有<think>": "✓" if r.tuned_has_think else "✗",
                "Tuned_Think长度": r.tuned_think_length,
            }
            
            if has_scores:
                row.update({
                    "Base_Think评分": f"{r.base_think_score:.1f}",
                    "Base_Answer评分": f"{r.base_answer_score:.1f}",
                    "Tuned_Think评分": f"{r.tuned_think_score:.1f}",
                    "Tuned_Answer评分": f"{r.tuned_answer_score:.1f}",
                })
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        csv_path = output_path / "detailed_results.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✅ 详细结果已保存: {csv_path}")
        
        # 5. 保存完整的响应
        full_results_path = output_path / "full_responses.jsonl"
        with open(full_results_path, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + '\n')
        print(f"✅ 完整响应已保存: {full_results_path}")
        
        # 6. 打印控制台报告
        print("\n" + "="*60)
        print("📊 评测结果汇总")
        print("="*60)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print("\n🎉 评测完成!")
        
    def run(self, output_dir: str = "./evaluation_results"):
        """
        运行完整评测流程
        """
        # 1. 基础评测
        results = self.basic_evaluation()
        
        # 2. 质量评分
        if self.judge_api_url and self.judge_api_key:
            results = self.judge_quality(results)
        
        # 3. 生成报告
        self.generate_report(results, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Chain-of-Thought模型评测脚本")
    parser.add_argument(
        "--base_model",
        type=str,
        default="/data/public/models/base/Qwen/Qwen3-4B",
        help="原始模型路径"
    )
    parser.add_argument(
        "--tuned_model",
        type=str,
        default="/data/private/models/qwen3_4b_merged1",
        help="微调后模型路径"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="qwen3_sft_alpaca_format1101-1200.json",
        help="测试数据路径"
    )
    parser.add_argument(
        "--judge_api_url",
        type=str,
        default=None,
        help="Judge API URL"
    )
    parser.add_argument(
        "--judge_api_key",
        type=str,
        default=None,
        help="Judge API Key"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="评测结果输出目录"
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=2048,
        help="模型最大长度"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="张量并行大小(多卡推理)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="使用前N条数据测试(None=全部)"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=15,
        help="Judge API并发调用数(默认15)"
    )
    
    args = parser.parse_args()
    
    # 创建评测器
    evaluator = CoTEvaluator(
        base_model_path=args.base_model,
        tuned_model_path=args.tuned_model,
        test_data_path=args.test_data,
        judge_api_url=args.judge_api_url,
        judge_api_key=args.judge_api_key,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        num_samples=args.num_samples,
        max_workers=args.max_workers,
    )
    
    # 运行评测
    evaluator.run(output_dir=args.output_dir)


if __name__ == "__main__":
    main()

