#!/usr/bin/env python3
"""
数据集Token分布统计脚本
用于分析Alpaca格式数据集中每个样本的token数量分布
"""

import json
import numpy as np
from collections import defaultdict
from typing import Dict, List
import argparse

# 尝试导入transformers tokenizer
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  未安装transformers库，将使用简单字符计数估算")


def simple_token_estimate(text: str) -> int:
    """简单的token估算（当transformers不可用时）"""
    # 中文字符大约1个字符=1个token，英文单词大约4个字符=1个token
    chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    other_chars = len(text) - chinese_chars
    return chinese_chars + other_chars // 4


def count_tokens(text: str, tokenizer=None) -> int:
    """计算文本的token数量"""
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    else:
        return simple_token_estimate(text)


def analyze_dataset(
    json_path: str,
    tokenizer_name: str = "Qwen/Qwen2.5-4B",
    max_samples: int = None
) -> Dict:
    """分析数据集的token分布"""
    
    print("=" * 70)
    print("数据集Token分布统计")
    print("=" * 70)
    
    # 加载tokenizer
    tokenizer = None
    if TRANSFORMERS_AVAILABLE:
        try:
            print(f"正在加载tokenizer: {tokenizer_name}")
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                trust_remote_code=True
            )
            print(f"✓ Tokenizer加载成功")
        except Exception as e:
            print(f"⚠️  Tokenizer加载失败: {e}")
            print("   将使用简单估算方法")
    else:
        print("⚠️  使用简单token估算（建议安装transformers获得精确结果）")
    
    print(f"\n正在读取数据: {json_path}")
    
    # 读取数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_samples = len(data)
    if max_samples:
        data = data[:max_samples]
        print(f"✓ 数据加载成功，总样本数: {total_samples}，分析样本数: {max_samples}")
    else:
        print(f"✓ 数据加载成功，总样本数: {total_samples}")
    
    # 统计数据
    stats = {
        'instruction_tokens': [],
        'input_tokens': [],
        'output_tokens': [],
        'total_tokens': [],
        'samples_with_empty_input': 0,
        'samples_with_nonempty_input': 0,
    }
    
    print("\n开始统计token分布...")
    for idx, sample in enumerate(data):
        if (idx + 1) % 100 == 0:
            print(f"  处理进度: {idx + 1}/{len(data)}")
        
        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '')
        output = sample.get('output', '')
        
        # 统计每个字段的token数
        inst_tokens = count_tokens(instruction, tokenizer)
        inp_tokens = count_tokens(input_text, tokenizer)
        out_tokens = count_tokens(output, tokenizer)
        total = inst_tokens + inp_tokens + out_tokens
        
        stats['instruction_tokens'].append(inst_tokens)
        stats['input_tokens'].append(inp_tokens)
        stats['output_tokens'].append(out_tokens)
        stats['total_tokens'].append(total)
        
        # 统计空input的样本
        if input_text.strip() == '':
            stats['samples_with_empty_input'] += 1
        else:
            stats['samples_with_nonempty_input'] += 1
    
    print(f"✓ 统计完成！\n")
    
    return stats, len(data), total_samples


def print_statistics(stats: Dict, analyzed_samples: int, total_samples: int):
    """打印统计结果"""
    
    print("=" * 70)
    print("统计结果")
    print("=" * 70)
    
    print(f"\n【数据集概况】")
    print(f"  总样本数: {total_samples:,}")
    print(f"  分析样本数: {analyzed_samples:,}")
    print(f"  空input样本: {stats['samples_with_empty_input']:,} ({stats['samples_with_empty_input']/analyzed_samples*100:.1f}%)")
    print(f"  非空input样本: {stats['samples_with_nonempty_input']:,} ({stats['samples_with_nonempty_input']/analyzed_samples*100:.1f}%)")
    
    # 分字段统计
    fields = [
        ('instruction_tokens', 'Instruction字段'),
        ('input_tokens', 'Input字段'),
        ('output_tokens', 'Output字段'),
        ('total_tokens', '总Token数（三字段之和）'),
    ]
    
    for field_key, field_name in fields:
        tokens = np.array(stats[field_key])
        
        print(f"\n【{field_name}】")
        print(f"  最小值: {tokens.min():,} tokens")
        print(f"  最大值: {tokens.max():,} tokens")
        print(f"  平均值: {tokens.mean():.1f} tokens")
        print(f"  中位数: {np.median(tokens):.1f} tokens")
        print(f"  标准差: {tokens.std():.1f}")
        print(f"  分位数:")
        print(f"    25%: {np.percentile(tokens, 25):.1f} tokens")
        print(f"    50%: {np.percentile(tokens, 50):.1f} tokens")
        print(f"    75%: {np.percentile(tokens, 75):.1f} tokens")
        print(f"    90%: {np.percentile(tokens, 90):.1f} tokens")
        print(f"    95%: {np.percentile(tokens, 95):.1f} tokens")
        print(f"    99%: {np.percentile(tokens, 99):.1f} tokens")
    
    # Token长度分布
    print(f"\n【总Token数分布区间】")
    total_tokens = np.array(stats['total_tokens'])
    
    ranges = [
        (0, 512, "0-512"),
        (512, 1024, "512-1024"),
        (1024, 2048, "1024-2048"),
        (2048, 4096, "2048-4096"),
        (4096, 8192, "4096-8192"),
        (8192, float('inf'), "8192+"),
    ]
    
    for min_val, max_val, label in ranges:
        count = np.sum((total_tokens >= min_val) & (total_tokens < max_val))
        percentage = count / len(total_tokens) * 100
        print(f"  {label:12s}: {count:5,} 样本 ({percentage:5.1f}%)")
    
    # 建议的cutoff_len
    print(f"\n【训练参数建议】")
    p95 = np.percentile(total_tokens, 95)
    p99 = np.percentile(total_tokens, 99)
    
    # 找到合适的2的幂次
    suggested_cutoffs = []
    for cutoff in [512, 1024, 2048, 4096, 8192]:
        coverage = np.sum(total_tokens <= cutoff) / len(total_tokens) * 100
        suggested_cutoffs.append((cutoff, coverage))
        print(f"  cutoff_len={cutoff:5d}: 覆盖 {coverage:5.1f}% 的样本")
    
    # 推荐值
    print(f"\n  推荐设置:")
    if p95 <= 2048:
        print(f"    cutoff_len=2048  (覆盖95%样本，性价比最高)")
    elif p95 <= 4096:
        print(f"    cutoff_len=4096  (覆盖95%样本)")
    else:
        print(f"    cutoff_len=8192  (数据较长，建议使用大context)")
    
    print(f"\n  注意: 超过cutoff_len的样本会被截断")


def save_detailed_report(stats: Dict, output_path: str, analyzed_samples: int):
    """保存详细的样本级别统计到JSON"""
    
    report = {
        'summary': {
            'total_samples': analyzed_samples,
            'samples_with_empty_input': stats['samples_with_empty_input'],
            'samples_with_nonempty_input': stats['samples_with_nonempty_input'],
        },
        'statistics': {}
    }
    
    for field in ['instruction_tokens', 'input_tokens', 'output_tokens', 'total_tokens']:
        tokens = np.array(stats[field])
        report['statistics'][field] = {
            'min': int(tokens.min()),
            'max': int(tokens.max()),
            'mean': float(tokens.mean()),
            'median': float(np.median(tokens)),
            'std': float(tokens.std()),
            'percentiles': {
                'p25': float(np.percentile(tokens, 25)),
                'p50': float(np.percentile(tokens, 50)),
                'p75': float(np.percentile(tokens, 75)),
                'p90': float(np.percentile(tokens, 90)),
                'p95': float(np.percentile(tokens, 95)),
                'p99': float(np.percentile(tokens, 99)),
            }
        }
    
    # 保存每个样本的详细信息
    report['samples'] = []
    for i in range(analyzed_samples):
        report['samples'].append({
            'index': i,
            'instruction_tokens': int(stats['instruction_tokens'][i]),
            'input_tokens': int(stats['input_tokens'][i]),
            'output_tokens': int(stats['output_tokens'][i]),
            'total_tokens': int(stats['total_tokens'][i]),
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 详细报告已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='统计Alpaca格式数据集的token分布')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='输入JSON文件路径'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='Qwen/Qwen2.5-4B',
        help='Tokenizer模型名称或路径（默认: Qwen/Qwen2.5-4B）'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='最大分析样本数（用于快速测试）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='保存详细报告的JSON文件路径（可选）'
    )
    
    args = parser.parse_args()
    
    # 分析数据集
    stats, analyzed_samples, total_samples = analyze_dataset(
        args.input,
        args.tokenizer,
        args.max_samples
    )
    
    # 打印统计结果
    print_statistics(stats, analyzed_samples, total_samples)
    
    # 保存详细报告
    if args.output:
        save_detailed_report(stats, args.output, analyzed_samples)
    
    print("\n" + "=" * 70)
    print("✅ 分析完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
