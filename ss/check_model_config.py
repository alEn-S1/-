#!/usr/bin/env python3
"""
检查Qwen3-4B模型配置，查找reasoning/thinking相关设置
"""

import json
from pathlib import Path

model_path = "/data/public/models/base/Qwen/Qwen3-4B"

print("🔍 检查Qwen3-4B模型配置")
print("="*70)

# 1. 检查config.json
config_file = Path(model_path) / "config.json"
if config_file.exists():
    print("\n📄 config.json:")
    print("-"*70)
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # 查找reasoning/thinking相关配置
    keywords = ['reasoning', 'thinking', 'enable', 'chat_template']
    found_keys = []
    
    for key, value in config.items():
        key_lower = key.lower()
        if any(kw in key_lower for kw in keywords):
            found_keys.append((key, value))
            print(f"  {key}: {value}")
    
    if not found_keys:
        print("  ❌ 没有找到reasoning/thinking相关配置")
    
    # 打印完整配置（前20个键）
    print("\n📋 前20个配置项:")
    print("-"*70)
    for i, (key, value) in enumerate(list(config.items())[:20]):
        print(f"  {key}: {value}")

# 2. 检查tokenizer_config.json
tokenizer_config_file = Path(model_path) / "tokenizer_config.json"
if tokenizer_config_file.exists():
    print("\n📄 tokenizer_config.json:")
    print("-"*70)
    with open(tokenizer_config_file, 'r') as f:
        tokenizer_config = json.load(f)
    
    # 查找chat_template
    if 'chat_template' in tokenizer_config:
        chat_template = tokenizer_config['chat_template']
        print(f"  chat_template长度: {len(chat_template)} 字符")
        
        # 检查是否包含thinking相关内容
        if 'thinking' in chat_template.lower():
            print("  ✅ chat_template包含'thinking'关键词")
            # 找出相关行
            lines = chat_template.split('\n')
            for i, line in enumerate(lines):
                if 'thinking' in line.lower():
                    print(f"    行{i}: {line[:100]}...")
        else:
            print("  ❌ chat_template不包含'thinking'关键词")
    
    # 查找其他相关配置
    for key, value in tokenizer_config.items():
        key_lower = key.lower()
        if any(kw in key_lower for kw in ['reasoning', 'thinking', 'enable']):
            print(f"  {key}: {value}")

# 3. 检查generation_config.json
gen_config_file = Path(model_path) / "generation_config.json"
if gen_config_file.exists():
    print("\n📄 generation_config.json:")
    print("-"*70)
    with open(gen_config_file, 'r') as f:
        gen_config = json.load(f)
    
    for key, value in gen_config.items():
        print(f"  {key}: {value}")

# 4. 测试tokenizer的实际行为
print("\n🧪 测试tokenizer实际行为:")
print("-"*70)

try:
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 检查tokenizer属性
    attrs = ['chat_template', 'eos_token', 'bos_token', 'pad_token']
    for attr in attrs:
        if hasattr(tokenizer, attr):
            value = getattr(tokenizer, attr)
            if isinstance(value, str) and len(value) > 100:
                print(f"  {attr}: {value[:100]}...")
            else:
                print(f"  {attr}: {value}")
    
    # 尝试不同的apply_chat_template调用
    messages = [{"role": "user", "content": "测试"}]
    
    print("\n  测试1: 默认")
    prompt1 = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"    长度: {len(prompt1)}")
    
    print("\n  测试2: enable_thinking=False")
    try:
        prompt2 = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False}
        )
        print(f"    长度: {len(prompt2)}")
        print(f"    与默认相同: {prompt1 == prompt2}")
    except Exception as e:
        print(f"    ❌ 失败: {e}")
    
    print("\n  测试3: enable_thinking=True")
    try:
        prompt3 = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": True}
        )
        print(f"    长度: {len(prompt3)}")
        print(f"    与默认相同: {prompt1 == prompt3}")
    except Exception as e:
        print(f"    ❌ 失败: {e}")
    
except Exception as e:
    print(f"  ❌ 加载tokenizer失败: {e}")

print("\n" + "="*70)
print("检查完成")

