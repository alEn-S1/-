#!/usr/bin/env python3
"""
提取并测试Qwen3-4B的chat_template，找出enable_thinking的问题
"""

from transformers import AutoTokenizer
import json

model_path = "/data/public/models/base/Qwen/Qwen3-4B"

print("🔍 深入分析chat_template")
print("="*70)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 1. 提取完整的chat_template
chat_template = tokenizer.chat_template

print("\n📄 完整chat_template内容:")
print("-"*70)
print(chat_template)
print("-"*70)

# 2. 查找enable_thinking相关的所有行
print("\n🔍 包含'enable_thinking'的行:")
print("-"*70)
lines = chat_template.split('\n')
for i, line in enumerate(lines, 1):
    if 'enable_thinking' in line.lower():
        print(f"行{i}: {line}")

# 3. 手动测试不同的参数组合
print("\n🧪 测试不同的参数传递方式:")
print("-"*70)

messages = [{"role": "user", "content": "1+1等于几？"}]

# 测试1: 不传参数
print("\n1. 不传任何参数:")
prompt1 = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"   Prompt: {prompt1}")
print(f"   长度: {len(prompt1)}")

# 测试2: enable_thinking=False
print("\n2. enable_thinking=False:")
try:
    prompt2 = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": False}
    )
    print(f"   Prompt: {prompt2}")
    print(f"   长度: {len(prompt2)}")
    print(f"   与默认相同: {prompt1 == prompt2}")
except Exception as e:
    print(f"   ❌ 失败: {e}")

# 测试3: enable_thinking=True
print("\n3. enable_thinking=True:")
try:
    prompt3 = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": True}
    )
    print(f"   Prompt: {prompt3}")
    print(f"   长度: {len(prompt3)}")
    print(f"   与默认相同: {prompt1 == prompt3}")
except Exception as e:
    print(f"   ❌ 失败: {e}")

# 测试4: 尝试其他可能的参数名
print("\n4. 尝试其他可能的参数:")
for param_name in ['reasoning', 'thinking', 'enable_reasoning', 'use_thinking']:
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={param_name: False}
        )
        if prompt != prompt1:
            print(f"   ✅ {param_name}=False 有效！生成的prompt不同")
            print(f"      长度: {len(prompt)}")
        else:
            print(f"   ❌ {param_name}=False 无效")
    except Exception as e:
        print(f"   ❌ {param_name} 报错: {e}")

# 5. 检查是否需要在messages中传递参数
print("\n5. 尝试在messages中传递:")
try:
    messages_with_meta = [
        {"role": "user", "content": "1+1等于几？", "enable_thinking": False}
    ]
    prompt = tokenizer.apply_chat_template(
        messages_with_meta,
        tokenize=False,
        add_generation_prompt=True
    )
    if prompt != prompt1:
        print(f"   ✅ 在messages中传递enable_thinking有效！")
        print(f"      Prompt: {prompt}")
    else:
        print(f"   ❌ 在messages中传递无效")
except Exception as e:
    print(f"   ❌ 失败: {e}")

print("\n" + "="*70)
print("分析完成")

# 6. 给出建议
print("\n💡 结论和建议:")
print("-"*70)
if prompt1 == prompt2 == prompt3:
    print("❌ enable_thinking参数完全无效")
    print("\n可能的原因:")
    print("1. Qwen3-4B的chat_template逻辑有问题")
    print("2. 参数传递方式不对")
    print("3. 可能需要特定的tokenizer版本")
    print("\n建议:")
    print("✅ 方案1: 手动修改prompt，不依赖chat_template_kwargs")
    print("✅ 方案2: 升级到Qwen2.5或更新版本（如果支持）")
    print("✅ 方案3: 直接对比两个模型的thinking质量，不控制生成")

