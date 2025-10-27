
"""
测试tokenizer是否支持enable_thinking参数
"""

from transformers import AutoTokenizer

model_path = "/data/public/models/base/Qwen/Qwen3-4B"

print("🔍 测试Qwen3-4B tokenizer的enable_thinking支持...")
print("="*60)

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    messages = [{"role": "user", "content": "1+1等于几？"}]
    
    # 测试1: enable_thinking=False
    print("\n📋 测试1: enable_thinking=False")
    print("-"*60)
    try:
        prompt_false = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False}
        )
        print("✅ 成功生成prompt (False)")
        print(f"Prompt前200字符:\n{prompt_false[:200]}")
    except Exception as e:
        print(f"❌ 失败: {e}")
    
    # 测试2: enable_thinking=True  
    print("\n📋 测试2: enable_thinking=True")
    print("-"*60)
    try:
        prompt_true = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": True}
        )
        print("✅ 成功生成prompt (True)")
        print(f"Prompt前200字符:\n{prompt_true[:200]}")
    except Exception as e:
        print(f"❌ 失败: {e}")
    
    # 对比
    if 'prompt_false' in locals() and 'prompt_true' in locals():
        print("\n🔍 对比两个prompts:")
        print("-"*60)
        if prompt_false == prompt_true:
            print("❌ 两个prompts完全一样！enable_thinking参数没有起作用！")
        else:
            print("✅ 两个prompts不同，参数有效果")
            print(f"\nFalse长度: {len(prompt_false)}")
            print(f"True长度: {len(prompt_true)}")
    
    # 测试3: 不传参数（默认行为）
    print("\n📋 测试3: 不传chat_template_kwargs")
    print("-"*60)
    prompt_default = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print(f"默认prompt前200字符:\n{prompt_default[:200]}")
    
except Exception as e:
    print(f"❌ 加载tokenizer失败: {e}")

print("\n" + "="*60)
print("测试完成")

