#!/usr/bin/env python3
"""
CultureBank模型评估脚本 - 简化版
适用于内存有限或依赖问题的情况

最小依赖:
pip install transformers peft torch

使用方法:
python eval_culturebank_simple.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")


def load_culturebank_model(model_path="./sft_preference_v0.3"):
    """
    加载CultureBank模型 - 简化版本
    """
    print("🚀 加载CultureBank模型...")

    base_model_name = "meta-llama/Llama-2-7b-chat-hf"

    try:
        # 加载基座模型（使用CPU或自动设备映射）
        print(f"📥 加载基座模型: {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        # 加载LoRA adapter
        print(f"📥 加载LoRA adapter: {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path)

        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("✅ 模型加载成功!")
        return model, tokenizer

    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        return None, None


def generate_response(model, tokenizer, message):
    """
    生成模型回应
    """
    # Llama-2 chat格式
    system_msg = "You are a helpful assistant with cultural knowledge."
    prompt = f"<s>[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n{message} [/INST]"

    # 编码输入
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # 生成回应
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 解码并提取回应
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_response[len(prompt):].strip()

    return response


def main():
    """
    主函数 - 简化版评估
    """
    print("🏛️  CultureBank模型评估器 (简化版)")
    print("=" * 50)

    # 加载模型
    model, tokenizer = load_culturebank_model()

    if model is None or tokenizer is None:
        print("❌ 无法加载模型，退出")
        return

    # 测试示例
    test_message = "你好"

    print(f"\n👤 测试输入: {test_message}")
    print("🤖 CultureBank回应:")
    print("-" * 30)

    response = generate_response(model, tokenizer, test_message)
    print(response)

    print("-" * 30)
    print("✅ 测试完成!")

    # 简单交互
    print("\n💬 简单交互测试 (输入 'quit' 退出)")
    while True:
        user_input = input("\n👤 您: ").strip()

        if user_input.lower() in ['quit', 'exit', '退出']:
            print("👋 再见!")
            break

        if user_input:
            print("🤖 CultureBank:")
            response = generate_response(model, tokenizer, user_input)
            print(response)


if __name__ == "__main__":
    main()