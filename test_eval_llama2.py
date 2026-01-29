#!/usr/bin/env python3
"""
简化测试版本 - 只测试模型加载，不进行生成
"""

import json
import os
import torch
import sentencepiece as spm
from eval_llama2 import Llama2Model

def main():
    print("🧪 简化测试：只加载模型，不生成文本")
    print("=" * 50)

    # 初始化模型
    model_path = "./Llama-2-7b-chat"
    llama_model = Llama2Model(model_path)

    # 加载模型
    if not llama_model.load_model():
        print("❌ 模型加载失败")
        return

    print("✅ 模型加载成功!")

    # 测试tokenizer
    test_text = "hello!"
    tokens = llama_model.encode(test_text)
    decoded = llama_model.decode(tokens)

    print(f"📝 原文: {test_text}")
    print(f"🔢 编码: {tokens}")
    print(f"📝 解码: {decoded}")

    # 测试一次前向传播（不生成）
    print("🔄 测试前向传播...")
    chat_prompt = llama_model.format_chat_prompt("hello")
    tokens = llama_model.encode(chat_prompt)
    tokens_tensor = torch.tensor([tokens], dtype=torch.long).to(llama_model.device)

    print(f"📏 输入tokens长度: {tokens_tensor.shape[1]}")

    with torch.no_grad():
        logits = llama_model.model.forward(tokens_tensor, 0)
        print(f"✅ 前向传播成功! 输出形状: {logits.shape}")

    print("🎉 所有测试通过!")

if __name__ == "__main__":
    main()