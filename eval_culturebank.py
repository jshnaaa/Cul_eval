#!/usr/bin/env python3
"""
简化版CultureBank模型加载脚本
加载完整的CultureBank模型并生成回应
"""

import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

class CultureBankEvaluator:
    def __init__(self, model_path="./CultureBank-Llama2-SFT/sft_preference_v0.3", use_4bit=True, local_base_model=None):
        """
        初始化CultureBank评估器

        Args:
            model_path: LoRA adapter路径
            use_4bit: 是否使用4bit量化
            local_base_model: 本地基座模型路径，如果为None则使用HF Hub模型
        """
        self.model_path = model_path

        # 检查是否使用本地模型
        if local_base_model and os.path.exists(local_base_model):
            self.base_model_name = local_base_model
            print(f"🏠 使用本地基座模型: {local_base_model}")
        else:
            # 使用HF格式的Llama-2-7b-chat模型
            self.base_model_name = "NousResearch/Llama-2-7b-chat-hf"  # 或 "meta-llama/Llama-2-7b-chat-hf"
            print(f"🌐 使用HF Hub模型: {self.base_model_name}")

        self.use_4bit = use_4bit
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        """
        加载CultureBank模型 (基座+LoRA adapter)
        """
        print("🎯 开始加载CultureBank模型...")

        if not os.path.exists(self.model_path):
            print(f"❌ 模型路径不存在: {self.model_path}")
            return False

        try:
            # 配置量化参数
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=self.use_4bit,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ) if self.use_4bit else None

            # 加载基座模型
            print(f"📥 加载基座模型: {self.base_model_name}")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )

            # 加载LoRA adapter
            print(f"📥 加载LoRA adapter: {self.model_path}")
            self.model = PeftModel.from_pretrained(
                base_model,
                self.model_path,
                torch_dtype=torch.bfloat16
            )

            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                trust_remote_code=True
            )

            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            print("✅ 模型加载成功!")
            return True

        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            return False

    def format_chat_prompt(self, message, system_message=None):
        """
        格式化Llama-2 chat格式的提示
        """
        if system_message is None:
            system_message = "You are a helpful, respectful and honest assistant with cultural knowledge."

        # Llama-2 chat格式
        prompt = f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{message} [/INST]"
        return prompt

    def generate_response(self, message, max_length=512, temperature=0.7, do_sample=True):
        """
        生成模型响应

        Args:
            message: 输入消息
            max_length: 最大生成长度
            temperature: 温度参数
            do_sample: 是否采样
        """
        if self.model is None or self.tokenizer is None:
            return "❌ 模型未加载"

        # 格式化输入
        prompt = self.format_chat_prompt(message)
        print(f"🔤 格式化提示: {prompt[:100]}...")

        # 编码输入
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        inputs = inputs.to(self.device)

        print(f"📏 输入tokens长度: {inputs.shape[1]}")

        # 生成响应
        start_time = time.time()

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        end_time = time.time()

        # 解码响应
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取模型生成的部分（去除输入提示）
        response = full_response[len(prompt):].strip()

        # 性能统计
        generation_time = end_time - start_time
        tokens_generated = outputs.shape[1] - inputs.shape[1]
        tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0

        print(f"⏱️  生成时间: {generation_time:.2f}秒")
        print(f"🎯 生成tokens: {tokens_generated}")
        print(f"🚀 生成速度: {tokens_per_second:.1f} tokens/秒")

        return response

def main():
    """
    主函数
    """
    print("🏛️  CultureBank模型简化版加载")
    print("=" * 60)

    # 初始化评估器
    # 如果你想使用本地Meta格式模型，取消下面一行的注释：
    # evaluator = CultureBankEvaluator(local_base_model="./Llama-2-7b-chat")
    evaluator = CultureBankEvaluator()  # 使用HF Hub模型

    # 加载模型
    if not evaluator.load_model():
        return

    # 测试用例
    test_input = "你好，介绍一下中国的传统节日"
    print(f"\n👤 用户输入: {test_input}")
    print("🤖 模型回应:")
    response = evaluator.generate_response(test_input)
    print(response)

if __name__ == "__main__":
    main()
