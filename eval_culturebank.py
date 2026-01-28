#!/usr/bin/env python3
"""
CultureBank模型评估脚本
用于加载完整的CultureBank模型并进行对话测试

依赖安装:
pip install transformers peft torch accelerate bitsandbytes

使用方法:
python eval_culturebank.py
"""

import os
import time
import torch
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, AutoPeftModelForCausalLM
import warnings
warnings.filterwarnings("ignore")


class CultureBankEvaluator:
    def __init__(self, model_path="./sft_preference_v0.3", use_4bit=True):
        """
        初始化CultureBank评估器

        Args:
            model_path: LoRA adapter路径
            use_4bit: 是否使用4bit量化
        """
        self.model_path = model_path
        self.base_model_name = "meta-llama/Llama-2-7b-chat-hf"
        self.use_4bit = use_4bit
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"🔧 设备信息: {self.device}")
        print(f"🔧 CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🔧 GPU: {torch.cuda.get_device_name()}")
            print(f"🔧 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    def get_memory_usage(self):
        """获取内存使用情况"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        cpu_memory = memory_info.rss / 1024**3  # GB

        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return f"CPU: {cpu_memory:.1f}GB, GPU: {gpu_memory:.1f}GB/{gpu_total:.1f}GB"
        else:
            return f"CPU: {cpu_memory:.1f}GB"

    def load_model_method1(self):
        """
        方法1: 使用AutoPeftModelForCausalLM直接加载
        """
        print("\n🚀 方法1: 使用AutoPeftModelForCausalLM加载...")

        try:
            # 配置量化参数
            if self.use_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                print("✅ 启用4bit量化")
            else:
                bnb_config = None

            # 加载模型
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )

            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            print("✅ 方法1加载成功!")
            print(f"📊 内存使用: {self.get_memory_usage()}")
            return True

        except Exception as e:
            print(f"❌ 方法1失败: {str(e)}")
            return False

    def load_model_method2(self):
        """
        方法2: 先加载基座模型，再加载LoRA adapter
        """
        print("\n🚀 方法2: 分步加载基座模型+LoRA...")

        try:
            # 配置量化参数
            if self.use_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                print("✅ 启用4bit量化")
            else:
                bnb_config = None

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

            print("✅ 方法2加载成功!")
            print(f"📊 内存使用: {self.get_memory_usage()}")
            return True

        except Exception as e:
            print(f"❌ 方法2失败: {str(e)}")
            return False

    def load_model(self):
        """
        尝试加载模型，优先使用方法1，失败后尝试方法2
        """
        print("=" * 60)
        print("🎯 开始加载CultureBank模型...")
        print("=" * 60)

        # 检查路径是否存在
        if not os.path.exists(self.model_path):
            print(f"❌ 模型路径不存在: {self.model_path}")
            return False

        # 尝试方法1
        if self.load_model_method1():
            return True

        # 方法1失败，尝试方法2
        print("\n🔄 方法1失败，尝试方法2...")
        if self.load_model_method2():
            return True

        print("❌ 所有加载方法都失败了!")
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
        print(f"📊 内存使用: {self.get_memory_usage()}")

        return response

    def run_evaluation(self):
        """
        运行评估测试
        """
        # 加载模型
        if not self.load_model():
            return

        print("\n" + "=" * 60)
        print("🎉 CultureBank模型加载成功！开始评估...")
        print("=" * 60)

        # 测试用例
        test_cases = [
            "你好",
            "介绍一下中国的传统节日",
            "What is the significance of the Spring Festival in Chinese culture?",
            "请解释一下儒家思想的核心理念",
            "Tell me about traditional Chinese medicine"
        ]

        for i, test_input in enumerate(test_cases, 1):
            print(f"\n{'='*20} 测试 {i}/{len(test_cases)} {'='*20}")
            print(f"👤 用户输入: {test_input}")
            print("🤖 CultureBank回应:")
            print("-" * 50)

            response = self.generate_response(test_input)
            print(response)
            print("-" * 50)

            # 等待一下，避免GPU过热
            time.sleep(1)

        print(f"\n✅ 评估完成！最终内存使用: {self.get_memory_usage()}")

    def interactive_chat(self):
        """
        交互式对话模式
        """
        if self.model is None or self.tokenizer is None:
            print("❌ 模型未加载，无法进行交互")
            return

        print("\n" + "=" * 60)
        print("💬 进入交互式对话模式 (输入 'quit' 退出)")
        print("=" * 60)

        while True:
            user_input = input("\n👤 您: ").strip()

            if user_input.lower() in ['quit', 'exit', '退出']:
                print("👋 再见！")
                break

            if not user_input:
                continue

            print("🤖 CultureBank:")
            print("-" * 40)
            response = self.generate_response(user_input)
            print(response)
            print("-" * 40)


def main():
    """
    主函数
    """
    print("🏛️  CultureBank模型评估器")
    print("基于Llama-2-7b-chat-hf + LoRA微调")
    print("=" * 60)

    # 检查依赖
    try:
        import transformers
        import peft
        import torch
        print(f"✅ transformers版本: {transformers.__version__}")
        print(f"✅ peft版本: {peft.__version__}")
        print(f"✅ torch版本: {torch.__version__}")
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请运行: pip install transformers peft torch accelerate bitsandbytes")
        return

    # 初始化评估器
    evaluator = CultureBankEvaluator()

    # 运行自动评估
    evaluator.run_evaluation()

    # 询问是否进入交互模式
    while True:
        choice = input("\n🤔 是否进入交互式对话模式? (y/n): ").strip().lower()
        if choice in ['y', 'yes', '是']:
            evaluator.interactive_chat()
            break
        elif choice in ['n', 'no', '否']:
            print("👋 评估完成，再见！")
            break
        else:
            print("请输入 y 或 n")


if __name__ == "__main__":
    main()