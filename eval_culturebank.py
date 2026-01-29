#!/usr/bin/env python3
"""
CultureBank模型加载脚本 - 直接加载Meta格式基座模型 + LoRA adapter
基于eval_llama2.py的成功加载方式
"""

import json
import os
import torch
import torch.nn.functional as F
import sentencepiece as spm
from eval_llama2 import Llama2Model, Transformer, ModelArgs
from typing import Dict, Any

# 尝试导入safetensors，如果不存在则使用备选方案
try:
    from safetensors import safe_open
    HAS_SAFETENSORS = True
except ImportError:
    print("⚠️  safetensors未安装，将只支持.bin格式的adapter")
    HAS_SAFETENSORS = False


class CultureBankModel:
    def __init__(self, base_model_path="./Llama-2-7b-chat", adapter_path="./CultureBank-Llama2-SFT/sft_preference_v0.3"):
        """
        初始化CultureBank模型

        Args:
            base_model_path: Meta格式基座模型路径
            adapter_path: LoRA adapter路径
        """
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 基座模型组件
        self.base_model = None
        self.tokenizer = None

        # adapter权重
        self.adapter_weights = {}

    def load_base_model(self):
        """加载Meta格式的基座模型"""
        print("🦙 加载Meta格式基座模型...")

        # 复用eval_llama2.py的加载逻辑
        llama_loader = Llama2Model(self.base_model_path)

        if not llama_loader.load_model():
            print("❌ 基座模型加载失败")
            return False

        # 获取加载好的模型和tokenizer
        self.base_model = llama_loader.model
        self.tokenizer = llama_loader.tokenizer

        print("✅ 基座模型加载成功!")
        return True

    def load_adapter_weights(self):
        """加载LoRA adapter权重"""
        print("📦 加载LoRA adapter权重...")

        if not os.path.exists(self.adapter_path):
            print(f"❌ Adapter路径不存在: {self.adapter_path}")
            return False

        try:
            # 查找adapter权重文件
            adapter_files = []
            for file in os.listdir(self.adapter_path):
                if file.endswith('.safetensors') or file.endswith('.bin'):
                    adapter_files.append(file)

            if not adapter_files:
                print(f"❌ 在{self.adapter_path}中未找到权重文件(.safetensors或.bin)")
                return False

            print(f"📁 找到adapter文件: {adapter_files}")

            # 加载adapter权重
            for file in adapter_files:
                file_path = os.path.join(self.adapter_path, file)

                if file.endswith('.safetensors') and HAS_SAFETENSORS:
                    # 加载safetensors格式
                    with safe_open(file_path, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            self.adapter_weights[key] = f.get_tensor(key)
                            print(f"  📋 加载权重: {key}, 形状: {self.adapter_weights[key].shape}")
                elif file.endswith('.safetensors') and not HAS_SAFETENSORS:
                    print(f"  ⚠️  跳过safetensors文件（需要安装safetensors包）: {file}")
                elif file.endswith('.bin'):
                    # 加载pytorch格式
                    weights = torch.load(file_path, map_location="cpu")
                    for key, value in weights.items():
                        self.adapter_weights[key] = value
                        print(f"  📋 加载权重: {key}, 形状: {value.shape}")

            print(f"✅ 成功加载{len(self.adapter_weights)}个adapter权重")
            return True

        except Exception as e:
            print(f"❌ 加载adapter权重失败: {str(e)}")
            return False

    def apply_lora_weights(self):
        """将LoRA权重应用到基座模型"""
        print("🔧 应用LoRA权重到基座模型...")

        try:
            # 获取基座模型的状态字典
            base_state_dict = self.base_model.state_dict()

            # 分组LoRA权重：将lora_A和lora_B配对
            lora_pairs = {}
            for key in self.adapter_weights.keys():
                if 'lora_A' in key:
                    base_name = key.replace('.lora_A.weight', '')
                    lora_pairs[base_name] = lora_pairs.get(base_name, {})
                    lora_pairs[base_name]['A'] = key
                elif 'lora_B' in key:
                    base_name = key.replace('.lora_B.weight', '')
                    lora_pairs[base_name] = lora_pairs.get(base_name, {})
                    lora_pairs[base_name]['B'] = key

            print(f"📊 找到{len(lora_pairs)}个LoRA权重对")

            # 应用LoRA权重：W_new = W_base + lora_B @ lora_A
            applied_count = 0
            for base_name, pair in lora_pairs.items():
                if 'A' in pair and 'B' in pair:
                    # 构建基座模型权重名称
                    base_key = base_name + '.weight'

                    if base_key in base_state_dict:
                        try:
                            # 获取LoRA权重
                            lora_A = self.adapter_weights[pair['A']].to(self.device)
                            lora_B = self.adapter_weights[pair['B']].to(self.device)

                            # 计算LoRA增量：delta_W = lora_B @ lora_A
                            delta_W = torch.matmul(lora_B, lora_A)

                            # 获取基座权重
                            base_weight = base_state_dict[base_key].to(self.device)

                            # 合并权重：W_new = W_base + delta_W
                            new_weight = base_weight + delta_W

                            # 更新模型权重
                            base_state_dict[base_key].copy_(new_weight)

                            print(f"  ✅ 应用LoRA: {base_name}")
                            applied_count += 1

                        except Exception as e:
                            print(f"  ❌ 应用LoRA失败 {base_name}: {str(e)}")
                    else:
                        print(f"  ⚠️  未找到对应基座权重: {base_key}")

            print(f"✅ 成功应用{applied_count}个LoRA权重对")
            return True

        except Exception as e:
            print(f"❌ 应用LoRA权重失败: {str(e)}")
            return False

    def load_model(self):
        """加载完整的CultureBank模型"""
        print("🎯 开始加载CultureBank模型...")

        # 1. 加载基座模型
        if not self.load_base_model():
            return False

        # 2. 加载adapter权重
        if not self.load_adapter_weights():
            return False

        # 3. 应用LoRA权重（简化版本）
        if not self.apply_lora_weights():
            return False

        print("🎉 CultureBank模型加载完成!")
        return True

    def format_chat_prompt(self, message: str, system_message: str = None):
        """格式化Llama-2 chat格式"""
        if system_message is None:
            system_message = "You are CultureBank, a helpful assistant with deep cultural knowledge from around the world."

        prompt = f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{message} [/INST]"
        return prompt

    def encode(self, text: str):
        """编码文本为token"""
        return self.tokenizer.encode(text, out_type=int)

    def decode(self, tokens):
        """解码token为文本"""
        return self.tokenizer.decode(tokens)

    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7):
        """生成文本响应"""
        if self.base_model is None or self.tokenizer is None:
            return "❌ 模型未加载"

        # 编码输入
        tokens = self.encode(prompt)
        tokens = torch.tensor([tokens], dtype=torch.long).to(self.device)

        print(f"📏 输入tokens长度: {tokens.shape[1]}")

        generated_tokens = []

        # 使用与eval_llama2.py相同的生成逻辑
        with torch.no_grad():
            current_tokens = tokens.clone()

            for i in range(max_tokens):
                try:
                    # 前向传播
                    logits = self.base_model.forward(current_tokens, 0)

                    # 获取最后一个位置的logits
                    last_logits = logits[0, -1, :]

                    # 应用temperature
                    if temperature > 0:
                        last_logits = last_logits / temperature
                        probs = F.softmax(last_logits, dim=-1)
                        next_token_id = torch.multinomial(probs, num_samples=1).item()
                    else:
                        next_token_id = torch.argmax(last_logits, dim=-1).item()

                    # 检查是否为结束token
                    if next_token_id == 2:  # </s> token
                        break

                    generated_tokens.append(next_token_id)

                    # 创建新的token并拼接到序列
                    next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=self.device)
                    current_tokens = torch.cat([current_tokens, next_token_tensor], dim=1)

                except Exception as e:
                    print(f"❌ 生成步骤 {i+1} 失败: {str(e)}")
                    break

        # 解码生成的文本
        if generated_tokens:
            generated_text = self.decode(generated_tokens)
            return generated_text.strip()
        else:
            return ""


def main():
    """主函数"""
    print("🏛️  CultureBank模型加载测试")
    print("=" * 60)

    # 初始化CultureBank模型
    culture_model = CultureBankModel()

    # 加载模型
    if not culture_model.load_model():
        print("❌ CultureBank模型加载失败，退出程序")
        return

    # 测试用例
    test_message = "Tell me about Chinese New Year traditions."
    print(f"\n👤 用户输入: {test_message}")

    # 格式化为chat格式
    chat_prompt = culture_model.format_chat_prompt(test_message)
    print(f"🔤 格式化提示: {chat_prompt[:100]}...")

    # 生成回复
    print("🤖 CultureBank回应:")
    response = culture_model.generate(chat_prompt, max_tokens=100, temperature=0.7)
    print(response)

    print("\n🎉 测试完成!")


if __name__ == "__main__":
    main()