#!/usr/bin/env python3
"""
CultureBank模型评测脚本 - 支持多数据集评测
基于eval_llama2.py的成功加载方式 + eval_spa.py的评测逻辑
"""

import json
import os
import re
import argparse
import torch
import torch.nn.functional as F
import sentencepiece as spm
from eval_llama2 import Llama2Model, Transformer, ModelArgs
from typing import Dict, Any, List
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from datetime import datetime

# 尝试导入safetensors，如果不存在则使用备选方案
try:
    from safetensors import safe_open
    HAS_SAFETENSORS = True
except ImportError:
    print("⚠️  safetensors未安装，将只支持.bin格式的adapter")
    HAS_SAFETENSORS = False


class CultureBankEvaluator:
    def __init__(self, base_model_path="./Llama-2-7b-chat", adapter_path="./CultureBank-Llama2-SFT/sft_preference_v0.3"):
        """
        初始化CultureBank评测器

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

    def convert_lora_to_base_key(self, lora_key: str) -> str:
        """
        将LoRA权重名称转换为基座模型权重名称

        Args:
            lora_key: LoRA权重名称，如 'base_model.model.model.layers.0.self_attn.q_proj'

        Returns:
            基座模型权重名称，如 'layers.0.attention.wq.weight'
        """
        # 移除前缀
        if lora_key.startswith('base_model.model.'):
            clean_name = lora_key.replace('base_model.model.', '')
        else:
            clean_name = lora_key

        # HF格式到Meta Llama格式的映射
        # model.layers.X.self_attn.Y_proj -> layers.X.attention.wY
        if 'model.layers.' in clean_name and 'self_attn.' in clean_name:
            import re
            # 使用正则表达式提取层号和投影类型
            pattern = r'model\.layers\.(\d+)\.self_attn\.([qkvo])_proj'
            match = re.search(pattern, clean_name)

            if match:
                layer_idx = match.group(1)
                proj_type = match.group(2)

                # 映射投影类型
                proj_mapping = {
                    'q': 'wq',
                    'k': 'wk',
                    'v': 'wv',
                    'o': 'wo'
                }

                if proj_type in proj_mapping:
                    return f"layers.{layer_idx}.attention.{proj_mapping[proj_type]}.weight"

        # 如果没有匹配到模式，返回原始名称加.weight
        return clean_name + '.weight'

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
            # 查找adapter权重文件，排除非权重文件
            adapter_files = []
            excluded_files = ['training_args.bin', 'optimizer.bin', 'scheduler.bin', 'rng_state.pth']

            for file in os.listdir(self.adapter_path):
                if file.endswith('.safetensors'):
                    if 'adapter' in file.lower() or 'lora' in file.lower():
                        adapter_files.append(file)
                elif file.endswith('.bin') and file not in excluded_files:
                    # 排除已知的非权重文件
                    if 'adapter' in file.lower() or 'lora' in file.lower():
                        adapter_files.append(file)

            if not adapter_files:
                print(f"❌ 在{self.adapter_path}中未找到权重文件(.safetensors或.bin)")
                return False

            print(f"📁 找到adapter文件: {adapter_files}")

            # 加载adapter权重
            for file in adapter_files:
                file_path = os.path.join(self.adapter_path, file)

                try:
                    if file.endswith('.safetensors') and HAS_SAFETENSORS:
                        # 加载safetensors格式
                        with safe_open(file_path, framework="pt", device="cpu") as f:
                            for key in f.keys():
                                self.adapter_weights[key] = f.get_tensor(key)
                    elif file.endswith('.bin'):
                        # 加载pytorch格式
                        weights = torch.load(file_path, map_location="cpu")

                        # 检查是否是字典格式的权重文件
                        if isinstance(weights, dict) and hasattr(weights, 'items'):
                            for key, value in weights.items():
                                if isinstance(value, torch.Tensor):
                                    self.adapter_weights[key] = value

                except Exception as e:
                    print(f"  ❌ 加载文件失败 {file}: {str(e)}")
                    continue

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
                    # 转换LoRA权重名称到基座模型权重名称
                    base_key = self.convert_lora_to_base_key(base_name)

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

                            applied_count += 1

                        except Exception as e:
                            print(f"  ❌ 应用LoRA失败 {base_name}: {str(e)}")

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

        # 3. 应用LoRA权重
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

    def generate_response(self, instruction: str, max_new_tokens: int = 1024, temperature: float = 0.0):
        """
        生成模型响应 - 限制输出长度，优先数字

        Args:
            instruction: 输入指令
            max_new_tokens: 最大新生成token数量
            temperature: 温度参数

        Returns:
            模型生成的回复
        """
        if self.base_model is None or self.tokenizer is None:
            return "❌ 模型未加载"

        # 格式化输入
        prompt = self.format_chat_prompt(instruction)

        # 编码输入
        tokens = self.encode(prompt)

        # 限制输入长度，避免过长的context
        if len(tokens) > 2000:
            tokens = tokens[-2000:]  # 只保留最后2000个token

        tokens = torch.tensor([tokens], dtype=torch.long).to(self.device)

        generated_tokens = []

        # 恢复工作的生成逻辑，但添加严格控制
        with torch.no_grad():
            current_tokens = tokens.clone()

            for i in range(max_new_tokens):
                try:
                    # 前向传播
                    logits = self.base_model.forward(current_tokens, 0)

                    # 获取最后一个位置的logits
                    last_logits = logits[0, -1, :]

                    # 贪婪解码
                    next_token_id = torch.argmax(last_logits, dim=-1).item()

                    # 检查是否为结束token
                    if next_token_id == 2:  # </s> token
                        break

                    generated_tokens.append(next_token_id)

                    # 创建新的token并拼接到序列
                    next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=self.device)
                    current_tokens = torch.cat([current_tokens, next_token_tensor], dim=1)

                except Exception as e:
                    break

        # 解码生成的文本
        if generated_tokens:
            generated_text = self.decode(generated_tokens)
            return generated_text.strip()
        else:
            return ""

    def extract_answer(self, response: str) -> str:
        """
        从模型回复中提取答案

        Args:
            response: 模型生成的回复

        Returns:
            提取的答案（1-4的数字），如果提取失败返回空字符串
        """
        # 清理响应文本
        response = response.strip()

        # 尝试多种模式提取答案
        patterns = [
            r'\b([1-4])\b',  # 匹配单独的数字1-4
            r'答案[是为]?\s*([1-4])',  # 匹配"答案是X"
            r'选择\s*([1-4])',  # 匹配"选择X"
            r'([1-4])\s*[.。]',  # 匹配"X."
            r'选项\s*([1-4])',  # 匹配"选项X"
        ]

        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1)

        # 如果都没匹配到，返回响应的第一个字符（如果是1-4）
        if len(response) > 0 and response[0] in '1234':
            return response[0]

        return ""

    def load_dataset(self, data_file: str) -> List[Dict]:
        """
        加载数据集

        Args:
            data_file: 数据文件路径

        Returns:
            数据集列表
        """
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                dataset = json.load(f)

            print(f"✅ 成功加载数据集: {data_file}")
            print(f"📊 数据集大小: {len(dataset)} 条")
            return dataset

        except Exception as e:
            print(f"❌ 加载数据集失败: {str(e)}")
            return []

    def calculate_metrics(self, predictions: List[str], ground_truths: List[str]) -> Dict:
        """
        计算评估指标

        Args:
            predictions: 预测结果列表
            ground_truths: 真实标签列表

        Returns:
            评估指标字典
        """
        # 过滤掉空预测
        filtered_predictions = []
        filtered_ground_truths = []

        for pred, truth in zip(predictions, ground_truths):
            if pred:  # 只考虑有预测结果的样本
                filtered_predictions.append(pred)
                filtered_ground_truths.append(truth)

        if not filtered_predictions:
            return {}

        # 计算基本指标
        accuracy = accuracy_score(filtered_ground_truths, filtered_predictions)

        # 计算精确率、召回率、F1
        precision, recall, f1, support = precision_recall_fscore_support(
            filtered_ground_truths, filtered_predictions, average='macro', zero_division=0
        )

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_macro': f1,
            'total_samples': len(predictions),
            'answered_samples': len(filtered_predictions),
            'answer_extraction_rate': len(filtered_predictions) / len(predictions) if predictions else 0
        }

    def evaluate_dataset(self, data_file: str, dataset_tag: str, output_dir: str) -> Dict:
        """
        评估数据集

        Args:
            data_file: 数据文件路径
            dataset_tag: 数据集标签
            output_dir: 输出目录

        Returns:
            评估结果字典
        """
        # 加载数据集
        dataset = self.load_dataset(data_file)
        if not dataset:
            return {}

        # 准备结果存储
        results = []
        predictions = []
        ground_truths = []

        print(f"\n🚀 开始评估 {dataset_tag} 数据集...")
        print("=" * 60)

        # 批量处理数据
        for i, item in enumerate(tqdm(dataset, desc="评估进度")):
            instruction = item.get('instruction', '')
            expected_output = item.get('output', '').strip()

            if not instruction:
                print(f"⚠️  第 {i+1} 条数据缺少instruction字段，跳过")
                continue

            # 生成模型回复
            model_response = self.generate_response(instruction)

            # 提取答案
            extracted_answer = self.extract_answer(model_response)

            # 判断是否一致
            is_correct = extracted_answer == expected_output

            # 打印前三条样本的详细信息
            if i < 3:
                print(f"\n=== 样本 {i+1} ===")
                print(f"问题 (instruction): {instruction}")
                print(f"期望答案 (output): {expected_output}")
                print(f"模型生成回答: {model_response}")
                print(f"提取答案: {extracted_answer}")
                print(f"是否正确: {is_correct}")
                print("=" * 60)

            # 记录结果
            result_item = {
                "question_id": i + 1,
                "instruction": instruction,
                "expected_answer": expected_output,
                "model_response": model_response,
                "extracted_answer": extracted_answer,
                "is_correct": is_correct
            }

            results.append(result_item)
            predictions.append(extracted_answer)
            ground_truths.append(expected_output)

        # 计算评估指标
        metrics = self.calculate_metrics(predictions, ground_truths)

        # 组织最终结果
        final_results = {
            "dataset_info": {
                "dataset_tag": dataset_tag,
                "data_file": data_file,
                "total_questions": len(dataset),
                "answered_questions": metrics.get('answered_samples', 0),
            },
            "performance_metrics": {
                "accuracy": metrics.get('accuracy', 0),
                "precision": metrics.get('precision', 0),
                "recall": metrics.get('recall', 0),
                "f1_macro": metrics.get('f1_macro', 0),
                "answer_extraction_rate": metrics.get('answer_extraction_rate', 0),
            },
            "statistics": {
                "overall_accuracy": metrics.get('accuracy', 0),
                "answer_extraction_rate": metrics.get('answer_extraction_rate', 0),
            },
            "timestamp": datetime.now().isoformat()
        }

        # 保存详细结果
        answers_file = os.path.join(output_dir, "generated_answers.json")
        with open(answers_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 保存评估结果
        eval_file = os.path.join(output_dir, "eval_results.json")
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)

        print(f"\n✅ 评估完成!")
        print(f"📊 整体准确率: {final_results['performance_metrics']['accuracy']:.4f}")
        print(f"📊 答案提取率: {final_results['performance_metrics']['answer_extraction_rate']:.4f}")
        print(f"📁 结果已保存到: {output_dir}")

        return final_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="CultureBank模型评估脚本")
    parser.add_argument("--dataset_id", type=int, required=True,
                       help="数据集ID (2=CulturalBench, 3=normad, 4=cultureLLM, 5=cultureAtlas)")
    parser.add_argument("--data_file", type=str, required=True,
                       help="数据集文件路径")
    parser.add_argument("--dataset_tag", type=str, required=True,
                       help="数据集标签")
    parser.add_argument("--output_dir", type=str, default="./",
                       help="输出目录")

    args = parser.parse_args()

    print("🏛️  CultureBank模型评估器")
    print(f"📊 数据集: {args.dataset_tag} (ID: {args.dataset_id})")
    print(f"📁 数据文件: {args.data_file}")
    print("=" * 60)

    # 检查数据文件是否存在
    if not os.path.exists(args.data_file):
        print(f"❌ 数据文件不存在: {args.data_file}")
        return

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化评估器
    evaluator = CultureBankEvaluator()

    # 加载模型
    if not evaluator.load_model():
        print("❌ CultureBank模型加载失败")
        return

    # 执行评估
    evaluator.evaluate_dataset(args.data_file, args.dataset_tag, args.output_dir)

    print("\n🎉 评估任务完成!")


if __name__ == "__main__":
    main()