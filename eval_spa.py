#!/usr/bin/env python3
"""
CultureSPA模型评估脚本
支持对多个文化知识数据集进行批量评估

使用方法:
python eval_spa.py --dataset_id 2 --data_file /path/to/dataset.json --dataset_tag CulturalBench

依赖:
pip install transformers torch tqdm scikit-learn
"""

import json
import argparse
import re
import time
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import warnings
warnings.filterwarnings("ignore")


class CultureSPAEvaluator:
    def __init__(self, model_path: str = None, device: str = "auto"):
        """
        初始化CultureSPA评估器

        Args:
            model_path: 模型路径（如果为None，使用默认路径）
            device: 设备设置
        """
        self.model_path = model_path
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None

        print(f"使用设备: {self.device}")

    def load_model(self):
        """加载CultureSPA模型和分词器"""
        try:
            print("正在加载分词器...")
            # 根据实际模型路径调整，这里使用通用加载方式
            if self.model_path:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            else:
                # 如果没有指定路径，尝试从环境变量或默认位置加载
                model_path = os.environ.get('CULTURESPA_MODEL_PATH', './culturespa_model')
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                self.model_path = model_path

            print("正在加载模型...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                dtype=torch.float16,  # 使用dtype替代torch_dtype
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            print("模型已通过 device_map='auto' 加载。")
            print("模型加载完成！")
            return True

        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            return False

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

    def generate_response(self, instruction: str, max_length: int = 512, temperature: float = 0.1) -> str:
        """
        生成模型回复

        Args:
            instruction: 输入指令
            max_length: 最大生成长度
            temperature: 温度参数

        Returns:
            模型生成的回复
        """
        try:
            # 编码输入
            inputs = self.tokenizer.encode(instruction, return_tensors="pt")
            input_length = inputs.shape[1]

            print(f"生成回复 (输入长度: {input_length})...")
            print(f"系统指令: ")
            print(f"用户问题: {repr((instruction,))}")

            # 生成回复
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.to(self.device),
                    max_length=min(input_length + max_length, 2048),
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=0.9 if temperature > 0 else None,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                )

            # 解码回复（只取新生成的部分）
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_response[len(instruction):].strip()

            print(f"模型回复: {response}")
            return response

        except Exception as e:
            print(f"❌ 生成回复时出错: {str(e)}")
            return ""

    def load_dataset(self, data_file: str) -> List[Dict]:
        """
        加载数据集

        Args:
            data_file: 数据集文件路径

        Returns:
            数据集列表
        """
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            print(f"✅ 成功加载数据集: {data_file}")
            print(f"📊 数据集大小: {len(data)} 条")
            return data

        except Exception as e:
            print(f"❌ 加载数据集失败: {str(e)}")
            return []

    def evaluate_dataset(self, data_file: str, dataset_tag: str, output_dir: str = "./") -> Dict:
        """
        评估整个数据集

        Args:
            data_file: 数据集文件路径
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

        # 评估开始时间
        start_time = time.time()

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

            # 记录结果
            result_item = {
                "question_id": i + 1,
                "instruction": instruction,
                "expected_answer": expected_output,
                "model_response": model_response,
                "extracted_answer": extracted_answer,
                "is_correct": is_correct,
                "timestamp": datetime.now().isoformat()
            }

            results.append(result_item)

            # 用于计算指标（只有成功提取答案的才参与计算）
            if extracted_answer:
                predictions.append(extracted_answer)
                ground_truths.append(expected_output)

            # 每10条数据显示一次进度
            if (i + 1) % 10 == 0:
                current_accuracy = sum(1 for r in results if r['is_correct']) / len(results)
                print(f"📊 已处理 {i+1}/{len(dataset)} 条，当前准确率: {current_accuracy:.3f}")

        # 计算评估指标
        evaluation_metrics = self.calculate_metrics(predictions, ground_truths)

        # 计算总体统计
        total_questions = len(dataset)
        answered_questions = len([r for r in results if r['extracted_answer']])
        correct_answers = len([r for r in results if r['is_correct']])

        # 评估结束时间
        end_time = time.time()
        evaluation_time = end_time - start_time

        # 准备最终结果
        final_results = {
            "dataset_info": {
                "dataset_tag": dataset_tag,
                "data_file": data_file,
                "total_questions": total_questions,
                "answered_questions": answered_questions,
                "unanswered_questions": total_questions - answered_questions,
            },
            "performance_metrics": evaluation_metrics,
            "statistics": {
                "overall_accuracy": correct_answers / total_questions if total_questions > 0 else 0,
                "answer_extraction_rate": answered_questions / total_questions if total_questions > 0 else 0,
                "evaluation_time_seconds": evaluation_time,
                "questions_per_second": total_questions / evaluation_time if evaluation_time > 0 else 0,
            },
            "timestamp": datetime.now().isoformat()
        }

        # 保存详细结果
        answers_file = os.path.join(output_dir, f"generated_answers_{dataset_tag}.json")
        with open(answers_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 保存评估结果
        eval_file = os.path.join(output_dir, f"eval_result_{dataset_tag}.json")
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)

        # 打印结果摘要
        self.print_evaluation_summary(final_results, answers_file, eval_file)

        return final_results

    def calculate_metrics(self, predictions: List[str], ground_truths: List[str]) -> Dict:
        """
        计算评估指标

        Args:
            predictions: 预测结果列表
            ground_truths: 真实标签列表

        Returns:
            评估指标字典
        """
        if not predictions or not ground_truths:
            return {
                "accuracy": 0.0,
                "precision_macro": 0.0,
                "recall_macro": 0.0,
                "f1_macro": 0.0,
                "precision_micro": 0.0,
                "recall_micro": 0.0,
                "f1_micro": 0.0,
                "per_class_metrics": {}
            }

        # 计算准确率
        accuracy = accuracy_score(ground_truths, predictions)

        # 计算精确率、召回率、F1分数
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            ground_truths, predictions, average='macro', zero_division=0
        )

        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            ground_truths, predictions, average='micro', zero_division=0
        )

        # 计算每个类别的指标
        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
            ground_truths, predictions, average=None, zero_division=0
        )

        # 获取所有类别
        unique_labels = sorted(list(set(ground_truths + predictions)))

        per_class_metrics = {}
        for i, label in enumerate(unique_labels):
            if i < len(precision_per_class):
                per_class_metrics[label] = {
                    "precision": float(precision_per_class[i]),
                    "recall": float(recall_per_class[i]),
                    "f1": float(f1_per_class[i]),
                    "support": int(support[i])
                }

        return {
            "accuracy": float(accuracy),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
            "precision_micro": float(precision_micro),
            "recall_micro": float(recall_micro),
            "f1_micro": float(f1_micro),
            "per_class_metrics": per_class_metrics
        }

    def print_evaluation_summary(self, results: Dict, answers_file: str, eval_file: str):
        """
        打印评估结果摘要

        Args:
            results: 评估结果字典
            answers_file: 详细答案文件路径
            eval_file: 评估结果文件路径
        """
        print("\n" + "=" * 60)
        print("🎉 评估完成！结果摘要:")
        print("=" * 60)

        dataset_info = results["dataset_info"]
        metrics = results["performance_metrics"]
        stats = results["statistics"]

        print(f"📊 数据集: {dataset_info['dataset_tag']}")
        print(f"📁 数据文件: {dataset_info['data_file']}")
        print(f"🔢 总问题数: {dataset_info['total_questions']}")
        print(f"✅ 成功回答: {dataset_info['answered_questions']}")
        print(f"❌ 未能回答: {dataset_info['unanswered_questions']}")

        print(f"\n📈 性能指标:")
        print(f"  整体准确率: {stats['overall_accuracy']:.4f}")
        print(f"  答案提取率: {stats['answer_extraction_rate']:.4f}")
        print(f"  准确率 (Accuracy): {metrics['accuracy']:.4f}")
        print(f"  精确率 (Precision-Macro): {metrics['precision_macro']:.4f}")
        print(f"  召回率 (Recall-Macro): {metrics['recall_macro']:.4f}")
        print(f"  F1分数 (F1-Macro): {metrics['f1_macro']:.4f}")

        print(f"\n⏱️  性能统计:")
        print(f"  评估总时间: {stats['evaluation_time_seconds']:.2f} 秒")
        print(f"  处理速度: {stats['questions_per_second']:.2f} 问题/秒")

        print(f"\n📁 输出文件:")
        print(f"  详细答案: {answers_file}")
        print(f"  评估结果: {eval_file}")

        # 打印每个类别的详细指标
        if metrics["per_class_metrics"]:
            print(f"\n📊 各类别详细指标:")
            for label, class_metrics in metrics["per_class_metrics"].items():
                print(f"  选项 {label}: P={class_metrics['precision']:.3f}, "
                      f"R={class_metrics['recall']:.3f}, "
                      f"F1={class_metrics['f1']:.3f}, "
                      f"Support={class_metrics['support']}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="CultureSPA模型评估脚本")
    parser.add_argument("--dataset_id", type=int, required=True,
                       help="数据集ID (2=CulturalBench, 3=normad, 4=cultureLLM, 5=cultureAtlas)")
    parser.add_argument("--data_file", type=str, required=True,
                       help="数据集文件路径")
    parser.add_argument("--dataset_tag", type=str, required=True,
                       help="数据集标签")
    parser.add_argument("--model_path", type=str, default=None,
                       help="模型路径")
    parser.add_argument("--output_dir", type=str, default="./",
                       help="输出目录")
    parser.add_argument("--device", type=str, default="auto",
                       help="设备设置")

    args = parser.parse_args()

    print("🏛️  CultureSPA模型评估器")
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
    evaluator = CultureSPAEvaluator(model_path=args.model_path, device=args.device)

    # 加载模型
    if not evaluator.load_model():
        print("❌ 模型加载失败，退出评估")
        return

    # 运行评估
    results = evaluator.evaluate_dataset(
        data_file=args.data_file,
        dataset_tag=args.dataset_tag,
        output_dir=args.output_dir
    )

    if results:
        print("✅ 评估成功完成！")
    else:
        print("❌ 评估失败！")


if __name__ == "__main__":
    main()