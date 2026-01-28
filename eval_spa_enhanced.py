#!/usr/bin/env python3
"""
CultureSPA模型增强评估脚本
支持配置文件、断点续传、详细日志等高级功能

使用方法:
python eval_spa_enhanced.py --dataset_id 2 --config eval_config.json

依赖:
pip install transformers torch tqdm scikit-learn
"""

import json
import argparse
import re
import time
import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import warnings
warnings.filterwarnings("ignore")


class EnhancedCultureSPAEvaluator:
    def __init__(self, config_file: str = "eval_config.json"):
        """
        初始化增强版CultureSPA评估器

        Args:
            config_file: 配置文件路径
        """
        self.config = self.load_config(config_file)
        self.model = None
        self.tokenizer = None
        self.logger = self.setup_logger()

        # 从配置获取设备信息
        device_config = self.config["model_settings"]["device"]
        self.device = device_config if device_config != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

        self.logger.info(f"使用设备: {self.device}")

    def load_config(self, config_file: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"✅ 成功加载配置文件: {config_file}")
            return config
        except Exception as e:
            print(f"⚠️  无法加载配置文件 {config_file}: {e}")
            print("使用默认配置...")
            return self.get_default_config()

    def get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            "model_settings": {
                "default_model_path": None,
                "generation_params": {
                    "max_length": 512,
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "repetition_penalty": 1.1
                },
                "device": "auto"
            },
            "evaluation_settings": {
                "output_dir": "./eval_results",
                "save_detailed_results": True,
                "save_evaluation_metrics": True,
                "progress_update_interval": 10,
                "answer_extraction_patterns": [
                    "\\b([1-4])\\b",
                    "答案[是为]?\\s*([1-4])",
                    "选择\\s*([1-4])",
                    "([1-4])\\s*[.。]",
                    "选项\\s*([1-4])"
                ]
            },
            "logging": {
                "enable_detailed_logging": True,
                "log_model_responses": True,
                "log_extraction_failures": True
            }
        }

    def setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('CultureSPA_Evaluator')
        logger.setLevel(logging.INFO)

        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 创建格式器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        return logger

    def load_model(self, model_path: Optional[str] = None):
        """加载CultureSPA模型和分词器"""
        try:
            # 确定模型路径
            if model_path is None:
                model_path = self.config["model_settings"]["default_model_path"]

            if model_path is None:
                model_path = os.environ.get('CULTURESPA_MODEL_PATH', './culturespa_model')

            self.logger.info("正在加载分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

            self.logger.info("正在加载模型...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.logger.info("模型已通过 device_map='auto' 加载。")
            self.logger.info("模型加载完成！")
            return True

        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")
            return False

    def extract_answer(self, response: str) -> str:
        """
        从模型回复中提取答案

        Args:
            response: 模型生成的回复

        Returns:
            提取的答案（1-4的数字），如果提取失败返回空字符串
        """
        response = response.strip()

        # 从配置获取提取模式
        patterns = self.config["evaluation_settings"]["answer_extraction_patterns"]

        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1)

        # 如果都没匹配到，返回响应的第一个字符（如果是1-4）
        if len(response) > 0 and response[0] in '1234':
            return response[0]

        # 记录提取失败
        if self.config["logging"]["log_extraction_failures"]:
            self.logger.warning(f"答案提取失败: {response[:100]}...")

        return ""

    def generate_response(self, instruction: str) -> str:
        """
        生成模型回复

        Args:
            instruction: 输入指令

        Returns:
            模型生成的回复
        """
        try:
            # 获取生成参数
            gen_params = self.config["model_settings"]["generation_params"]

            # 编码输入
            inputs = self.tokenizer.encode(instruction, return_tensors="pt")
            input_length = inputs.shape[1]

            if self.config["logging"]["log_model_responses"]:
                self.logger.debug(f"生成回复 (输入长度: {input_length})")
                self.logger.debug(f"用户问题: {instruction[:200]}...")

            # 生成回复
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.to(self.device),
                    max_length=min(input_length + gen_params["max_length"], 2048),
                    temperature=gen_params["temperature"],
                    do_sample=gen_params["temperature"] > 0,
                    top_p=gen_params.get("top_p", 0.9),
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=gen_params.get("repetition_penalty", 1.1),
                )

            # 解码回复（只取新生成的部分）
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_response[len(instruction):].strip()

            if self.config["logging"]["log_model_responses"]:
                self.logger.debug(f"模型回复: {response[:200]}...")

            return response

        except Exception as e:
            self.logger.error(f"生成回复时出错: {str(e)}")
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

            self.logger.info(f"成功加载数据集: {data_file}")
            self.logger.info(f"数据集大小: {len(data)} 条")
            return data

        except Exception as e:
            self.logger.error(f"加载数据集失败: {str(e)}")
            return []

    def save_checkpoint(self, results: List[Dict], checkpoint_file: str):
        """保存检查点"""
        try:
            checkpoint_data = {
                "timestamp": datetime.now().isoformat(),
                "processed_count": len(results),
                "results": results
            }
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"检查点已保存: {checkpoint_file}")
        except Exception as e:
            self.logger.error(f"保存检查点失败: {str(e)}")

    def load_checkpoint(self, checkpoint_file: str) -> Tuple[List[Dict], int]:
        """加载检查点"""
        try:
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                results = checkpoint_data.get("results", [])
                processed_count = checkpoint_data.get("processed_count", 0)
                self.logger.info(f"从检查点恢复: 已处理 {processed_count} 条数据")
                return results, processed_count
        except Exception as e:
            self.logger.warning(f"加载检查点失败: {str(e)}")

        return [], 0

    def evaluate_dataset_with_resume(self, data_file: str, dataset_tag: str, output_dir: str = "./") -> Dict:
        """
        支持断点续传的数据集评估

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

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 检查点文件
        checkpoint_file = os.path.join(output_dir, f"checkpoint_{dataset_tag}.json")

        # 尝试从检查点恢复
        results, start_index = self.load_checkpoint(checkpoint_file)

        # 准备结果存储
        predictions = []
        ground_truths = []

        # 评估开始时间
        start_time = time.time()

        self.logger.info(f"开始评估 {dataset_tag} 数据集...")
        if start_index > 0:
            self.logger.info(f"从第 {start_index + 1} 条数据继续评估")

        # 进度更新间隔
        progress_interval = self.config["evaluation_settings"]["progress_update_interval"]

        # 批量处理数据
        for i in range(start_index, len(dataset)):
            item = dataset[i]
            instruction = item.get('instruction', '')
            expected_output = item.get('output', '').strip()

            if not instruction:
                self.logger.warning(f"第 {i+1} 条数据缺少instruction字段，跳过")
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

            # 定期显示进度和保存检查点
            if (i + 1) % progress_interval == 0:
                current_accuracy = sum(1 for r in results if r['is_correct']) / len(results)
                self.logger.info(f"已处理 {i+1}/{len(dataset)} 条，当前准确率: {current_accuracy:.3f}")

                # 保存检查点
                self.save_checkpoint(results, checkpoint_file)

        # 删除检查点文件（评估完成）
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            self.logger.info("评估完成，删除检查点文件")

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
            "config_used": self.config,
            "timestamp": datetime.now().isoformat()
        }

        # 保存结果
        self.save_results(results, final_results, dataset_tag, output_dir)

        return final_results

    def calculate_metrics(self, predictions: List[str], ground_truths: List[str]) -> Dict:
        """计算评估指标"""
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

    def save_results(self, results: List[Dict], final_results: Dict, dataset_tag: str, output_dir: str):
        """保存评估结果"""
        # 保存详细结果
        if self.config["evaluation_settings"]["save_detailed_results"]:
            answers_file = os.path.join(output_dir, f"generated_answers_{dataset_tag}.json")
            with open(answers_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            self.logger.info(f"详细答案已保存: {answers_file}")

        # 保存评估结果
        if self.config["evaluation_settings"]["save_evaluation_metrics"]:
            eval_file = os.path.join(output_dir, f"eval_result_{dataset_tag}.json")
            with open(eval_file, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, ensure_ascii=False, indent=2)
            self.logger.info(f"评估结果已保存: {eval_file}")

        # 打印结果摘要
        self.print_evaluation_summary(final_results)

    def print_evaluation_summary(self, results: Dict):
        """打印评估结果摘要"""
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


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="CultureSPA模型增强评估脚本")
    parser.add_argument("--dataset_id", type=int, required=True,
                       help="数据集ID")
    parser.add_argument("--config", type=str, default="eval_config.json",
                       help="配置文件路径")
    parser.add_argument("--model_path", type=str, default=None,
                       help="模型路径（覆盖配置文件设置）")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="输出目录（覆盖配置文件设置）")

    args = parser.parse_args()

    print("🏛️  CultureSPA模型增强评估器")
    print(f"📊 数据集ID: {args.dataset_id}")
    print("=" * 60)

    # 初始化评估器
    evaluator = EnhancedCultureSPAEvaluator(config_file=args.config)

    # 从配置获取数据集信息
    datasets_config = evaluator.config.get("datasets", {})
    dataset_id_str = str(args.dataset_id)

    if dataset_id_str not in datasets_config:
        print(f"❌ 无效的数据集ID: {args.dataset_id}")
        print(f"支持的数据集: {list(datasets_config.keys())}")
        return

    dataset_info = datasets_config[dataset_id_str]
    data_file = dataset_info["file_path"]
    dataset_tag = dataset_info["name"]

    print(f"📊 数据集: {dataset_tag}")
    print(f"📁 数据文件: {data_file}")

    # 检查数据文件是否存在
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return

    # 确定输出目录
    output_dir = args.output_dir or evaluator.config["evaluation_settings"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # 加载模型
    if not evaluator.load_model(args.model_path):
        print("❌ 模型加载失败，退出评估")
        return

    # 运行评估
    results = evaluator.evaluate_dataset_with_resume(
        data_file=data_file,
        dataset_tag=dataset_tag,
        output_dir=output_dir
    )

    if results:
        print("✅ 评估成功完成！")
    else:
        print("❌ 评估失败！")


if __name__ == "__main__":
    main()