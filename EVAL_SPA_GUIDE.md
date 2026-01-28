# CultureSPA模型评估系统使用指南

## 概述

CultureSPA评估系统提供了完整的文化知识数据集评估功能，支持多个数据集的批量评估、详细的性能指标计算和结果分析。

## 文件说明

### 核心文件

- **`eval_spa.py`** - 基础评估脚本
- **`eval_spa_enhanced.py`** - 增强版评估脚本（推荐）
- **`run_eval_spa.sh`** - 自动化运行脚本
- **`eval_config.json`** - 配置文件

### 输出文件

- **`generated_answers_{dataset}.json`** - 详细的问答结果
- **`eval_result_{dataset}.json`** - 评估指标和统计信息
- **`eval_log_{dataset}_{timestamp}.log`** - 运行日志

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install transformers torch tqdm scikit-learn

# 确保数据文件存在
ls -la /root/autodl-fs/CulturalBench_merge_gen_gpt.json
```

### 2. 基础使用

```bash
# 使用运行脚本（推荐）
./run_eval_spa.sh 2  # 评估CulturalBench数据集

# 直接运行Python脚本
python eval_spa.py --dataset_id 2 --data_file /path/to/data.json --dataset_tag CulturalBench
```

### 3. 高级使用

```bash
# 使用增强版脚本（支持断点续传）
python eval_spa_enhanced.py --dataset_id 2 --config eval_config.json

# 自定义输出目录
./run_eval_spa.sh 2 --output_dir ./my_results

# 指定模型路径
./run_eval_spa.sh 2 --model_path /path/to/model
```

## 支持的数据集

| ID | 数据集名称 | 文件路径 | 描述 |
|----|-----------|----------|------|
| 2  | CulturalBench | `/root/autodl-fs/CulturalBench_merge_gen_gpt.json` | 文化知识基准测试集 |
| 3  | normad | `/root/autodl-fs/normad_merge_gen_gpt.json` | 规范和文化行为数据集 |
| 4  | cultureLLM | `/root/autodl-fs/cultureLLM_merge_gen_gpt.json` | 文化LLM评估数据集 |
| 5  | cultureAtlas | `/root/autodl-fs/cultureAtlas_merge_gen_gpt.json` | 文化地图知识数据集 |

## 配置文件说明

### eval_config.json 结构

```json
{
  "model_settings": {
    "default_model_path": null,           // 默认模型路径
    "generation_params": {
      "max_length": 512,                  // 最大生成长度
      "temperature": 0.1,                 // 温度参数
      "top_p": 0.9,                      // top_p采样
      "repetition_penalty": 1.1           // 重复惩罚
    },
    "device": "auto"                      // 设备设置
  },
  "evaluation_settings": {
    "output_dir": "./eval_results",       // 输出目录
    "progress_update_interval": 10,       // 进度更新间隔
    "answer_extraction_patterns": [...]   // 答案提取正则模式
  }
}
```

## 输出结果说明

### generated_answers_{dataset}.json

包含每个问题的详细信息：

```json
[
  {
    "question_id": 1,
    "instruction": "问题内容...",
    "expected_answer": "1",
    "model_response": "模型的完整回复...",
    "extracted_answer": "1",
    "is_correct": true,
    "timestamp": "2024-01-01T12:00:00"
  }
]
```

### eval_result_{dataset}.json

包含评估指标和统计信息：

```json
{
  "dataset_info": {
    "dataset_tag": "CulturalBench",
    "total_questions": 1000,
    "answered_questions": 995,
    "unanswered_questions": 5
  },
  "performance_metrics": {
    "accuracy": 0.8500,
    "precision_macro": 0.8400,
    "recall_macro": 0.8300,
    "f1_macro": 0.8350,
    "per_class_metrics": {
      "1": {"precision": 0.85, "recall": 0.82, "f1": 0.83},
      "2": {"precision": 0.83, "recall": 0.84, "f1": 0.84},
      "3": {"precision": 0.84, "recall": 0.83, "f1": 0.83},
      "4": {"precision": 0.84, "recall": 0.83, "f1": 0.84}
    }
  },
  "statistics": {
    "overall_accuracy": 0.8500,
    "answer_extraction_rate": 0.9950,
    "evaluation_time_seconds": 1200.5,
    "questions_per_second": 0.83
  }
}
```

## 高级功能

### 1. 断点续传

增强版脚本支持断点续传功能：

- 自动保存进度检查点
- 意外中断后可从中断点继续
- 检查点文件：`checkpoint_{dataset}.json`

### 2. 详细日志

```bash
# 查看实时日志
tail -f eval_results/eval_log_CulturalBench_20240101_120000.log

# 搜索错误信息
grep "ERROR" eval_results/*.log
```

### 3. 性能监控

脚本会实时显示：
- 处理进度和速度
- 当前准确率
- 内存使用情况
- 预计完成时间

## 故障排除

### 常见问题

**1. 模型加载失败**
```bash
# 检查模型路径
export CULTURESPA_MODEL_PATH=/path/to/your/model
python eval_spa.py --model_path /path/to/model
```

**2. 数据文件不存在**
```bash
# 检查文件路径和权限
ls -la /root/autodl-fs/CulturalBench_merge_gen_gpt.json
```

**3. GPU内存不足**
```bash
# 使用CPU模式
python eval_spa.py --device cpu

# 或减少生成长度
# 修改配置文件中的max_length参数
```

**4. 答案提取失败率高**
```bash
# 检查提取模式
grep "答案提取失败" eval_results/*.log

# 调整配置文件中的answer_extraction_patterns
```

### 性能优化

**1. 加速评估**
```bash
# 使用更小的生成长度
# 降低temperature参数
# 使用GPU而非CPU
```

**2. 减少内存使用**
```bash
# 使用量化模型
# 减少batch_size
# 使用gradient_checkpointing
```

## 使用示例

### 评估所有数据集

```bash
#!/bin/bash
# 批量评估所有数据集

for dataset_id in 2 3 4 5; do
    echo "开始评估数据集 $dataset_id"
    ./run_eval_spa.sh $dataset_id --output_dir ./results_all
    echo "数据集 $dataset_id 评估完成"
done

echo "所有数据集评估完成！"
```

### 比较不同配置

```bash
# 使用不同温度参数评估
cp eval_config.json eval_config_temp01.json
# 修改temperature为0.1

cp eval_config.json eval_config_temp05.json
# 修改temperature为0.5

python eval_spa_enhanced.py --dataset_id 2 --config eval_config_temp01.json
python eval_spa_enhanced.py --dataset_id 2 --config eval_config_temp05.json
```

### 结果分析

```python
# 分析评估结果的Python脚本
import json
import pandas as pd

# 加载结果
with open('eval_results/eval_result_CulturalBench.json', 'r') as f:
    results = json.load(f)

# 提取关键指标
metrics = results['performance_metrics']
print(f"准确率: {metrics['accuracy']:.4f}")
print(f"F1分数: {metrics['f1_macro']:.4f}")

# 分析每个选项的表现
for option, metrics in results['performance_metrics']['per_class_metrics'].items():
    print(f"选项 {option}: 精确率={metrics['precision']:.3f}, 召回率={metrics['recall']:.3f}")
```

## 技术支持

### 日志级别

- **INFO**: 一般信息
- **WARNING**: 警告信息（如答案提取失败）
- **ERROR**: 错误信息（如模型加载失败）

### 调试技巧

1. **启用详细日志**：修改配置文件中的logging设置
2. **检查中间结果**：查看checkpoint文件
3. **单条数据测试**：先用小数据集验证
4. **性能分析**：使用`time`命令测量执行时间

### 联系信息

如遇到问题，请检查：
1. 模型文件是否完整
2. 数据文件格式是否正确
3. 依赖包版本是否兼容
4. 硬件资源是否充足