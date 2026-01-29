#!/bin/bash

# 临时测试脚本
set -e

DATA_ID=test
MODEL_PATH="./CultureSPA"
OUTPUT_DIR="./eval_results_test"
TRAIN_FILE="./test_data.json"
DATASET_TAG="test"

echo "开始测试 $DATASET_TAG 数据集..."
echo "数据文件: $TRAIN_FILE"
echo "模型路径: $MODEL_PATH"
echo "输出目录: $OUTPUT_DIR"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 执行评估
python3 ./eval_spa.py \
    --dataset_id 999 \
    --data_file "$TRAIN_FILE" \
    --dataset_tag "$DATASET_TAG" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --device auto

echo "测试完成！"
echo "结果文件保存在: $OUTPUT_DIR"