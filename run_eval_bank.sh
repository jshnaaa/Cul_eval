#!/bin/bash

# CultureBank模型评估运行脚本
# 使用方法: bash run_eval_bank.sh <DATA_ID>
# 示例: bash run_eval_bank.sh 2

set -e

# 检查参数
if [ $# -ne 1 ]; then
    echo "错误：需要提供DATA_ID参数"
    echo "使用方法: bash run_eval_bank.sh <DATA_ID>"
    echo "示例: bash run_eval_bank.sh 2"
    exit 1
fi

DATA_ID=$1

# 固定配置
OUTPUT_DIR="./eval_results_${DATA_ID}"

# 根据DATA_ID设置数据集配置
case $DATA_ID in
    2)
        TRAIN_FILE="/root/autodl-fs/CulturalBench_merge_gen_gpt.json"
        DATASET_TAG="CulturalBench"
        ;;
    3)
        TRAIN_FILE="/root/autodl-fs/normad_merge_gen_gpt.json"
        DATASET_TAG="normad"
        ;;
    4)
        TRAIN_FILE="/root/autodl-fs/cultureLLM_merge_gen_gpt.json"
        DATASET_TAG="cultureLLM"
        ;;
    5)
        TRAIN_FILE="/root/autodl-fs/cultureAtlas_merge_gen_gpt.json"
        DATASET_TAG="cultureAtlas"
        ;;
    *)
        echo "错误：无效的DATA_ID: $DATA_ID"
        echo "支持的DATA_ID: 2(CulturalBench), 3(normad), 4(cultureLLM), 5(cultureAtlas)"
        exit 1
        ;;
esac

echo "开始评估 CultureBank 模型在 $DATASET_TAG 数据集上的表现..."
echo "数据文件: $TRAIN_FILE"
echo "输出目录: $OUTPUT_DIR"

# 检查文件是否存在
if [ ! -f "$TRAIN_FILE" ]; then
    echo "错误：数据文件不存在: $TRAIN_FILE"
    exit 1
fi

if [ ! -f "./eval_culturebank.py" ]; then
    echo "错误：评估脚本不存在: ./eval_culturebank.py"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 执行评估
python3 ./eval_culturebank.py \
    --dataset_id $DATA_ID \
    --data_file "$TRAIN_FILE" \
    --dataset_tag "$DATASET_TAG" \
    --output_dir "$OUTPUT_DIR"

echo "评估完成！"
echo "结果文件保存在: $OUTPUT_DIR"
echo "  - generated_answers.json: 详细的问答结果"
echo "  - eval_results.json: 评估指标统计"