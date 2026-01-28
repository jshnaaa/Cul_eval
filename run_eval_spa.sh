#!/bin/bash

# CultureSPA模型评估运行脚本
# 使用方法: ./run_eval_spa.sh <DATA_ID>
# 示例: ./run_eval_spa.sh 2

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示使用说明
show_usage() {
    echo "🏛️  CultureSPA模型评估脚本"
    echo ""
    echo "使用方法:"
    echo "  $0 <DATA_ID> [OPTIONS]"
    echo ""
    echo "DATA_ID 选项:"
    echo "  2  - CulturalBench 数据集"
    echo "  3  - normad 数据集"
    echo "  4  - cultureLLM 数据集"
    echo "  5  - cultureAtlas 数据集"
    echo ""
    echo "可选参数:"
    echo "  --model_path PATH     指定模型路径"
    echo "  --output_dir PATH     指定输出目录 (默认: ./eval_results)"
    echo "  --device DEVICE       指定设备 (默认: auto)"
    echo ""
    echo "示例:"
    echo "  $0 2                                    # 评估CulturalBench数据集"
    echo "  $0 3 --output_dir ./results             # 评估normad并指定输出目录"
    echo "  $0 4 --model_path /path/to/model        # 评估cultureLLM并指定模型路径"
    echo ""
}

# 检查参数
if [ $# -lt 1 ]; then
    print_error "缺少必需的DATA_ID参数"
    show_usage
    exit 1
fi

# 获取DATA_ID
DATA_ID=$1
shift  # 移除第一个参数，剩下的是可选参数

# 默认设置
MODEL_PATH=""
OUTPUT_DIR="./eval_results"
DEVICE="auto"

# 解析可选参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "未知参数: $1"
            show_usage
            exit 1
            ;;
    esac
done

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
        print_error "无效的DATA_ID: $DATA_ID (支持: 2=CulturalBench, 3=normad, 4=cultureLLM, 5=cultureAtlas)"
        show_usage
        exit 1
        ;;
esac

# 显示配置信息
print_info "评估配置信息:"
echo "  📊 数据集ID: $DATA_ID"
echo "  🏷️  数据集标签: $DATASET_TAG"
echo "  📁 数据文件: $TRAIN_FILE"
echo "  📂 输出目录: $OUTPUT_DIR"
echo "  🖥️  设备: $DEVICE"
if [ -n "$MODEL_PATH" ]; then
    echo "  🤖 模型路径: $MODEL_PATH"
else
    echo "  🤖 模型路径: 使用默认路径"
fi
echo ""

# 检查数据文件是否存在
if [ ! -f "$TRAIN_FILE" ]; then
    print_error "数据文件不存在: $TRAIN_FILE"
    print_warning "请确保数据文件路径正确，或者数据已经下载到指定位置"
    exit 1
fi

# 检查Python脚本是否存在
EVAL_SCRIPT="./eval_spa.py"
if [ ! -f "$EVAL_SCRIPT" ]; then
    print_error "评估脚本不存在: $EVAL_SCRIPT"
    print_warning "请确保 eval_spa.py 文件在当前目录下"
    exit 1
fi

# 创建输出目录
print_info "创建输出目录: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# 检查Python环境和依赖
print_info "检查Python环境和依赖..."

# 检查Python
if ! command -v python3 &> /dev/null; then
    print_error "未找到python3命令"
    exit 1
fi

# 检查必需的Python包
REQUIRED_PACKAGES=("torch" "transformers" "tqdm" "scikit-learn")
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! python3 -c "import $package" &> /dev/null; then
        MISSING_PACKAGES+=("$package")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    print_error "缺少必需的Python包: ${MISSING_PACKAGES[*]}"
    print_warning "请运行: pip install ${MISSING_PACKAGES[*]}"
    exit 1
fi

print_success "Python环境检查通过"

# 构建Python命令
PYTHON_CMD="python3 $EVAL_SCRIPT"
PYTHON_CMD="$PYTHON_CMD --dataset_id $DATA_ID"
PYTHON_CMD="$PYTHON_CMD --data_file $TRAIN_FILE"
PYTHON_CMD="$PYTHON_CMD --dataset_tag $DATASET_TAG"
PYTHON_CMD="$PYTHON_CMD --output_dir $OUTPUT_DIR"
PYTHON_CMD="$PYTHON_CMD --device $DEVICE"

if [ -n "$MODEL_PATH" ]; then
    PYTHON_CMD="$PYTHON_CMD --model_path $MODEL_PATH"
fi

# 记录开始时间
START_TIME=$(date +%s)
START_TIME_STR=$(date '+%Y-%m-%d %H:%M:%S')

print_info "开始评估..."
print_info "开始时间: $START_TIME_STR"
print_info "执行命令: $PYTHON_CMD"
echo ""

# 创建日志文件
LOG_FILE="$OUTPUT_DIR/eval_log_${DATASET_TAG}_$(date +%Y%m%d_%H%M%S).log"
print_info "日志文件: $LOG_FILE"

# 执行评估（同时输出到控制台和日志文件）
if eval "$PYTHON_CMD" 2>&1 | tee "$LOG_FILE"; then
    # 计算耗时
    END_TIME=$(date +%s)
    END_TIME_STR=$(date '+%Y-%m-%d %H:%M:%S')
    DURATION=$((END_TIME - START_TIME))

    # 格式化耗时
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))
    SECONDS=$((DURATION % 60))

    echo ""
    print_success "评估完成！"
    echo "  📊 数据集: $DATASET_TAG"
    echo "  ⏰ 开始时间: $START_TIME_STR"
    echo "  ⏰ 结束时间: $END_TIME_STR"
    echo "  ⏱️  总耗时: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo ""

    # 显示输出文件
    print_info "生成的文件:"
    echo "  📄 详细答案: $OUTPUT_DIR/generated_answers_${DATASET_TAG}.json"
    echo "  📊 评估结果: $OUTPUT_DIR/eval_result_${DATASET_TAG}.json"
    echo "  📋 运行日志: $LOG_FILE"

    # 如果评估结果文件存在，显示关键指标
    RESULT_FILE="$OUTPUT_DIR/eval_result_${DATASET_TAG}.json"
    if [ -f "$RESULT_FILE" ]; then
        echo ""
        print_info "关键指标摘要:"

        # 使用Python提取关键指标
        python3 -c "
import json
try:
    with open('$RESULT_FILE', 'r') as f:
        result = json.load(f)

    stats = result.get('statistics', {})
    metrics = result.get('performance_metrics', {})
    dataset_info = result.get('dataset_info', {})

    print(f'  📈 整体准确率: {stats.get(\"overall_accuracy\", 0):.4f}')
    print(f'  📈 答案提取率: {stats.get(\"answer_extraction_rate\", 0):.4f}')
    print(f'  📈 F1分数(宏平均): {metrics.get(\"f1_macro\", 0):.4f}')
    print(f'  📊 总问题数: {dataset_info.get(\"total_questions\", 0)}')
    print(f'  ✅ 成功回答: {dataset_info.get(\"answered_questions\", 0)}')

except Exception as e:
    print(f'  ⚠️  无法解析结果文件: {e}')
"
    fi

    echo ""
    print_success "🎉 所有任务已完成！"

else
    # 评估失败
    END_TIME=$(date +%s)
    END_TIME_STR=$(date '+%Y-%m-%d %H:%M:%S')
    DURATION=$((END_TIME - START_TIME))

    echo ""
    print_error "评估失败！"
    echo "  📊 数据集: $DATASET_TAG"
    echo "  ⏰ 开始时间: $START_TIME_STR"
    echo "  ⏰ 失败时间: $END_TIME_STR"
    echo "  ⏱️  运行时长: ${DURATION}秒"
    echo "  📋 详细错误请查看日志: $LOG_FILE"

    print_warning "常见问题排查:"
    echo "  1. 检查数据文件是否存在且格式正确"
    echo "  2. 检查模型是否正确加载"
    echo "  3. 检查GPU内存是否足够"
    echo "  4. 检查Python依赖是否完整"

    exit 1
fi