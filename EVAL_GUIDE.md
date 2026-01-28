# CultureBank模型评估指南

## 概述

CultureBank是基于Llama-2-7b-chat-hf的文化知识增强模型，使用LoRA微调技术训练。

## 文件说明

- `eval_culturebank.py` - 完整版评估脚本（推荐）
- `eval_culturebank_simple.py` - 简化版评估脚本
- `requirements.txt` - 依赖包列表

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

或手动安装核心依赖：
```bash
pip install transformers peft torch accelerate bitsandbytes
```

### 2. 运行评估

**完整版（推荐）:**
```bash
python eval_culturebank.py
```

**简化版（内存有限时）:**
```bash
python eval_culturebank_simple.py
```

## 功能特性

### 完整版 (eval_culturebank.py)

- ✅ 两种模型加载方式（自动fallback）
- ✅ 4bit量化支持（节省GPU内存）
- ✅ 内存使用监控
- ✅ 性能统计（生成速度、tokens/秒）
- ✅ 自动化测试用例
- ✅ 交互式对话模式
- ✅ 详细的错误处理和调试信息

### 简化版 (eval_culturebank_simple.py)

- ✅ 基本模型加载
- ✅ 简单对话测试
- ✅ 最小依赖要求
- ✅ 适合调试和快速测试

## 系统要求

### 硬件要求

**推荐配置:**
- GPU: 16GB+ VRAM (如RTX 4090, A100)
- RAM: 32GB+
- 存储: 20GB+ 可用空间

**最低配置 (使用4bit量化):**
- GPU: 8GB+ VRAM (如RTX 3080, RTX 4070)
- RAM: 16GB+
- 存储: 15GB+ 可用空间

**CPU模式 (不推荐):**
- RAM: 64GB+
- 推理速度极慢

### 软件要求

- Python 3.8+
- CUDA 11.8+ (GPU模式)
- PyTorch 2.0+

## 使用示例

### 基本对话测试

```python
# 运行脚本后，模型会自动测试以下用例：
test_cases = [
    "你好",
    "介绍一下中国的传统节日",
    "What is the significance of the Spring Festival?",
    "请解释一下儒家思想的核心理念",
    "Tell me about traditional Chinese medicine"
]
```

### 交互式对话

运行脚本后选择进入交互模式，可以进行实时对话：

```
👤 您: 你好，请介绍一下自己
🤖 CultureBank: 你好！我是CultureBank，一个专注于文化知识的AI助手...

👤 您: 中国古代四大发明是什么？
🤖 CultureBank: 中国古代四大发明是指...
```

## 故障排除

### 常见问题

**1. CUDA内存不足**
```
RuntimeError: CUDA out of memory
```
解决方案：
- 使用4bit量化（默认开启）
- 减少max_length参数
- 使用简化版脚本

**2. 模型下载失败**
```
OSError: meta-llama/Llama-2-7b-chat-hf does not appear to be a repository
```
解决方案：
- 确保有HuggingFace访问权限
- 登录HuggingFace: `huggingface-cli login`
- 申请Llama-2访问权限

**3. 依赖版本冲突**
```
ImportError: cannot import name 'xxx'
```
解决方案：
- 更新依赖: `pip install -r requirements.txt --upgrade`
- 使用虚拟环境隔离依赖

**4. 生成结果异常**
- 检查模型路径是否正确
- 确认adapter文件完整性
- 尝试不同的生成参数

### 性能优化

**GPU内存优化:**
```python
# 启用4bit量化（默认）
use_4bit = True

# 减少生成长度
max_length = 256  # 默认512

# 使用梯度检查点
gradient_checkpointing = True
```

**推理速度优化:**
```python
# 禁用采样使用贪心解码
do_sample = False

# 减少beam数量
num_beams = 1

# 使用float16精度
torch_dtype = torch.float16
```

## 模型信息

- **基座模型**: meta-llama/Llama-2-7b-chat-hf
- **微调方法**: LoRA (Low-Rank Adaptation)
- **目标模块**: q_proj, v_proj
- **LoRA参数**: rank=8, alpha=16, dropout=0.05
- **量化**: 4bit NF4 + double quantization

## 许可证

请遵守Llama-2模型的使用许可证和相关法律法规。

## 联系支持

如遇到问题，请检查：
1. 硬件配置是否满足要求
2. 依赖版本是否正确
3. 模型文件是否完整
4. HuggingFace访问权限是否正常