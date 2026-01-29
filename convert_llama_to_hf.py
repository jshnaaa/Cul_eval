#!/usr/bin/env python3
"""
Llama Meta格式转HF格式转换脚本
基于transformers库的官方转换逻辑
"""

import argparse
import json
import os
import shutil
import torch
from pathlib import Path
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f, indent=2)


def write_model(model_path, input_base_path, model_size):
    """
    转换Meta格式的Llama模型到HF格式
    """
    os.makedirs(model_path, exist_ok=True)
    print(f"正在转换模型到: {model_path}")

    # 读取原始参数
    params_path = os.path.join(input_base_path, "params.json")
    if not os.path.exists(params_path):
        print(f"❌ 找不到 params.json 文件: {params_path}")
        return False

    params = read_json(params_path)
    print(f"📋 读取模型参数: {params}")

    # 创建HF配置
    config = LlamaConfig(
        vocab_size=params.get("vocab_size", 32000),
        hidden_size=params["dim"],
        intermediate_size=params.get("ffn_dim_multiplier", 1) * params["dim"] * 8 // 3,
        num_hidden_layers=params["n_layers"],
        num_attention_heads=params["n_heads"],
        num_key_value_heads=params.get("n_kv_heads", params["n_heads"]),
        max_position_embeddings=params.get("max_seq_len", 4096),
        rms_norm_eps=params.get("norm_eps", 1e-5),
        rope_theta=params.get("rope_theta", 10000.0),
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
        torch_dtype="float16",
    )

    # 保存配置
    config.save_pretrained(model_path)
    print("✅ 保存config.json")

    # 加载权重文件
    weight_file = os.path.join(input_base_path, "consolidated.00.pth")
    if not os.path.exists(weight_file):
        print(f"❌ 找不到权重文件: {weight_file}")
        return False

    print("📦 加载原始权重...")
    state_dict = torch.load(weight_file, map_location="cpu")

    # 转换权重命名
    print("🔄 转换权重命名...")
    new_state_dict = {}

    # 权重映射规则
    for key, value in state_dict.items():
        if key == "tok_embeddings.weight":
            new_state_dict["model.embed_tokens.weight"] = value
        elif key == "norm.weight":
            new_state_dict["model.norm.weight"] = value
        elif key == "output.weight":
            new_state_dict["lm_head.weight"] = value
        elif key.startswith("layers."):
            # 处理transformer层
            layer_num = key.split(".")[1]
            if "attention.wq.weight" in key:
                new_state_dict[f"model.layers.{layer_num}.self_attn.q_proj.weight"] = value
            elif "attention.wk.weight" in key:
                new_state_dict[f"model.layers.{layer_num}.self_attn.k_proj.weight"] = value
            elif "attention.wv.weight" in key:
                new_state_dict[f"model.layers.{layer_num}.self_attn.v_proj.weight"] = value
            elif "attention.wo.weight" in key:
                new_state_dict[f"model.layers.{layer_num}.self_attn.o_proj.weight"] = value
            elif "attention_norm.weight" in key:
                new_state_dict[f"model.layers.{layer_num}.input_layernorm.weight"] = value
            elif "feed_forward.w1.weight" in key:
                new_state_dict[f"model.layers.{layer_num}.mlp.gate_proj.weight"] = value
            elif "feed_forward.w2.weight" in key:
                new_state_dict[f"model.layers.{layer_num}.mlp.down_proj.weight"] = value
            elif "feed_forward.w3.weight" in key:
                new_state_dict[f"model.layers.{layer_num}.mlp.up_proj.weight"] = value
            elif "ffn_norm.weight" in key:
                new_state_dict[f"model.layers.{layer_num}.post_attention_layernorm.weight"] = value

    # 保存转换后的权重
    print("💾 保存转换后的权重...")
    torch.save(new_state_dict, os.path.join(model_path, "pytorch_model.bin"))
    print("✅ 保存pytorch_model.bin")

    # 处理tokenizer
    print("🔤 处理tokenizer...")
    tokenizer_model_path = os.path.join(input_base_path, "tokenizer.model")
    if os.path.exists(tokenizer_model_path):
        # 复制tokenizer.model文件
        shutil.copy(tokenizer_model_path, os.path.join(model_path, "tokenizer.model"))

        # 创建tokenizer配置
        tokenizer_config = {
            "tokenizer_class": "LlamaTokenizer",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": None,
            "sp_model_kwargs": {},
            "spaces_between_special_tokens": False,
            "legacy": True
        }

        write_json(tokenizer_config, os.path.join(model_path, "tokenizer_config.json"))

        # 创建special_tokens_map
        special_tokens_map = {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>"
        }
        write_json(special_tokens_map, os.path.join(model_path, "special_tokens_map.json"))

        print("✅ 保存tokenizer文件")

    print("🎉 转换完成!")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Meta格式模型目录路径",
        default="./Llama-2-7b-chat"
    )
    parser.add_argument(
        "--output_dir",
        help="HF格式输出目录路径",
        default="./Llama-2-7b-chat-hf"
    )
    parser.add_argument(
        "--model_size",
        help="模型大小",
        default="7B"
    )
    args = parser.parse_args()

    print("🦙 Llama Meta格式 -> HF格式转换工具")
    print("=" * 50)
    print(f"📁 输入目录: {args.input_dir}")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"📏 模型大小: {args.model_size}")
    print("=" * 50)

    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"❌ 输入目录不存在: {args.input_dir}")
        return

    required_files = ["params.json", "consolidated.00.pth", "tokenizer.model"]
    for file in required_files:
        file_path = os.path.join(args.input_dir, file)
        if not os.path.exists(file_path):
            print(f"❌ 必需文件不存在: {file_path}")
            return

    # 执行转换
    success = write_model(args.output_dir, args.input_dir, args.model_size)

    if success:
        print(f"\n🎉 转换成功! HF格式模型保存在: {args.output_dir}")
        print("\n📋 生成的文件:")
        for file in os.listdir(args.output_dir):
            print(f"  - {file}")
    else:
        print("\n❌ 转换失败!")


if __name__ == "__main__":
    main()