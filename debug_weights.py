#!/usr/bin/env python3
"""
调试权重名称映射
"""

from eval_culturebank import CultureBankModel

def debug_weight_mapping():
    print("🔍 调试权重名称映射")
    print("=" * 50)

    # 初始化模型
    culture_model = CultureBankModel()

    # 只加载基座模型
    if not culture_model.load_base_model():
        print("❌ 基座模型加载失败")
        return

    # 只加载adapter权重（不应用）
    if not culture_model.load_adapter_weights():
        print("❌ adapter权重加载失败")
        return

    # 分析权重名称
    print("\n📋 LoRA权重名称样例:")
    lora_keys = list(culture_model.adapter_weights.keys())
    for i, key in enumerate(lora_keys[:10]):
        print(f"  {i+1}. {key}")

    print(f"\n📋 基座模型权重名称样例:")
    base_keys = list(culture_model.base_model.state_dict().keys())
    for i, key in enumerate(base_keys[:10]):
        print(f"  {i+1}. {key}")

    # 测试名称转换
    print(f"\n🔄 测试名称转换:")
    test_lora_key = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    expected_base_key = test_lora_key.replace('base_model.model.', '').replace('.lora_A.weight', '.weight')
    print(f"LoRA权重: {test_lora_key}")
    print(f"转换后: {expected_base_key}")
    print(f"是否存在: {expected_base_key in culture_model.base_model.state_dict()}")

    # 查找匹配的权重
    print(f"\n🎯 查找匹配的权重:")
    base_state_dict = culture_model.base_model.state_dict()
    matched_count = 0
    for key in lora_keys[:20]:  # 只检查前20个
        if 'lora_A' in key:
            clean_name = key.replace('base_model.model.', '').replace('.lora_A.weight', '.weight')
            if clean_name in base_state_dict:
                print(f"  ✅ 匹配: {key} -> {clean_name}")
                matched_count += 1
            else:
                print(f"  ❌ 未匹配: {key} -> {clean_name}")

    print(f"\n📊 匹配率: {matched_count}/{min(20, len([k for k in lora_keys if 'lora_A' in k]))}")

if __name__ == "__main__":
    debug_weight_mapping()