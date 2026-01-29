#!/usr/bin/env python3
"""
简化测试版本 - 只测试基座模型加载，不加载adapter
"""

from eval_culturebank import CultureBankModel

def main():
    print("🧪 简化测试：只加载基座模型")
    print("=" * 50)

    # 初始化模型
    culture_model = CultureBankModel()

    # 只加载基座模型
    if not culture_model.load_base_model():
        print("❌ 基座模型加载失败")
        return

    print("✅ 基座模型加载成功!")

    # 测试简单生成
    test_message = "hello!"
    chat_prompt = culture_model.format_chat_prompt(test_message)

    print(f"\n👤 用户输入: {test_message}")
    print("🤖 基座模型回应:")

    response = culture_model.generate(chat_prompt, max_tokens=20, temperature=0.7)
    print(response)

    print("\n🎉 基座模型测试完成!")

if __name__ == "__main__":
    main()