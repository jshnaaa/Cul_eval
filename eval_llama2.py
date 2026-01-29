#!/usr/bin/env python3
"""
直接加载Meta格式Llama-2-7b-chat模型脚本
无需转换为HF格式，直接从原始文件加载
"""

import json
import os
import struct
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple
import sentencepiece as spm


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_kv_heads = args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # repeat k/v heads if n_kv_heads < n_heads
        keys = xk.repeat_interleave(self.n_rep, dim=2)
        values = xv.repeat_interleave(self.n_rep, dim=2)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output


class ModelArgs:
    def __init__(self, **kwargs):
        self.dim = kwargs.get('dim', 4096)
        self.n_layers = kwargs.get('n_layers', 32)
        self.n_heads = kwargs.get('n_heads', 32)
        # 对于Llama-2-7B，n_kv_heads通常等于n_heads（没有GQA）
        self.n_kv_heads = kwargs.get('n_kv_heads', self.n_heads)
        self.vocab_size = kwargs.get('vocab_size', 32000)
        self.multiple_of = kwargs.get('multiple_of', 256)
        self.ffn_dim_multiplier = kwargs.get('ffn_dim_multiplier', None)
        self.norm_eps = kwargs.get('norm_eps', 1e-5)
        self.max_seq_len = kwargs.get('max_seq_len', 4096)

        # 添加调试信息
        print(f"🔧 ModelArgs: dim={self.dim}, n_heads={self.n_heads}, n_kv_heads={self.n_kv_heads}, vocab_size={self.vocab_size}")


class Llama2Model:
    def __init__(self, model_path: str):
        """
        初始化Llama2模型

        Args:
            model_path: Meta格式模型路径
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        """加载模型和tokenizer"""
        print("🦙 开始加载Llama-2-7b-chat模型...")

        # 检查必要文件
        required_files = ["params.json", "consolidated.00.pth", "tokenizer.model"]
        for file in required_files:
            file_path = os.path.join(self.model_path, file)
            if not os.path.exists(file_path):
                print(f"❌ 找不到必要文件: {file_path}")
                return False

        try:
            # 加载模型参数
            with open(os.path.join(self.model_path, "params.json"), "r") as f:
                params_dict = json.load(f)

            print(f"📋 模型参数: {params_dict}")

            # 先加载tokenizer来获取准确的vocab_size
            print("🔤 预加载tokenizer获取vocab_size...")
            tokenizer_path = os.path.join(self.model_path, "tokenizer.model")
            temp_tokenizer = spm.SentencePieceProcessor()
            temp_tokenizer.load(tokenizer_path)
            actual_vocab_size = temp_tokenizer.vocab_size()

            # 修复vocab_size问题
            if params_dict.get('vocab_size', -1) == -1:
                print(f"⚠️  检测到vocab_size为-1，使用tokenizer实际大小: {actual_vocab_size}")
                params_dict['vocab_size'] = actual_vocab_size
            else:
                print(f"📋 使用配置文件中的vocab_size: {params_dict['vocab_size']}")

            # 创建模型配置
            model_args = ModelArgs(**params_dict)

            # 初始化模型
            print("🏗️  初始化模型结构...")
            self.model = Transformer(model_args)

            # 加载权重
            print("📦 加载模型权重...")
            checkpoint_path = os.path.join(self.model_path, "consolidated.00.pth")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(checkpoint, strict=False)

            # 移动到设备
            print(f"🚀 移动模型到设备: {self.device}")
            self.model = self.model.to(self.device)
            self.model.eval()

            # 使用之前预加载的tokenizer
            print("🔤 使用预加载的tokenizer...")
            self.tokenizer = temp_tokenizer

            print("✅ 模型加载完成!")
            return True

        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            return False

    def encode(self, text: str):
        """编码文本为token"""
        return self.tokenizer.encode(text, out_type=int)

    def decode(self, tokens):
        """解码token为文本"""
        return self.tokenizer.decode(tokens)

    def format_chat_prompt(self, message: str, system_message: str = None):
        """格式化Llama-2 chat格式"""
        if system_message is None:
            system_message = "You are a helpful, respectful and honest assistant."

        prompt = f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{message} [/INST]"
        return prompt

    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7):
        """生成文本"""
        if self.model is None or self.tokenizer is None:
            return "❌ 模型未加载"

        # 编码输入
        tokens = self.encode(prompt)
        tokens = torch.tensor([tokens], dtype=torch.long).to(self.device)

        print(f"📏 输入tokens长度: {tokens.shape[1]}")

        generated_tokens = []
        start_pos = 0

        with torch.no_grad():
            for i in range(max_tokens):
                # 前向传播
                if i == 0:
                    # 第一次传入完整序列
                    logits = self.model.forward(tokens, start_pos)
                    start_pos = tokens.shape[1]
                else:
                    # 后续只传入新生成的token
                    logits = self.model.forward(next_token.unsqueeze(0), start_pos)
                    start_pos += 1

                # 获取最后一个位置的logits
                last_logits = logits[0, -1, :]

                # 应用temperature
                if temperature > 0:
                    last_logits = last_logits / temperature
                    probs = F.softmax(last_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(last_logits, dim=-1, keepdim=True)

                next_token = next_token.reshape(1, 1)

                # 检查是否为结束token
                if next_token.item() == 2:  # </s> token
                    break

                generated_tokens.append(next_token.item())

                # 拼接tokens用于下次输入
                tokens = torch.cat([tokens, next_token], dim=1)

        # 解码生成的文本
        if generated_tokens:
            generated_text = self.decode(generated_tokens)
            return generated_text.strip()
        else:
            return ""


def main():
    """主函数"""
    print("🏛️  Llama-2-7b-chat Meta格式直接加载测试")
    print("=" * 60)

    # 初始化模型
    model_path = "./Llama-2-7b-chat"
    llama_model = Llama2Model(model_path)

    # 加载模型
    if not llama_model.load_model():
        print("❌ 模型加载失败，退出程序")
        return

    # 测试用例
    test_message = "hello!"
    print(f"\n👤 用户输入: {test_message}")

    # 格式化为chat格式
    chat_prompt = llama_model.format_chat_prompt(test_message)
    print(f"🔤 格式化提示: {chat_prompt[:100]}...")

    # 生成回复
    print("🤖 模型回应:")
    response = llama_model.generate(chat_prompt, max_tokens=50, temperature=0.7)
    print(response)

    print("\n🎉 测试完成!")


if __name__ == "__main__":
    main()