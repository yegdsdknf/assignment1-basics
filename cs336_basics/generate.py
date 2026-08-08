import argparse
import json
from pathlib import Path

import torch

from cs336_basics.model import TransformerLM, softmax
from cs336_basics.runtime import resolve_device, set_random_seed
from cs336_basics.tokenizer import Tokenizer

END_OF_TEXT = "<|endoftext|>"


def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    if temperature < 0:
        raise ValueError("temperature 不能为负数")
    if not 0 < top_p <= 1:
        raise ValueError("top_p 必须位于 (0, 1]")

    if temperature == 0:
        return int(torch.argmax(logits).item())

    probabilities = softmax(logits / temperature, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
    remove_mask = torch.cumsum(sorted_probs, dim=-1) > top_p

    # 保留第一个使累计概率越过 top_p 的 token。
    remove_mask[1:] = remove_mask[:-1].clone()
    remove_mask[0] = False

    sorted_probs[remove_mask] = 0
    sampled_position = torch.multinomial(sorted_probs / sorted_probs.sum(), num_samples=1)
    return int(sorted_indices[sampled_position].item())


@torch.inference_mode()
def generate(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    context_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    end_token_id: int,
    device: torch.device,
) -> str:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens 必须为正数")

    model.eval()
    generated_ids = tokenizer.encode(prompt) or [end_token_id]

    for _ in range(max_new_tokens):
        input_tensor = torch.tensor(
            generated_ids[-context_length:],
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)

        next_token_id = sample_next_token(
            logits=model(input_tensor)[0, -1],
            temperature=temperature,
            top_p=top_p,
        )
        generated_ids.append(next_token_id)

        if next_token_id == end_token_id:
            break

    return tokenizer.decode(generated_ids)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 TinyStories LM 生成文本")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="训练实验的 run_config.json")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args(argv)


def load_model(checkpoint_path: Path, config: dict, device: torch.device) -> tuple[TransformerLM, int]:
    model_config = config["model"]
    dataset_config = config["dataset"]
    context_length = model_config["context_length"]

    # 先在 CPU 读取 checkpoint，避免把不需要的优化器状态加载到 GPU。
    model = TransformerLM(
        vocab_size=dataset_config["vocab_size"],
        context_length=context_length,
        d_model=model_config["d_model"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        d_ff=model_config["d_ff"],
        rope_theta=model_config["rope_theta"],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, context_length


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.config.is_file():
        raise FileNotFoundError(f"找不到配置文件：{args.config}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"找不到 checkpoint：{args.checkpoint}")

    device = resolve_device(args.device)
    set_random_seed(args.seed)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    dataset_config = config["dataset"]

    tokenizer = Tokenizer.from_files(
        vocab_filepath=Path(dataset_config["tokenizer_vocab"]),
        merges_filepath=Path(dataset_config["tokenizer_merges"]),
        special_tokens=[END_OF_TEXT],
    )
    model, context_length = load_model(args.checkpoint, config, device)

    print(
        generate(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            context_length=context_length,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            end_token_id=tokenizer.special_to_id[END_OF_TEXT],
            device=device,
        )
    )


if __name__ == "__main__":
    main()
