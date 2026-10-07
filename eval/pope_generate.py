#!/usr/bin/env python3
"""Generate POPE answers with LLaVA-NeXT and ACG."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from acg import apply_acg
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


MODEL_CONFIGS = {
    "7b": {
        "path": "liuhaotian/llava-v1.6-vicuna-7b",
        "layers": 32,
    },
    "13b": {
        "path": "liuhaotian/llava-v1.6-vicuna-13b",
        "layers": 40,
    },
}
CONVERSATION_MODE = "llava_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, required=True)
    parser.add_argument("--question-file", type=Path, required=True)
    parser.add_argument("--image-folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--end-layer", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument(
        "--limit-images",
        type=int,
        help="Evaluate only the first N image records.",
    )
    parser.add_argument("--seed", type=int, default=927)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_records(path: Path) -> list[dict]:
    records = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    for index, record in enumerate(records):
        if not all(key in record for key in ("image", "text", "label")):
            raise ValueError(f"Malformed POPE record at line {index + 1}.")
        if len(record["text"]) != len(record["label"]):
            raise ValueError(f"Question/label length mismatch at line {index + 1}.")
    return records


def resolve_image(image_folder: Path, image_name: str) -> Path:
    original = image_folder / image_name
    if original.is_file():
        return original

    numeric_name = image_name.rsplit("_", 1)[-1]
    numeric = image_folder / numeric_name
    if numeric.is_file():
        return numeric

    raise FileNotFoundError(f"Could not resolve {image_name} under {image_folder}.")


def build_prompt(tokenizer, query: str) -> tuple[str, torch.Tensor]:
    conversation = conv_templates[CONVERSATION_MODE].copy()
    conversation.append_message(
        conversation.roles[0],
        DEFAULT_IMAGE_TOKEN + "\n" + query,
    )
    conversation.append_message(conversation.roles[1], None)
    prompt = conversation.get_prompt()
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0)
    return prompt, input_ids


def image_token_span(input_ids: torch.Tensor, model) -> tuple[int, int]:
    positions = torch.where(input_ids == IMAGE_TOKEN_INDEX)[1]
    if positions.numel() != 1:
        raise ValueError(f"Expected one image token, found {positions.numel()}.")
    image_start = int(positions.item())
    vision_tower = model.get_vision_tower()
    image_length = (
        vision_tower.config.image_size // vision_tower.config.patch_size
    ) ** 2
    return image_start, image_start + image_length


def generated_tokens(
    output_ids: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    input_length = input_ids.shape[1]
    if (
        output_ids.shape[1] >= input_length
        and torch.equal(output_ids[:, :input_length], input_ids)
    ):
        return output_ids[:, input_length:]
    return output_ids


def main() -> None:
    args = parse_args()
    config = MODEL_CONFIGS[args.model_size]
    end_layer = config["layers"] if args.end_layer is None else args.end_layer

    if not 0 <= args.start_layer <= end_layer <= config["layers"]:
        raise ValueError(
            f"Layer range must lie in [0, {config['layers']}] for "
            f"LLaVA-NeXT {args.model_size}."
        )
    if args.limit_images is not None and args.limit_images < 1:
        raise ValueError("--limit-images must be at least 1.")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this evaluation script.")

    set_seed(args.seed)
    disable_torch_init()

    model_path = config["path"]
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        None,
        get_model_name_from_path(model_path),
    )
    model.config.image_aspect_ratio = "pad"
    model.eval()

    records = load_records(args.question_file)
    if args.limit_images is not None:
        records = records[: args.limit_images]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        for record in tqdm(records, desc=f"POPE LLaVA-NeXT {args.model_size}"):
            image = Image.open(
                resolve_image(args.image_folder, record["image"])
            ).convert("RGB")
            image_tensor = image_processor(
                image, return_tensors="pt"
            )["pixel_values"].to(device=model.device, dtype=torch.float16)

            for query, label in zip(record["text"], record["label"]):
                prompt, input_ids = build_prompt(tokenizer, query)
                input_ids = input_ids.to(model.device)
                image_start, image_end = image_token_span(input_ids, model)

                if args.gamma != 0:
                    apply_acg(
                        model,
                        image_start=image_start,
                        image_end=image_end,
                        gamma=args.gamma,
                        start_layer=args.start_layer,
                        end_layer=end_layer,
                    )

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor,
                        do_sample=False,
                        num_beams=1,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                    )

                answer = tokenizer.batch_decode(
                    generated_tokens(output_ids, input_ids),
                    skip_special_tokens=True,
                )[0].strip()
                output_file.write(
                    json.dumps(
                        {
                            "query": query,
                            "label": 1 if label == "yes" else 0,
                            "ans": answer,
                            "question": prompt,
                            "file_path": args.output.stem,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                output_file.flush()


if __name__ == "__main__":
    main()
