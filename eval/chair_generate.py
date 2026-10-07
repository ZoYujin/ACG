#!/usr/bin/env python3
"""Generate captions for the fixed CHAIR subset with LLaVA-NeXT and ACG."""

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
from llava.mm_utils import (
    get_model_name_from_path,
    tokenizer_image_token,
)
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
PROMPT = "Please describe this image in detail."
CONVERSATION_MODE = "llava_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, required=True)
    parser.add_argument("--image-folder", type=Path, required=True)
    parser.add_argument(
        "--image-ids",
        type=Path,
        default=Path("data/chair_image_ids.txt"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--end-layer", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N images.")
    parser.add_argument("--seed", type=int, default=927)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_image_ids(path: Path) -> list[int]:
    image_ids = [
        int(line.strip())
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not image_ids:
        raise ValueError(f"No image IDs found in {path}.")
    return image_ids


def image_path(image_folder: Path, image_id: int) -> Path:
    candidates = (
        image_folder / f"COCO_val2014_{image_id:012d}.jpg",
        image_folder / f"{image_id:012d}.jpg",
        image_folder / f"{image_id}.jpg",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find COCO image {image_id} under {image_folder}."
    )


def build_prompt(tokenizer) -> torch.Tensor:
    conversation = conv_templates[CONVERSATION_MODE].copy()
    conversation.append_message(
        conversation.roles[0],
        DEFAULT_IMAGE_TOKEN + "\n" + PROMPT,
    )
    conversation.append_message(conversation.roles[1], None)
    return tokenizer_image_token(
        conversation.get_prompt(),
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0)


def image_token_span(input_ids: torch.Tensor, model) -> tuple[int, int]:
    positions = torch.where(input_ids == IMAGE_TOKEN_INDEX)[1]
    if positions.numel() != 1:
        raise ValueError(
            f"Expected one image token, found {positions.numel()}."
        )
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
    """Handle both legacy and current LLaVA generate return conventions."""
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

    image_ids = load_image_ids(args.image_ids)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1.")
        image_ids = image_ids[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as output_file:
        for image_id in tqdm(image_ids, desc=f"LLaVA-NeXT {args.model_size}"):
            path = image_path(args.image_folder, image_id)
            image = Image.open(path).convert("RGB")
            image_tensor = image_processor(
                image, return_tensors="pt"
            )["pixel_values"].to(device=model.device, dtype=torch.float16)

            input_ids = build_prompt(tokenizer).to(model.device)
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

            caption = tokenizer.batch_decode(
                generated_tokens(output_ids, input_ids),
                skip_special_tokens=True,
            )[0].strip()
            output_file.write(
                json.dumps(
                    {"image_id": image_id, "caption": caption},
                    ensure_ascii=False,
                )
                + "\n"
            )
            output_file.flush()


if __name__ == "__main__":
    main()
