"""FireRed-Image-Edit Inference - AMD Strix Halo (Ryzen AI MAX+ 395).

96GB unified VRAM, RDNA 3.5 (gfx1151), BF16 supported.
Supports full-precision BF16, INT8, or NF4 quantization.
"""

import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel
from diffusers import BitsAndBytesConfig as DiffusersBnBConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FireRed-Image-Edit inference (Strix Halo)")
    parser.add_argument("--model_path", type=str, default="FireRedTeam/FireRed-Image-Edit-1.0")
    parser.add_argument("--input_image", type=Path, default=Path("./examples/edit_example.png"))
    parser.add_argument("--output_image", type=Path, default=Path("output_edit.png"))
    parser.add_argument("--prompt", type=str, default="Add a red hat on the cat")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--true_cfg_scale", type=float, default=4.0)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument(
        "--quant",
        type=str,
        choices=["none", "int8", "nf4"],
        default="none",
        help="Quantization: none (BF16 full precision), int8, or nf4",
    )
    return parser.parse_args()


def load_pipeline(model_path: str, quant: str) -> QwenImageEditPlusPipeline:
    """Load pipeline on cuda:0 with optional quantization."""
    props = torch.cuda.get_device_properties(0)
    print(f"Loading model from {model_path}...")
    print(f"Quantization: {quant}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {props.total_memory / 1e9:.1f} GB")

    if quant == "none":
        print("Loading pipeline (BF16 full precision)...")
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        )
        pipe.to("cuda:0")
    else:
        if quant == "nf4":
            print("Loading transformer (NF4 quantized)...")
            quant_config = DiffusersBnBConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            print("Loading transformer (INT8 quantized)...")
            quant_config = DiffusersBnBConfig(load_in_8bit=True)

        transformer = QwenImageTransformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
        )
        print("Transformer loaded.")

        print("Loading pipeline (BF16)...")
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            model_path,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        pipe.to("cuda:0")

    print("Pipeline loaded on cuda:0.")
    return pipe


def main() -> None:
    args = parse_args()

    pipeline = load_pipeline(args.model_path, args.quant)

    image = Image.open(args.input_image).convert("RGB")
    print(f"Input image: {args.input_image} ({image.size})")
    print(f"Prompt: {args.prompt}")

    inputs = {
        "image": [image],
        "prompt": args.prompt,
        "generator": torch.Generator("cpu").manual_seed(args.seed),
        "true_cfg_scale": args.true_cfg_scale,
        "negative_prompt": " ",
        "num_inference_steps": args.num_inference_steps,
        "num_images_per_prompt": 1,
    }
    if args.height:
        inputs["height"] = args.height
    if args.width:
        inputs["width"] = args.width

    print(f"Running inference ({args.num_inference_steps} steps, cfg={args.true_cfg_scale})...")
    with torch.inference_mode():
        result = pipeline(**inputs)

    output_image = result.images[0]

    img_array = np.array(output_image)
    mean_val = img_array.mean()
    print(f"Output image stats: mean={mean_val:.1f}, min={img_array.min()}, max={img_array.max()}")
    if mean_val < 1.0:
        print("WARNING: Output is essentially black - likely numerical corruption")

    output_image.save(args.output_image)
    print(f"Output saved to: {args.output_image.resolve()}")


if __name__ == "__main__":
    main()
