"""FireRed-Image-Edit Inference - Adapted for 4x NVIDIA P40.

P40s don't support bfloat16 and FP16 causes numerical corruption (NaN/overflow)
in the diffusion scheduler and VAE. All arithmetic runs in FP32.

Quantization modes:
  int8  - Recommended. ~22GB transformer on GPU0, VAE on GPU2, text_enc on GPU1.
          ~88s/step, clean output. (3 GPUs used)
  nf4   - ~10GB transformer + VAE on GPU0, text_enc on GPU1.
          ~146s/step, noisy output due to 4-bit precision loss. (2 GPUs used)
  none  - FP16 split across GPUs 0-2. (Not recommended on P40 - FP16 corruption)
"""

import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel
from diffusers import BitsAndBytesConfig as DiffusersBnBConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FireRed-Image-Edit inference (P40)")
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
        choices=["nf4", "int8", "none"],
        default="nf4",
        help="Quantization mode: nf4 (4-bit, ~10GB), int8 (~20GB), none (FP16 split across GPUs)",
    )
    return parser.parse_args()


def _move_tensors(obj, device):
    """Recursively move tensors in nested structures to device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_tensors(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: _move_tensors(v, device) for k, v in obj.items()}
    return obj


def load_pipeline(model_path: str, quant: str) -> QwenImageEditPlusPipeline:
    """Load pipeline with optional quantization and manual device placement."""
    n_gpus = torch.cuda.device_count()
    print(f"Loading model from {model_path}...")
    print(f"Quantization mode: {quant}")
    print(f"Available GPUs: {n_gpus}")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")

    if quant in ("nf4", "int8"):
        # Load quantized transformer on cuda:0.
        if quant == "nf4":
            print("Loading transformer (NF4 quantized)...")
            quant_config = DiffusersBnBConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float32,
                bnb_4bit_use_double_quant=True,
            )
        else:
            print("Loading transformer (INT8 quantized)...")
            quant_config = DiffusersBnBConfig(load_in_8bit=True)

        transformer = QwenImageTransformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            quantization_config=quant_config,
            torch_dtype=torch.float32,
            device_map="cuda:0",
        )
        print("Transformer loaded on GPU 0 (non-quantized layers in FP32).")

        # Load pipeline in FP32 so scheduler/VAE arithmetic avoids FP16 overflow.
        # Text encoder will be cast back to FP16 to fit on a single GPU.
        print("Loading pipeline (FP32 for scheduler/VAE)...")
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            model_path,
            transformer=transformer,
            torch_dtype=torch.float32,
        )

        # INT8 transformer takes ~22GB on cuda:0, leaving no room for VAE decode.
        # NF4 transformer takes ~10GB, so VAE fits on cuda:0 alongside it.
        if quant == "int8":
            vae_device = torch.device("cuda:2")
        else:
            vae_device = torch.device("cuda:0")

        # Text encoder in FP16 on cuda:1 (16.6GB). VAE in FP32 on vae_device.
        print("Moving text_encoder to cuda:1 (FP16)...")
        pipe.text_encoder.to(device="cuda:1", dtype=torch.float16)
        print(f"Moving VAE to {vae_device} (FP32)...")
        pipe.vae.to(vae_device)

        # Override _execution_device so the pipeline creates latents and
        # prepares images on cuda:0 (where transformer lives).
        main_device = torch.device("cuda:0")
        text_enc_device = torch.device("cuda:1")
        pipe.__class__._execution_device = property(lambda self: main_device)
        print(f"Overrode _execution_device -> {main_device}")

        # If VAE is on a different device from the transformer, patch decode
        # to move latents to the VAE device and results back to main.
        if vae_device != main_device:
            original_vae_decode = pipe.vae.decode

            def patched_vae_decode(z, *args, **kwargs):
                z = z.to(vae_device)
                result = original_vae_decode(z, *args, **kwargs)
                if hasattr(result, 'sample'):
                    result.sample = result.sample.to(main_device)
                return result

            pipe.vae.decode = patched_vae_decode

            original_vae_encode = pipe.vae.encode

            def patched_vae_encode(x, *args, **kwargs):
                x = x.to(vae_device)
                result = original_vae_encode(x, *args, **kwargs)
                # Move latent tensors back to main device for the denoising loop.
                if hasattr(result, 'latent_dist'):
                    for attr in ('loc', 'scale', 'logvar', 'mean', 'std', 'var'):
                        if hasattr(result.latent_dist, attr):
                            val = getattr(result.latent_dist, attr)
                            if isinstance(val, torch.Tensor):
                                setattr(result.latent_dist, attr, val.to(main_device))
                return result

            pipe.vae.encode = patched_vae_encode
            print(f"Patched VAE encode/decode for cross-device transfer ({vae_device} <-> {main_device})")

        # Monkey-patch encode_prompt: redirect device arg to cuda:1
        # (text_encoder device), then move outputs back to cuda:0.
        original_encode_prompt = pipe.encode_prompt

        def patched_encode_prompt(*args, **kwargs):
            # encode_prompt(prompt, image, device, ...) - device is 3rd positional
            args = list(args)
            if len(args) >= 3:
                args[2] = text_enc_device
            if "device" in kwargs:
                kwargs["device"] = text_enc_device
            result = original_encode_prompt(*args, **kwargs)
            # Move embeddings to cuda:0 and upcast to FP32 for scheduler math.
            return tuple(
                r.to(device=main_device, dtype=torch.float32) if isinstance(r, torch.Tensor) else r
                for r in result
            )

        pipe.encode_prompt = patched_encode_prompt
        print("Patched encode_prompt: inputs -> cuda:1, outputs -> cuda:0")

    else:
        # No quantization: FP16 split across GPUs 0-2.
        print("Loading transformer (FP16, split across GPUs)...")
        max_mem = {i: "22GiB" for i in range(min(n_gpus, 3))}
        if n_gpus > 3:
            max_mem[3] = "0GiB"
        transformer = QwenImageTransformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory=max_mem,
        )
        print("Transformer loaded.")

        print("Loading pipeline with balanced device distribution...")
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            model_path,
            transformer=transformer,
            torch_dtype=torch.float16,
            device_map="balanced",
        )

    if hasattr(pipe, 'hf_device_map'):
        print(f"Pipeline device map: {pipe.hf_device_map}")

    print("Pipeline loaded successfully.")
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

    # Check for invalid output (all black / NaN corruption)
    img_array = np.array(output_image)
    mean_val = img_array.mean()
    print(f"Output image stats: mean={mean_val:.1f}, min={img_array.min()}, max={img_array.max()}")
    if mean_val < 1.0:
        print("WARNING: Output is essentially black - likely numerical corruption")

    output_image.save(args.output_image)
    print(f"Output saved to: {args.output_image.resolve()}")


if __name__ == "__main__":
    main()
