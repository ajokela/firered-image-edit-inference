# FireRed-Image-Edit Inference Scripts

Inference scripts for running [FireRed-Image-Edit-1.0](https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.0), a 57.7GB diffusion-based image editing model, on different GPU platforms.

## Scripts

### `inference_p40.py` — NVIDIA Tesla P40 (4x, 24GB each)

Runs FireRed on four Tesla P40 GPUs (Pascal, 2016) with multi-GPU device orchestration:

- **INT8 quantization** (recommended): Clean output, ~87.5s/step, 58 min for 40 steps
- **NF4 quantization**: Faster but noisy output due to 4-bit precision loss
- **FP32 pipeline**: Required on Pascal — FP16 causes silent NaN corruption producing all-black images

Key challenges solved:
- FP16 numerical overflow in the diffusion scheduler and VAE (switched to FP32 everywhere)
- Multi-GPU device placement: transformer on GPU 0, text encoder on GPU 1, VAE on GPU 2
- Cross-device tensor transfer patches for VAE encode/decode (INT8 mode)
- `_execution_device` and `encode_prompt` monkey-patches for correct device routing

```bash
# INT8 quantization (recommended)
python inference_p40.py --quant int8 --num_inference_steps 40

# NF4 quantization (faster but noisy)
python inference_p40.py --quant nf4 --num_inference_steps 40
```

### `inference_strix.py` — AMD Strix Halo (Ryzen AI MAX+ 395, 96GB VRAM)

Runs FireRed on AMD's Strix Halo APU with unified memory:

- **BF16 full precision** (recommended): Clean output, ~82.6s/step, 55 min for 40 steps
- No quantization needed — 96GB unified VRAM fits the entire model
- Single GPU, no device orchestration, ~50 lines of code

```bash
# BF16 full precision (recommended)
python inference_strix.py --num_inference_steps 40

# INT8/NF4 also supported but don't improve speed (compute-bound workload)
python inference_strix.py --quant int8 --num_inference_steps 10
```

### `inference_original.py` — Reference Script

The original BF16 inference script from the FireRed repository. Requires a single GPU with ~75GB+ VRAM (e.g., A100 80GB). Included for reference.

## Performance Comparison

| System | Configuration | Per-Step | 40 Steps | Quality |
|--------|--------------|----------|----------|---------|
| Strix Halo | BF16 full precision | **82.6s** | **55 min** | Clean |
| 4x P40 | INT8 + FP32 pipeline | 87.5s | 58 min | Clean |
| 4x P40 | NF4 + FP32 pipeline | 145.9s | 97 min | Noisy |

## Requirements

**P40 (NVIDIA, CUDA):**
- Python 3.12+
- PyTorch 2.6+ with CUDA 12.x
- `diffusers` (from git main: `pip install git+https://github.com/huggingface/diffusers.git`)
- `bitsandbytes >= 0.49`
- `transformers`, `accelerate`, `qwen-vl-utils`, `Pillow`, `numpy`

**Strix Halo (AMD, ROCm):**
- Python 3.12+
- PyTorch 2.7+ with ROCm 7.9
- `diffusers` (from git main)
- `bitsandbytes >= 0.49` (only needed for `--quant` modes)
- `transformers`, `accelerate`, `qwen-vl-utils`, `Pillow`, `numpy`

## Blog Post

For the full write-up including debugging history and architectural details, see:
[Image Editing on 8-Year-Old GPUs: NVIDIA P40 vs AMD Strix Halo](https://tinycomputers.io/posts/image-editing-on-8-year-old-gpus-nvidia-p40-vs-amd-strix-halo.html)
