"""Smoke test: can this Modal workspace get an L4, and do INT8/FP8 matmuls run on it?
Run: modal run pilot/gpu_smoke.py"""
import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.8.0")
app = modal.App("thirdmoment-gpu-smoke", image=image)


@app.function(gpu="L4", timeout=300)
def smoke() -> dict:
    import subprocess

    import torch

    out = {"nvidia_smi": subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()}
    out["torch"] = str(torch.__version__)
    out["cuda"] = str(torch.version.cuda)
    out["capability"] = tuple(int(c) for c in torch.cuda.get_device_capability())

    a = torch.randint(-128, 127, (64, 256), dtype=torch.int8, device="cuda")
    b = torch.randint(-128, 127, (256, 128), dtype=torch.int8, device="cuda")
    try:
        torch._int_mm(a, b); out["int8_int_mm"] = "ok"
    except Exception as e:
        out["int8_int_mm"] = repr(e)[:200]

    x = torch.randn(64, 256, device="cuda").to(torch.float8_e4m3fn)
    w = torch.randn(128, 256, device="cuda").to(torch.float8_e4m3fn).t()
    one = torch.tensor(1.0, device="cuda")
    try:
        torch._scaled_mm(x, w, scale_a=one, scale_b=one, out_dtype=torch.bfloat16); out["fp8_per_tensor"] = "ok"
    except Exception as e:
        out["fp8_per_tensor"] = repr(e)[:200]
    try:
        torch._scaled_mm(x, w, scale_a=torch.ones(64, 1, device="cuda"),
                         scale_b=torch.ones(1, 128, device="cuda"), out_dtype=torch.bfloat16)
        out["fp8_per_row"] = "ok"
    except Exception as e:
        out["fp8_per_row"] = repr(e)[:200]
    return out


@app.local_entrypoint()
def main():
    for k, v in smoke.remote().items():
        print(f"{k:16} {v}")
