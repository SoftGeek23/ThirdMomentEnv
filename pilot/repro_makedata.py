"""Reproduce the verifier image's make_data.py --split test step to get its traceback."""
from pathlib import Path

import modal

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.8.0", "torchao==0.13.0", "transformers==4.56.2", "huggingface-hub==0.35.3",
                      "datasets==4.1.1", "numpy==2.3.3")
         .add_local_dir(str(Path(__file__).parent / "repro"), remote_path="/repro"))
app = modal.App("thirdmoment-repro-makedata", image=image)


@app.function(cpu=4, memory=16384, timeout=1200)
def run():
    import subprocess
    p = subprocess.run(["python", "/repro/make_data.py", "--split", "test", "--out", "/tmp/eval.json"],
                       capture_output=True, text=True)
    return p.returncode, p.stdout[-2000:], p.stderr[-6000:]


@app.local_entrypoint()
def main():
    rc, out, err = run.remote()
    print("RC", rc)
    print(out)
    print(err)
