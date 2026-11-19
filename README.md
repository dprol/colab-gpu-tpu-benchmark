# Comparing GPUs and TPUs on Google Colab

COSC 6385 Computer Architecture — Assignment 2 (Experiment).
Benchmarks a free-tier **NVIDIA T4 GPU** and a **Google TPU v5e-1** on Colab with
one JAX codebase, so measured differences come from the hardware and XLA, not
from different code.

* **Part 1** — n×n matrix-multiply microbenchmark in three numeric modes
  (`fp32-highest`, `fp32-default`, `bf16`); reports achieved TFLOPS and % of peak.
* **Part 2** — a 3-block CNN on CIFAR-10 (Flax NNX) across precisions (fp32, bf16)
  and batch sizes (128, 512); reports images/s, step time, test accuracy and cost.
* **Part 3** — LoRA fine-tuning of Gemma (Keras 3 + JAX backend) on the TPU and
  the GPU; compares time per epoch and sample generations.

## How to run
1. Open `colab_gpu_tpu_benchmark.ipynb` in Google Colab.
2. `Runtime → Change runtime type → T4 GPU`, then **Run all**.
3. Download `results_matmul.csv` and `results_cnn.csv` (Colab deletes local files
   when a runtime is recycled).
4. Repeat with **TPU v5e-1**.
5. Upload both CSVs (e.g. `results_matmul_T4.csv`) and run the final cell to merge
   them and write `comparison.png`.
6. Part 3 follows Google's *Fine-tune Gemma in Keras using LoRA* tutorial after
   adding `KAGGLE_USERNAME` / `KAGGLE_KEY` to Colab Secrets.

Do **not** run `pip install -U jax` on a TPU runtime — it breaks the JAX/libtpu
pairing. Use the version Colab preinstalls.

## Files
- `colab_gpu_tpu_benchmark.ipynb` — the starter notebook (Parts 1–2 + merge/plot).
- `gemma_lora_part3.ipynb` — Part 3: Gemma LoRA fine-tune (Keras 3 / JAX backend),
  run unchanged on T4 then TPU.
- `Experiment-Colab-GPU-TPU-report.pdf` / `.docx` — the report (figures embedded).
- `comparison_T4.png` / `comparison_TPU.png` — the merged plots per device
  (matmul throughput + CNN training throughput), used as Figures 1–2 in the report.
- `results/` — merged `results_matmul.csv` and `results_cnn.csv` go here
  (submitted to the course Team together with the report PDF).

## Deliverables
Per the assignment, the **report PDF and the merged CSVs** are submitted to the
course Team; this repository mirrors them plus the notebook.
