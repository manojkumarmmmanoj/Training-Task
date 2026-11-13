# NimbusPay Ticket Triage Fine-Tuning

**Machine Learning Intern Take-Home Assignment – LLM Fine-Tuning, VIMA3YA**

- **Base Model ID:** [`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) (3.09B parameters)
- **Loaded in 4-bit:** **Yes** (BitsAndBytes NormalFloat4 `NF4` with double quantization and float16 compute)
- **One-Line Summary of Best Result:** **84.5% Exact Match and 96.8% Mean Field Accuracy on `dev.jsonl` using 4-bit QLoRA ($r=16, \alpha=32$) trained on audited & cleaned data with completion-only loss masking.**

---

## Repository Structure

```
.
├── README.md               # Base model ID, 4-bit info, best results, reproduction guide
├── report.md               # 3-page engineering report (Audit, Baselines, Experiments, Error Analysis)
├── notebook.ipynb          # End-to-end reproducible Google Colab notebook with outputs
├── cleaning_log.csv        # Log of all 198 audited rows (id, action, reason)
├── predictions_test.jsonl  # Model predictions for all 400 test inputs
├── adapter/                # Final PEFT LoRA adapter
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── README.md
├── score.py                # Official strict evaluation script
├── SCHEMA.md               # Ground-truth NimbusPay v2 labelling specification
└── data/
    ├── train.jsonl         # Original raw training dataset (1,445 tickets)
    ├── train_cleaned.jsonl # Cleaned training dataset (1,410 tickets)
    ├── dev.jsonl           # Clean dev evaluation benchmark (200 tickets)
    └── test_inputs.jsonl   # Test inputs for submission (400 tickets)
```

---

## Key Results Summary

All metrics scored strictly using `score.py`:

| Model / Pipeline Stage | Dev Exact Match | Dev Mean Field Acc | Peak GPU VRAM | Train Time |
|---|:---:|:---:|:---:|:---:|
| **Zero-Shot Base Model (Fixed Prompt)** | 4.0% | 58.6% | — | — |
| **In-Context Rules Prompt Baseline** | 31.5% | 79.2% | — | — |
| **Fine-Tuned on Raw Data (4-bit QLoRA)** | 62.0% | 89.5% | 4.1 GB | 11.2 min |
| **Fine-Tuned on Cleaned Data (4-bit QLoRA, Best)** | **84.5%** | **96.8%** | **4.1 GB** | **10.8 min** |
| **16-bit LoRA (Ablation)** | 85.0% | 97.0% | 8.4 GB | 14.2 min |

---

## Quickstart: Reproducing Results

### 1. Run in Google Colab (Under 3 Hours on Free T4)
Open [`notebook.ipynb`](notebook.ipynb) directly in Google Colab with a free T4 GPU runtime:
1. In Colab, select **Runtime $\rightarrow$ Change runtime type $\rightarrow$ T4 GPU**.
2. Run all cells sequentially. The notebook executes top-to-bottom:
   - Inspects and cleans `data/train.jsonl` into `data/train_cleaned.jsonl` and generates `cleaning_log.csv`.
   - Runs zero-shot and in-context baselines on `data/dev.jsonl`.
   - Fine-tunes `Qwen/Qwen2.5-3B-Instruct` in 4-bit QLoRA for 3 epochs (~11 minutes).
   - Evaluates dev performance using `score.py`.
   - Performs the error analysis on dev failure cases.
   - Generates `predictions_test.jsonl` for all 400 test tickets.
   - Saves the final PEFT adapter into `adapter/`.

### 2. Standalone Inference with PEFT Adapter
```python
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-3B-Instruct"
adapter_dir = "./adapter"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, adapter_dir)
model.eval()

# Example ticket
messages = [
    {"role": "system", "content": "You are NimbusPay's ticket triage model. Read the ticket and return the triage JSON."},
    {"role": "user", "content": "Received: 2026-08-14 (Fri)\nSource: email\n\nI was charged twice for Rs 1,450 on UPI. Txn ID NP1234567890. Please refund."}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False  # Greedy decoding
    )

response = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
print(response)
```

### 3. Verify Scoring with `score.py`
```bash
python score.py --gold data/dev.jsonl --pred dev_predictions.jsonl
```
