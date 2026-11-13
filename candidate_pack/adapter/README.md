---
base_model: Qwen/Qwen2.5-3B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- qlora
- ticket-triage
- nimbuspay
---

# LoRA Adapter for NimbusPay Ticket Triage

This adapter is fine-tuned on top of `Qwen/Qwen2.5-3B-Instruct` using 4-bit QLoRA to parse customer support tickets into structured JSON triage records according to NimbusPay's Schema v2.

## Model Details
- **Base Model**: `Qwen/Qwen2.5-3B-Instruct`
- **PEFT Type**: LoRA (r=16, alpha=32, dropout=0.05)
- **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Trainable Parameters**: 20.2M (~0.65% of base model)
- **Precision**: 4-bit NormalFloat (NF4) with Double Quantization
