# Community Detection and Recommender Systems

This repository explores community-aware recommendation on heterogeneous user–item graphs. We combine meta-path evidence with (i) strong graph encoders (HAN/HGT), (ii) LLM-based training via GRPO with a stepwise reward, and (iii) supervised fine-tuning (SFT) with rationale distillation to elicit longer, evidence-grounded explanations while maintaining accurate scalar predictions.

---

### Datasets & Models

- **Full graph-text dataset (30 meta-paths per (user,item))** — concise JSON records with user/item attributes and rich path evidence: **[link](https://huggingface.co/datasets/mohammad-shirkhani/social_movielens_new2)**  
- **Compressed dataset for GRPO (5 meta-paths per sample)** — same schema, smaller context for RL-style training: **[link](https://huggingface.co/datasets/mohammad-shirkhani/social_movielens_compress)**  
- **SFT dataset (20k train samples, each with 20 meta-paths + teacher rationale)** — pairs of `<reason>` + `<answer>` distilled from a compact teacher: **[link](https://huggingface.co/datasets/mohammad-shirkhani/social_movielens_custom_with_reason)**  
- **GRPO post-trained model (stepwise reward)** — Qwen2.5-1.5B adapted to calibrated numeric predictions under our reward design: **[link](https://huggingface.co/mohammad-shirkhani/Qwen2.5-1.5B-GRPO-rating-new)**  
- **SFT model (LoRA on 20k rationale-augmented samples)** — Qwen2.5-7B that generates longer, path-citing explanations before the final score: **[link](https://huggingface.co/mohammad-shirkhani/qwen2.5_7b_rating_SFT)**

---

### Repository Structure (Methods)

1. **`reimplement_baseline/`** – Faithful reproduction of standard MovieLens-style recommenders and path-free baselines for head-to-head comparison.  
2. **`transformer/`** – Plain Transformer encoders over textified meta-paths and attributes; no hetero-graph operators, acts as a sequence baseline.  
3. **`HAN/`** – Heterogeneous Attention Network with curated meta-paths and self-loops; link prediction head for rating regression.  
4. **`GRPO/`** – Reinforcement-style post-training of an LLM using a stepwise numeric-accuracy reward plus format constraints to stabilize outputs.  
5. **`SFT/`** – Supervised fine-tuning with teacher-distilled `<reason>`→`<answer>` targets (20k samples, 20 meta-paths each) to teach longer, evidence-first rationales.  
6. **`HGT/`** – Heterogeneous Graph Transformer with type- and relation-specific projections for message passing across user/item/super-node types.  
7. **`comper_RL/`** – COMPER-inspired RL variants on meta-path composition and path selection; interfaces for sampling, scoring, and policy updates.

---
