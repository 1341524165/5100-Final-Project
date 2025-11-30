# DPO Final Project Plan (Custom Implementation)

## Project Goal

Implement and analyze Direct Preference Optimization (DPO) with a focus on:

1.  **Noise Robustness**: How does DPO perform when preference labels are noisy?
2.  **Multi-Objective Optimization**: Can DPO optimize for multiple conflicting objectives?

## Current Architecture

The project is built on a custom DPO implementation located in the `DPO/` directory, removing the dependency on the `trl` library for core algorithms.

### Directory Structure

```
5100-Final-Project/
├── DPO/
│   ├── config.py       # Configuration dataclasses
│   ├── data.py         # Dataset handling (HH-RLHF)
│   ├── losses.py       # Core DPO loss implementation
│   ├── models.py       # Model loading & LoRA support
│   ├── train.py        # Training loop
│   ├── train_dpo.py    # Main entry point
│   └── utils.py        # Utilities
├── results/            # Experiment outputs
└── README.md
```

---

## Implementation Roadmap

### Phase 1: Baseline Verification (Current)

**Goal**: Ensure the custom DPO implementation works correctly on the HH-RLHF dataset.

- [ ] Run `python DPO/train_dpo.py` with a small model (e.g., GPT-2 or Pythia-160m) to verify the pipeline.
- [ ] Check loss convergence.

### Phase 2: Noise Robustness (Innovation 1)

**Goal**: Test DPO performance under label noise (randomly flipping 'chosen' vs 'rejected').

- [ ] **Modify `DPO/data.py`**: Add a `noise_rate` parameter to `PreferenceDataset`.
  - Implement label flipping logic: with probability `p`, swap `chosen` and `rejected`.
- [ ] **Experiment**: Train models with noise rates [0.0, 0.1, 0.2, 0.3].
- [ ] **Analysis**: Compare evaluation loss and win-rates (if possible) across noise levels.

### Phase 3: Multi-Objective DPO (Innovation 2)

**Goal**: Extend DPO to optimize multiple rewards simultaneously.

- [ ] **Modify `DPO/losses.py`**: Implement `multi_objective_dpo_loss`.
  - $L_{total} = \sum w_i L_{DPO_i}$
- [ ] **Modify `DPO/data.py`**: Support datasets with multiple reward signals (or simulate them).
- [ ] **Experiment**: Train with varying weights for different objectives.

---

## Quick Start

```bash
chmod +x run_noise_sweep.sh
./run_noise_sweep.sh
# Train Baseline
python DPO/train_dpo.py --model_name gpt2 --batch_size 2 --output_dir outputs/baseline

python DPO/train_dpo.py --model_name gpt2 --batch_size 2 --num_epochs 1 --noise_rate 0.1 --output_dir outputs/noise_10

python experiments/run_multi_objective.py --model_name gpt2 --batch_size 16 --num_epochs 1 --base_output outputs/mo_groupA --weights_list "base=0.6,brevity=0.4;base=0.75,brevity=0.25;base=0.85,brevity=0.15" --brevity_coefs "0.02"

python experiments/run_multi_objective.py --model_name gpt2 --batch_size 16 --num_epochs 1 --base_output outputs/mo_groupB --weights_list "base=0.6,brevity=0.4;base=0.75,brevity=0.25;base=0.85,brevity=0.15" --brevity_coefs "0.01"

python experiments/run_multi_objective.py --model_name gpt2 --batch_size 16 --num_epochs 1 --base_output outputs/mo_groupC --weights_list "base=0.6,brevity=0.4;base=0.75,brevity=0.25;base=0.85,brevity=0.15" --brevity_coefs "0.0"
```
