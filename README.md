# Policy-invariant reward shaping from LLM feedback

Reference implementation for the paper **[Policy-Invariant Reward Shaping from LLM Feedback: A Framework for Hybrid RL Agents](https://arxiv.org/abs/2608.18008)** (arXiv:2608.18008, cs.LG, August 2026).

**Authors:** [Christophe D. Hounwanou](https://www.linkedin.com/in/chrishounwanou/)<sup>1,2</sup>, [John E. Eze](https://www.linkedin.com/in/john-e-eze/)<sup>1</sup>, [Yaé U. Gaba](https://www.linkedin.com/in/gabayae/)<sup>2,3,4</sup>

<sup>1</sup> African Institute for Mathematical Sciences, Rwanda &nbsp;·&nbsp; <sup>2</sup> AIRINA Labs, AI.Technipreneurs, Bénin &nbsp;·&nbsp; <sup>3</sup> Sefako Makgatho Health Sciences University, South Africa &nbsp;·&nbsp; <sup>4</sup> African Center for Advanced Studies, Cameroon

[Abstract](https://arxiv.org/abs/2608.18008) · [PDF](https://arxiv.org/pdf/2608.18008) · [HTML](https://arxiv.org/html/2608.18008v1)

[![arXiv](https://img.shields.io/badge/arXiv-2608.18008-b31b1b.svg)](https://arxiv.org/abs/2608.18008)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![CPU-runnable](https://img.shields.io/badge/compute-CPU%20only-lightgrey.svg)](#compute-requirements)

---

## TL;DR

We formalize the LLM-planner + RL-controller architecture as a **Goal-Augmented MDP (GA-MDP)** and prove (Proposition 4.1) that when the LLM's per-state progress score is used as a **bounded potential function** in the sense of Ng, Harada & Russell (1999), the resulting shaping term **cannot change the set of optimal policies** — no matter how wrong the LLM's scores are at individual states. A bad potential can slow learning; it cannot corrupt the fixed point. This is a guarantee that general LLM-as-reward approaches (Text2Reward, Eureka) do not provide.

This repository contains the minimal, fully auditable implementation of that pipeline, a numerical verification of the guarantee, and the two small-scale validations reported in the paper.

---

## What this repository contains

| Component | What it does | Paper |
|---|---|---|
| **GA-MDP construction** | Augmented state `(s, g)`, subgoal transition kernel driven by a pluggable `Done` oracle | §3, §4.1 |
| **LLM planner** | Emits an ordered subgoal list from the task string and a state caption (prompt P1) | §4.2, App. A |
| **Subgoal-conditioned policy** | PPO (sb3) conditioned on state + frozen MiniLM-L6-v2 subgoal embedding (d = 384) | §4.3 |
| **Potential-based shaping** | `Φ(s,g) = c · s_LLM(s,g)` with `s_LLM ∈ [0,1]` (prompt P2); `F = γΦ(s',g) − Φ(s,g)` | §4.4 |
| **Numerical verification of Prop. 4.1** | Exact enumeration on a 3-state / 2-action MDP under four Φ configurations | §4.4, Table 1 |
| **Replanning + Algorithm 1** | Periodic (every `H` steps) and failure-triggered (budget `B`) replanning (prompt P4) | §4.5 |
| **Done-oracle variants** | (i) environment event flags, (ii) learned MLP classifier, (iii) LLM completion prompt (P3) | §4.2, §6.3 |
| **Planner-in-isolation audit** | Grades Qwen-2.5:14b plans against hand-curated ground truth on 20 MiniGrid tasks | §4.7.1, Table 2 |
| **Pipeline validation** | PPO vs. hybrid on `MiniGrid-DoorKey-6x6-v0`, 3 seeds, IQM ± 95% bootstrap CI | §4.7.2, §5, Table 3 |

Every module maps 1:1 onto a definition or algorithm in the paper, so the code can be read alongside Section 4.

### Repository layout

```
.
├── HybridLLM_RL_Notebook.ipynb     # Walk-through notebook covering every component above
├── code/
│   ├── verify/prop1_numerical.py   # Table 1: numerical check of Proposition 4.1
│   ├── audit/tasks.json            # 20 MiniGrid tasks with ground-truth subgoal decompositions
│   └── ...                         # Planner, PotentialShaper, SubgoalScheduler, ReplanTrigger
├── configs/                        # YAML configs: reference hybrid + ablations
│                                   #   (subgoal source, shaping, replanning, Done oracle, LLM backbone)
├── tests/                          # 26 unit tests (25 run offline; 1 skipped without a live LLM)
├── requirements.txt
└── README.md
```

> Adjust this tree to match the actual repository contents.

---

## Quick start

### Install

```bash
git clone https://github.com/chrishounwanou/hybrid-llm-rl-framework.git
cd hybrid-llm-rl-framework
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Core dependencies: `numpy`, `gymnasium`, `minigrid`, `stable-baselines3`, `sentence-transformers`, `jupyter`.

### Reproduce the main theoretical result (no LLM needed, < 1 min on CPU)

```bash
python code/verify/prop1_numerical.py
```

This enumerates all four deterministic policies of the 3-state MDP (γ = 0.9) under the four potential configurations used in Table 1 and asserts that the optimal policy is unchanged in every case:

| Φ configuration | Base V*(s₀) | Shaped Ṽ*(s₀) | Optimal policy preserved? |
|---|---|---|---|
| Φ ≡ 0 | 0.9000 | 0.9000 | yes |
| Φ ≡ 1 | 0.9000 | 0.7100 | yes |
| sign-flip | 0.9000 | 1.4000 | yes |
| adversarial (≈ 20× reward scale) | 0.9000 | −5.0500 | yes |

The shaped value shifts by −Φ(s₀) plus a vanishing telescope tail, exactly as the theorem predicts.

### Run the notebook

```bash
jupyter notebook HybridLLM_RL_Notebook.ipynb
```

Sections are ordered to follow the paper: GA-MDP → shaping → Prop. 4.1 check → Algorithm 1 → planner audit → DoorKey validation. The first three sections run fully offline.

### Set up a local LLM (needed for the planner audit and DoorKey pipeline)

The paper's experiments use Qwen-2.5:14b served locally through [Ollama](https://ollama.com):

```bash
ollama pull qwen2.5:14b
ollama serve
```

Cloud backends are pluggable through the `Planner` interface. The failure-triggered replan and the LLM-as-Done oracle each add LLM calls per step; see the compute notes below.

### Run the tests

```bash
pytest tests/
```

25 tests run unconditionally (including a Prop. 4.1 sanity check on a 2-state MDP); one is skipped when no live LLM backend is reachable.

---

## Results reported in the paper

**Planner-in-isolation audit** (Qwen-2.5:14b, 20 MiniGrid tasks, prompt P1): 100% parse rate, 54.8% mean ground-truth coverage, mean plan length 8.0 vs. ≈ 3.6 for ground truth, ≈ 5.3 extraneous subgoals per plan. Plans tend toward verbose low-level phrasing ("Turn left to face the wall") rather than compact subgoals ("go to the key").

**Pipeline validation** (`MiniGrid-DoorKey-6x6-v0`, 3 seeds, 30k steps/seed, last 25% of episodes):

| Config | Success rate | Final return | Steps/episode | LLM calls/seed |
|---|---|---|---|---|
| PPO baseline | 28.1% [4.8, 65.2] | 0.106 [0.008, 0.262] | 340.4 [317.2, 352.4] | 0 |
| Hybrid (Qwen-2.5:14b) | 9.3% [0.0, 14.3] | 0.058 [0.000, 0.095] | 348.0 [340.0, 358.5] | ≈ 30 |

Neither configuration is expected to converge at this budget (DoorKey-6x6 typically needs ~5×10⁵ PPO steps); confidence intervals overlap. The instructive finding is that **subgoal advances were zero across all hybrid seeds**: Qwen's verbose plans never matched the environment-event Done oracle's keyword patterns, so the scheduler stayed on the first subgoal all episode. This is the vocabulary-mismatch failure mode anticipated in §6.3, and the Done-oracle ablation axis (learned classifier or LLM-as-Done) is the intended remedy.

What the pilot does establish: the pipeline runs end-to-end on CPU for 90k steps without error, subgoal embedding integrates cleanly with PPO, the plan cache keeps LLM usage to ~30 calls per seed, and the IQM/bootstrap reporting is stable.

---

## Compute requirements

- **Prop. 4.1 verification, unit tests, GA-MDP + shaping code:** CPU, seconds, no LLM.
- **Planner audit:** CPU + local Qwen-2.5:14b; ≈ 22 min for 20 plans in the paper.
- **DoorKey pipeline:** CPU; ≈ 1.3 h of Ollama wall clock across all 6 runs with the plan cache. Shaping (`c > 0`), failure-triggered replanning (`B > 0`), and LLM-as-Done each require roughly one LLM call per environment step and are practical only with a GPU or cloud LLM.

---

## Scope and limitations

- The implementation is intentionally minimal and pedagogical, written for auditability rather than performance.
- The paper addresses the **fixed-point / soundness question** (can bad LLM outputs corrupt the optimal policy? — no, under the potential construction). It does **not** address the **convergence-rate question** (does the hybrid learn faster?), which needs a larger empirical study (BabyAI, ALFWorld, Crafter, multiple baselines) planned as follow-up work.
- Proposition 4.1 requires Φ to be bounded. The scoring prompt requests a value in [0, 1] and outputs are clamped; if malformed outputs exceed ~1%, shaping is no longer strictly potential-based. Monitor this in practice.
- LLM outputs are nondeterministic; audit and pipeline numbers will vary across runs and backends. The Prop. 4.1 verification is deterministic.

---

## Citation

```bibtex
@article{hounwanou2026policyinvariant,
  title   = {Policy-Invariant Reward Shaping from {LLM} Feedback: A Framework for Hybrid {RL} Agents},
  author  = {Hounwanou, Christophe D. and Eze, John Emeka and Gaba, Ya{\'e} U.},
  journal = {arXiv preprint arXiv:2608.18008},
  year    = {2026},
  url     = {https://arxiv.org/abs/2608.18008}
}
```

---

## Contact

Corresponding author: Christophe D. Hounwanou — christophe.hounwanou@aims.ac.rw
Bug reports and questions: please open a [GitHub issue](https://github.com/chrishounwanou/hybrid-llm-rl-framework/issues).
