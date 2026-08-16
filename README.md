# Policy-Invariant Reward Shaping from LLM Feedback  
Reference Implementation (Notebook)

This repository contains the official reference implementation accompanying the paper:

**Policy-Invariant Reward Shaping from LLM Feedback: A Framework for Hybrid RL Agents**

The notebook provides a minimal, fully auditable version of the hybrid LLM-planner + RL-controller pipeline, along with numerical verification of the theoretical guarantee presented in the paper.

---

##  Overview

This implementation focuses on the *fixed point* of hybrid LLM+RL training:  
how the shaping term is constructed, why it preserves optimal policies, and how the pipeline behaves under controlled conditions.

The notebook includes:

- A clean implementation of the **Goal-Augmented MDP (GA-MDP)** formulation.  
- The **potential-based shaping mechanism** using LLM-derived per-state progress scores.  
- A numerical verification of **policy invariance** on a small MDP under four potential configurations, including an adversarial potential scaled to \(20\times\) the base reward.  
- The full **hybrid inference pipeline** (LLM planner + RL controller) as specified in the paper.  
- A lightweight **planner-audit module** used to evaluate LLM plan quality independently of RL training.

---

##  Contents

- `HybridLLM_RL_Notebook.ipynb` — main reference notebook  
- GA-MDP construction  
- Potential-based shaping implementation  
- Numerical experiments  
- Planner audit on MiniGrid tasks  
- End-to-end pipeline validation on DoorKey-6x6  

---

##  Purpose

This repository is designed to make the theoretical core of the framework:

- transparent,  
- reproducible,  
- and easy to inspect.

It serves as a foundation for future empirical extensions and larger-scale experiments planned in follow-up work.

---

##  Notes

- The implementation is intentionally minimal and pedagogical.  
- It is meant for researchers interested in the theoretical aspects of LLM-augmented RL.  
- All components correspond directly to sections of the paper for easy cross-reference.

---

##  Citation

If you use this code in your research, please cite the paper:

**Policy-Invariant Reward Shaping from LLM Feedback: A Framework for Hybrid RL Agents**

---

