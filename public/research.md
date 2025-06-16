[**Autoregressive Semantic Visual Reconstruction Helps VLMs Understand Better**](https://arxiv.org/pdf/2506.09040)
- The Problem with Most Vision-Language Models (LVLMs)
- Large models like LLaVA or GPT-4V are trained to understand both text and images. But most of them only learn by predicting text — even when images are also part of the input.
- This leads to three main problems:
    - They can’t learn from images alone – If there’s no caption, the image is ignored.
    - Captions may miss important visual details – So the model doesn’t learn everything that’s actually in the image.
    - Some things in images are just hard to describe in words – So the model might never learn them well.

- *The Solution: ASVR (Autoregressive Semantic Visual Reconstruction)*
- ASVR trains the model to also reconstruct the meaning of the image, step-by-step, in an autoregressive way.
    - Trying to recreate the raw image pixels doesn’t help and can hurt performance.
    - recreating the "semantic" content of the image — like object labels, scene descriptions, or high-level tokens — works really well.


[**CONFIDENCE IS ALL YOU NEED: FEW-SHOT RL FINE-TUNING OF LANGUAGE MODELS**](https://arxiv.org/pdf/2506.06395)
- Reinforcement Learning via Self-Confidence (RLSC)
- Method to improve large language models (LLMs) without using labeled data, human feedback, or complex reward models.
    - RLSC uses the model's own confidence (e.g. softmax probabilities) as the reward signal during reinforcement learning.
    - This makes training label-free, preference-free, and scalable.
- Intuition: RLSC assumes that if a model is internally more confident about an answer, it is more likely to be correct. While not perfect, this heuristic tends to work surprisingly well, especially when the model is already reasonably capable in the domain.
- How It Works:
    - Candidate Generation: possible outputs
    - Confidence Scoring: output passed through model to compute confidence score 
    - Reward Assignment: higher the confidence, higher the reward 
    - Policy Update: model updated using RL 


[**Multiverse: Your Language Models Secretly Decide How to Parallelize and Merge Generation**](https://arxiv.org/pdf/2506.09991)
- The porblem is traditional autoregressive LLMs generate text step-by-step, one token at a time, which limits how much parallel computation they can do. This sequential generation can be slow.

- Multiverse is a new type of model designed to generate text in parallel instead of sequentially, speeding up the process without losing quality.

- Multiverse uses a MapReduce-inspired approach in three stages:
    - Map stage: Breaks down a big generation task into smaller, manageable subtasks dynamically.
    - Process stage: Executes these smaller subtasks in parallel (at the same time).
    - Reduce stage: Combines all the partial results back into a single final output without any loss of information.

[**ReasonMed: A 370K Multi-Agent Generated Dataset for Advancing Medical Reasoning**](https://arxiv.org/pdf/2506.09513)
- Problem: LLMs perform well in math and coding, but underperform in medical QA tasks.
- Solution: Introduce ReasonMed, the largest medical reasoning dataset:
    - 370,000 high-quality examples distilled from 1.7 million reasoning paths.
    - Uses a multi-agent pipeline:
    - A Verifier flags weak reasoning.
    - An Error Refiner improves those steps.
- Training Strategy:
    - Combines Chain-of-Thought (CoT) reasoning with concise answer summaries

- ReasonMed dataset was created at scale using a multi-agent system (MAS) to improve LLM performance on medical question answering (QA).
    - Why Scale Matters: Larger and better-quality datasets lead to stronger models, especially for complex tasks like medical QA.
    - Multi-Agent System (MAS): Used 3 strong LLMs to generate reasoning paths for these questions:
    - How They Generate Diversity: By tweaking generation settings (like temperature and top-p) for each agent, they produce a wide variety of multi-step reasoning paths.
    - Outcome: About 1.75 million reasoning chains were generated.

[**SWE-Factory: Your Automated Factory for Issue Resolution Training Data and Evaluation Benchmarks**](https://arxiv.org/pdf/2506.10954)
- automated pipeline for building large-scale, high-quality datasets to train and evaluate Large Language Models (LLMs) on GitHub issue resolution tasks
- SWE-Builder (Multi-Agent System) -> set up coding problem
    - A multi-agent system with 4 agents working in collaborative loops.
    - Automatically constructs the environment for evaluating a given GitHub issue
    - Uses a shared memory pool to pass info efficiently between agents.
- Exit-code Based Grading -> code runs correctly means problem solved 
    - Instead of writing custom checkers, it uses standardized Unix exit codes (e.g., 0 for success) to automatically determine whether the LLM’s solution works.
    - This grading method is 100% accurate compared to manual checks.
- Automated Fail2Pass Validation -> double check
    - After grading, it checks whether the model turned a failing test into a passing one.





