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



[**Beyond Homogeneous Attention: Memory-Efficient LLMs via Fourier-Approximated KV Cache**](https://arxiv.org/pdf/2506.11886)
- Problem
    - When LLMs process longer inputs (longer contexts), they store more Key-Value (KV) pairs in memory. 
    - This KV cache becomes very large and slows things down.
- Solution: FourierAttention
    - Key idea: Attention heads don't all do the same thing:
        - Lower-dimensional heads focus on local context (e.g., recent tokens).
        - Higher-dimensional heads capture long-range dependencies.
- FourierAttention projects the lower-dimensional (local context) parts into Fourier space:
    - Instead of saving every detail, they convert the short-term memory parts into a simpler summary using Fourier transforms
- These coefficients are fixed-length, so memory use stays constant, not growing with context length.
- This method works without training

- What's Fourier?
    - Fourier transform = turning something complicated (like a wave or signal) into a few repeating patterns (frequencies).


[**Qwen3 Technical Report**](https://arxiv.org/pdf/2505.09388)
- Key innovation
    - Integration of thinking mode (reasoning) and non-thinking mode
    - Eliminates the need to switch between nodes 
    - Thinking budget mechanism


- Pretraining 
- Synthesize data using domain specific mode
- 3 stages
    - General knowledge
    - Further trained on knowledge-intensive data such as math, coding 
    - Trained on long context
- Pretraining dataset
    - "To further expand the pre-training data corpus, we first employ a vision language model to perform text recognition on a large volume of PDF-like documents (scanned paper)."
    - The recognized text is then refined which helps improve its quality. 


- Postraining
- First 2 stages (developing strong reasoning abilities)
- Step 1: long chain-of-thought (CoT) cold-start fine tuning 
    - Chain-of-thought
        - Technique where model generates intermediate reasoning steps before arriving at a final answer 
    - Cold-Start
        - Model is random weights or basic LLM
    - CoT-formatted datasets — where inputs include a question, and outputs are long, logical step-by-step solutions. - use finetuning or reinforcement learning 
    - Warm-Start
        - Two step process
        - 1. Pretrain model on basic instruction following or Q&A
        - 2. Finetune on more complex CoT datasets 
- Step 2: reinforcement learning focusing on mathematics and coding tasks.
    - 1. Choose task format
    - 2. Define environment
        - State: current input
        - Action: next token or sequence to generate
        - Reward: 0-1 for being correct
    - 3. Dataset preparation (preprocess into input-output pairs)
        - GSM8K (grade school math 8 k), HumanEval
        - Human written grade-school math problems 
    - 4. Warm start with fine tuning 
        - Using the prepared dataset 
        - Teacher forcing: model given ground truth (actual previous tokens) rather than auto-regressive generation 
    - 5. Reward modeling 
        - Collect samples, label/rank response, train reward model
    - 6. Reinforcement Learning 
        - Use SFT model as the policy network for PPO
        - Optimize PPO
    - 7. Evaluation
        - For math: numerical answer matching
        - Last Resort: Human Evaluation 

- Final 2 stages
 - Step 1: 
    - we combine data with and without reasoning paths into a unified dataset for further fine-tuning, enabling the model to handle both types of input effectively, 
    - apply general domain reinforcement learning to improve performance across a wide range of downstream tasks
- Step : 2For smaller models, use strong-to-weak distilation 
    - Off-policy distillation: The student learns from data collected by the teacher, without interacting with the environment.
        - Lets student learn from the teacher’s experiences 
        - More efficient
    - On-policy distillation: The student interacts with the environment and receives guidance or targets from the teacher during training.
        - Learn while acting in the environment with the teacher’s guidance
        - More stable 

 - *Increasing thinking budget leads to consistent improvement in model’s performance



[**MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention**](https://arxiv.org/pdf/2506.13585)
- Key Ideas
- Hybrid Mixture-of-Experts (MoE) 
    - It combines different types of attention mechanisms or expert networks within the model. 
    - choose diverse experts for different styles of processing — for example, some for reasoning, some for memory, some for simple tasks.
- Lightning Attention 
    - Reduce memory usage and increase speed
    - How: 
        - Kernel Approximation: replaces expensive attention formula with a simpler version (similar to kernel trick in SVMs)
        - Sparse Attention: Rather than attending to all tokens, only look at the few important ones 
- CISPO Algorithm (clipping importance sampling weights)
    - controls the learning step size by limiting how much new data influences updates 
    - contrary to directly limiting changes at each token level (traditional)
    - smoother and faster training


[**Xolver: Multi-Agent Reasoning with Holistic Experience Learning Just Like an Olympiad Team**](https://arxiv.org/pdf/2506.14234)
- Problem?
    - Current LLMs typically solve each problem independently, without learning from previous problems or experiences. 
    - They don’t accumulate knowledge over time or collaborate like expert human teams do.

- Xolver is a multi-agent reasoning system designed to give an LLM a kind of persistent memory and collaborative reasoning ability without additional training (training-free). 
    - Remember and retrieve past experiences (both self-generated and external).
    - Use tools and external resources.
    - Collaborate with other "agents"
    - Evaluate its own reasoning and improve it iteratively.
- In other words, instead of solving problems from scratch every time, Xolver leverages accumulated experience and teamwork between agents to solve problems more effectively.

- Xolver demonstrates a shift from isolated problem-solving toward experience-aware, collaborative AI agents that reason more like expert humans do.

[**Revisiting Reinforcement Learning for LLM Reasoning from A Cross-Domain Perspective**](https://arxiv.org/pdf/2506.14965)
- Problem?
    - Designing good reward signals for RL in diverse reasoning areas.
-  re-evaluates prior claims about RL’s effect on LLM reasoning:
- RL isn’t just about extracting pretraining knowledge.
    - For well-represented domains in pretraining (e.g., Math, Code, Science), cross-domain RL works well.
    - For less common domains (e.g., Logic, Simulation, Tabular), RL must be done within-domain to see gains.
- This suggests that RL can actually teach new skills, not just reinforce memorized ones.


[**EMONET-VOICE: A Fine-Grained, Expert-Verified Benchmark for Speech Emotion Detection**](https://arxiv.org/pdf/2506.09827)
- Problem?
- Modern AI systems struggle to recognize complex, context-dependent human emotions in spoken languag
    - Low granularity: Most use basic emotions (happy, sad, angry...), missing nuanced ones like bittersweet, envy, embarrassment.
    - Unrealistic data: Datasets use studio-quality, acted speech with little linguistic or cultural diversity.
    - Small scale: Privacy, licensing, and cost issues limit size—making them unsuitable for modern deep learning.
- Solution
- Instead of treating emotions as discrete labels, the authors adopt a constructionist view:
    - Emotions are context-driven and best modeled with multi-label, intensity-aware approaches.
    - They use frameworks like valence-arousal (e.g., how positive/negative and how intense an emotion is) to better capture emotional nuance.
- Steps
    - 1. Create a massive, realistic training dataset using high-quality synthetic voices in multiple languages, with 40 different emotions — not just basic ones.
    - 2. Build a benchmark with expert-labeled emotion clips to test how well models can understand not just what emotions are present, but how intense they are.
    - 3. Train a new AI model, called EmpathicInsight-Voice, that can recognize these fine-grained emotions better than current systems.


