**Learning of the day (1/1)**  
- research 
    - Current LLMs are not great and following rules
    - Possible improvements are
- test-time steering 
    - supervised fine-tuning.
    - Learning
- Regularization: prevents overfitting by adding a penalty term to the loss function 
- L1: Lasso -> when many features are useless 
- L2: ridge -> all features are useful but need to control the impact 


**Learning of the day (1/2)**
- Neural Network from Scratch
    - Derivative step of the sum function is 1
    - Weight * input -> + bias -> sum function -> Activation
    - Derivative of categorical cross entropy
    - - y_true/y_pred
    - Derivative of softmax
    - np.diagflat(sftmax_output) - np.dot(sftmax_output, sftmax_output.T)
- DPO (direct preference optimization)
    - Provides higher probabilities to positive responses using cross entropy while keeping aspects of the baseline model (prevent reward hacking)
    - Bypasses traditional 2 step RLHF (reinforcement learning with human Feedback) using a reward model
- Learning
- Supervised learning: train model with labeled data: regression
- Un-supervised learning: clustering, PCA 

**Learning of the day (1/3)**
- Learning
    - What is gradient descent?
    - Optimization algorithm to minimize loss function by iteratively updating the model parameters in the opposite direction of the gradient of the loss function
- ADAM: adaptive learning rates
    - Too low: converge too slowly
    - Too high: overshoot the minimum
- Stochastic vs Batch
    - Batch: entire dataset
    - Stochastic: randomly selected subset of the dataset

**Learning of the day (1/4)**
- Learning 
    - How do you combat the curse of dimensionality? 
    - Exponential increase in computational effort as the number of dimension increases
    - Harder to find patterns
    - overfitting 
    - Computational complexity 
- Feature Selection, PCA, t-SNE, UMAP, LLE (locally linear embedding)
    - LLE: project high-dimensional data into a lower-dimensional space while preserving the local relationships between data points
    - t-SNE: maps high-dimensional data onto a lower-dimensional space
    - PCA: identifies direction in which data varies the most
- Concept
    - PPO (Preferred Policy Optimization)
    - Used for reward model
    - Policy function: prob of choosing action at a certain state
    - Reward function: expected reward of action at a certain state (Q)
    - Back propagation using negative loss function -> gradient ascent (global maximum)

**Learning of the day (1/5)**
- Learning
- Why use RELU over Sigmoid?
    - Computational efficiency
    - Reduce likelihood of vanishing gradient
    - Sparsity (neuron inputs are negative and less activated) -> network is lighter
- Concept
    - Vanishing Gradient
    - gradients used to update the network weights become extremely small during backpropagation
    - prevents the network from learning effectively in deeper layers
    - Solutions
        - Use ReLU
        - weight initialization strategies like Xavier (1/input+output - Sigmoid/tanh) initialization or He(kaiming) initialization (2/n - ReLU)
        - Scaled variance - balances signal across layers so activations don’t get too large or too small
    - Exploding Gradient
    - gradients calculated during backpropagation become excessively large
    - drastic weight updates, potentially causing the model to jump across the loss landscape and fail to converge to an optimal solution
    - Happens when there are many layers and using activation functions such as Sigmoid: Jump in weights, NaN
    - Solution
        - Use optimizer such as ADAM to control learning rate
        - Use RNN (LSTM) 
        - Gradient clipping: limit size of gradients from becoming too large 
- Neural Network from Scratch
    - Learning rate decay: decreasing the learning rate over time/steps 
    - Good way is to program a decay rate
    - Momentum: creates a rolling average of gradients and uses this average with the unique gradient each step
    - Uses previous update’s direction to influence the next update’s direction 
    - With momentum, more likely to pass through local minimums 

**Learning of the day (1/6)**
- Learning 
    - Describe how convolution works?
    - a convolution operation is applied to the input image using a small matrix called a kernel or filter.
    - The kernel slides over the image in small steps, called strides, and performs element-wise multiplications with the corresponding elements of the image and then sums up the results -> this result is called a feature map 
    - Benefits
        - Robust to noise
        - Reduces memory and computational needs (reduce dimensionality) 
        - Feature learning: automatically learn features from raw data
        - Transfer learning - can be pretrained 
    - Disadvantages
        - Require a large amount of labeled data
        - Prone to overfitting 
- Concept
    - DPO: Instead of optimizing reward function, optimize the optimal policy

**Learning of the day (1/7)**
- Learning
    - What is data normalization and why do we need it
    - Data normalization is a process of rescaling the input data so that it fits within a specific range, usually between 0 and 1. It is calculated by subtracting the mean and dividing by the standard deviance. Data normalization ensures all features are weighted equally and assures better convergence during back propagation
- Concept
    - There is a token to specify the start and end of sentence 
    - SOS: Start of Sentence
    - EOS: End of Sentence
- Neural Networks
    - decay  -> reduce learning rate
    - Momentum -> creates a moving average of gradients and uses this with the unique gradient at each step
    - More likely to pass through local minimums 
    - Adagrad (Adaptive Gradient) & RMSprop -> normalize parameter updates by keeping history of previous updates
        - Bigger the update is in a certain direction, the smaller updates are made further in training
        - Learning rates for parameters with smaller gradients decrease slowly and vice versa 
        - Less likely to overshoot global minimum
    - RMSprop: divide by the square root of gradients squared 
    - ADAM
        - Combination of Momentum and RMSprop
        - Parameters beta1, beta2,
        - Learning rate: base learning rate
        - Decay: rate of decay
        - Epsilon: prevents dividing by zero for RMSprop
        - Beta_1: how much weight to give to momentum
        - Beta_2: how much weight to give to cache 

**Learning of the day (1/8)**
- Learning
    - Why do we max pool in classification CNNs?
    - Max pooling reduces computations since it feature maps become smaller
    - enhances invariance to object position and orientation within an image
    - Potentially prevents overfitting by focusing on the most prominent features in a local region
- Neural Network
- Regularization
    - Calculate a penalty to add to the loss value
    - Penalize model for large weights and biases
    - Generally better to have many neurons contribute rather than a select few
        - L1: absolute value
        - L2: squared value 
    - Hyperparameter Lambda
        - Delicate how much impact we want the regularization penalty to carry
        - Higher the lambda, higher the penalty 
- Concept
    - Negative Sampling 
    - Solves computational program 
    - Minimize the probability that randomly sampled word-context  (2~20) pairs are observed rather than the entire dictionary

**Learning of the day (1/9)**
- Learning
    - What is the significance of residual networks? 
    - Introduce residual learning to address challenges such as vanishing gradient problem
    - Uses skip connections that bypass one or more layers which helps network focus learning the differences, making optimization easier 
    - Allowed direct feature access from previous layers: features from previous layers are retained and important information is not lost. 
- Concept
    - Encoder -> context vector -> decoder
    - Encoder combines words in the sequence and extract the “meaning”
    - This “meaning” is stored in the context vector 
    - The context vector is inputted to the decoder and similar words are outputted based on similar vectors (which means similar words)
- Neural Networks        
    - Dropout
    - Disables some neurons, while other are unchanged
    - Disables during every forward pass, learn to make predictions with only random neurons remaining
    - Disabled mean setting the output values to zeros
    - Prevent becoming too dependent on any neuron
    - Prevent co-adoption: neurons depend on output of other neurons and doesn’t learn function on their own
    - Aims to combat overfitting 

**Learning of the day (1/10)**
- Learning
    - What is Batch Normalization and how does it work?
    - Batch Normalization makes training faster and more stable by normalizing the inputs to each layer. It’s like resetting each layer ensure the inputs are neither too big nor too small
    - Subtract mean and divided by STD
    - Introduces the shift and scale learnable parameters
    - Faster convergence and reduces overfitting
- Concept
    - Recurrent Neural Networks
    - RNNs have an internal memory (hidden state) that captures information about prior inputs in the sequence. This means that it keeps earlier data to help understand the next output
    - The same weights are used across all time steps and reduces the number of parameters
    - Problems:
        - vanishing/exploding gradients
        - Short-term memory: forget what happened long time ago
    - Fixes
        - LSTM: Long short term memory
        - GRU: Gated Recurrent Unit 

**Learning of the day (1/11)**
- Learning
    - Why would you use many small convolutional kernels such as 3x3 rather than a few large ones?
    - smaller kernels allow less parameters and computations for the same receptive field and spatial content
    - Will be using more filters and more activation functions and have a more distinct mapping functions 
- Concept
    - Attention:
    - Communication mechanism between tokens 
        - No notion of space, which is why we need to position encoding 
    -    Each example across batches is completely independent.(They do not talk to each other
    - Encoder block: allow all tokens to communicate
    -    ex) used for sentiment analysis
    - Decoder block: mask future tokens 
        - Autoregressive settings: predictions made sequentially based on previous inputs or outputs 
    - In Self attention, all K, Q, V comes from the same source 
    - In Cross Attention, K, Q, V can come from different nodes
        - ex) use Keys to pull information from different nodes (Q, V)
    - Scaling (divide by sqrt(head_size) is used to prevent softmax values from being too extreme 
    - Logit
        - raw, unnormalized output of a neural network layer 
        - Before activation function
        - Used as an input for softmax
    - Batches
        - Batch size = number of samples processed at once
        - Parallel computation - efficient
    - Self Attention
        - Keys:what do i contain?
        - Queries: what do I look for?
        - Affinity: dot product between key and query
    - In Self attention, all K, Q, V comes from the same source 
    - In Cross Attention, K, Q, V can come from different nodes
    - ex) use Keys to pull information from different nodes (Q, V)

**Learning of the day (1/12)**
- Learning
    - Why do we need a validation set and test set? What is the difference between them? 
    - We divide the data into 3 sets, Training, Validation, and Test
    - The training set is used to learn and fit the model parameters
    - The validation set is used to tune the hyperparameters of the model.
    - The test set is used to measure how well the data performs on unseen data. 
- Concept
    - Residual Connections (skip connections)
    - Intuition: "Remember what I said earlier, and add this new piece”
    - Secure original information is retained and not lost
        - How: adding input directly to the output before activation 
        - Used in Transformer Blocks 
    - Benefits
        - Reduce vanishing gradient 
        - Preserves information
        - Faster convergence 

**Learning of the day (1/13)**
- Learning
    - What is stratified cross-validation and when should we use it?
    - Cross validation is the process of dividing the data in train sets and validation sets.
    - Stratified CV splits the data categories into the same ratios.
      - LOOCV is used by using one set as the validation set and using the rest as the train set. 
    - CV is used for hyperparameter tuning to prevent overfitting 

**Learning of the day (1/14)**
- Learning
    - Why do ensembles typically have higher scores than individual models?
    - Ensemble uses multiple models for a single prediction. The key idea is to have each model make different errors so the other model can compensate for it by making the right prediction. 
    - It also helps with overfitting by reducing the impact of a single model’s biases. 


**Learning of the day (1/15)**
- Learning
    - What is an imbalanced dataset? Can you list some ways to deal with it?
    - Different proportions of data in each category 
    - oversampling/undersampling 
    - Data augmentation 
    - Using appropriate metrics such as precision, recall, f1 score 
- Concept
    - Pytorch Steps
        - Set to train()
        - Logits = model(X_train)
        - Preds = softmax
        - Calculate loss
        - Calculate accuracy
        - Set zero_grad 
        - Backward step
        - Optimizer step
        - Set to eval()  
        - Inference mode

**Learning of the day (1/16)**
- Learning
    - What is data augmentation?
    - Technique for synthesizing new data by modifying existing data such that the target is unchanged or changed in a known way 
    - Computer vision:
        - Resize, reshape, rotate, add noise, modify colors, deform
- Concept
    - Auto-encoder
    - Encoder -> compresses data into lower dimension
        - This can be done by capturing similar data structures
    - Latent space ->  bottleneck that enforces dimensionality reduction
    - Decoder -> reconstructs original data from latent representation
    - Usage
        - Dimensionality reduction 
        - Data denoising
        - Anomaly Detection
        - Data generation 

**Learning of the day (1/17)**
- Learning 
    - Precision, Recall, F1 Score?
        - Precision: percentage of true positives among all labeled as positives 
        - Recall: percentage of true positives identified among all positives 
        - F1 Score: weighted average of precision and recall 
- Concepts
    - PyTorch
    - DataLoader: stored data into batches based on transforms
    - Predicted values from models are logit

**Learning of the day (1/18)**
- Learning
    - Define Learning Rate?
    - Hyperparameter that adjusts how much we are updating the parameters of the model with respect to the loss gradient
    - If the learning rate is too high, it may not converge to the global minimum. If learning rate is too low, it may take too long to converge
    - We can apply adjustments such as decay (L2 regularization) and momentum to the learning rate 

**Learning of the day (1/19)**
- Learning 
    - What is momentum?
    - Momentum adjusts the learning rate appropriately by keeping track of the past gradients by calculating the moving average. Momentum allows the optimizer to avoid getting stuck at local minimums
    - Web/PDF scraping 
    - PDF: Import fitz (PyMuPDF)

**Learning of the day (1/20)**
- Learning
    - What is the difference between Batch Gradient Descent and Stochastic Gradient Descent?
    - Batch Gradient Descent computes the gradients for the entire dataset whereas Stochastic Gradient Descent computes gradient using a single sample 
    - Batch Gradient Descent is better for smaller datasets since it has a slow computational time and memory usage in exchange for smoother convergence
    - There is also a mini-batch gradient descent witches uses one-batch 


**Learning of the day (1/21)**
- Learning
    - What is the vanishing gradient?
    - As we add more hidden layers, backpropagation becomes less useful in passing information to deeper layers and the gradients become extremely small compared to the weights which prevents the network from learning. We can reduce this impact by using residual networks or ReLU activation function 
- Concept
    - LoRA (low rank adaptation)
    - Reduces the number of parameters that needs to be trained during fine tuning
    - Original weights are frozen, only low rank updates are learned 
        - Delta W = A * B
    - Small low rank decomposition
    - Linear combinations of set of smaller vectors (dimensionality reduction)
        - Hyperparameters: alpha, rank
            - Alpha: scale parameter, how much impact given to pretrained weights
            - Rank: controls low rank size 
    - Usually, alpha = r 
    - Efficient, scalable, reusable 

**Learning of the day (1/22)**
- Learning
    - What are dropouts?
    - Dropout is a regularization technique which disables a percentage of neurons at random to prevent overfitting. This means that during training, certain neurons are not used which allows all neurons to learn features and reduce co-adaptation
- Concept
- BERT (Bidirectional encoder representation of transformers)
    - Designed to generate contextualized word embeddings
    - Not next word prediction
        - Masked language model: predict word inside sentence
        - Next sentence prediction
    - Fine tuned by adding task specific layers
    - Bidirectional: captures context on both sides at once (no masking)        
        - Wordpiece tokenizer 
        - CLS token (Classification)
        - Representative summary of the entire input sentence (contextualized embedding)
        - CLS token can attend to all other tokens in the sentence, making it ideal for gather context 

**Learning of the day (1/23)**
- Learning
    - What are the components of GAN (Generative Adversarial Network)?
    - Generator: create synthetic data that cannot be distinguished from real data
    - Discriminator: determine whether or not data is real
    - Goal: create synthetic data that cannot be distinguished as fake
- Concept
    - Stable Diffusion
    - text- to-image generative model
    - Diffusion process: add random noise to images 
    - Model learns to reverse this process by removing noise
        - Uses U-Net 
        - Encoder -> Bottleneck -> Decoder 
    - Skip connections between encoder, decoder helps preservation of image details 
    - Predicts noise present in the image and subtract the noise from the input
    - Text-to-image Translation
        - Uses a latent diffusion model (LDM)
        - Lower dimensional latent space (computationally efficient)
        - Transformer-based text encoder to understand textual input

**Learning of the day (1/24)**
- Learning
    - What is the transformer architecture?
    - Transformer architecture has an encoder and decoder, each with 3 main layers
    - The first layer is the embedding layer which converts tokens into vectors. Positional embeddings are added to capture the order of the tokens
    - The second layer is multi-head attention. Attention scores are calculated by comparing queries to keys and it determines how much focus each token should receive from the current token. Multi-head attention allows the model to capture relationships simultaneously. 
    - The last layer is the feedforward network which applies the non-linear transformations, 
    - The encoder and decoder communicate through the cross attention mechanism where the encoder can be seen as the ‘comprehension’ module which understands and processes the input and the decoder can be seen as the ‘generation’ module which produces the output 
    - The main advantages of the transformer architecture compared to RNNs are efficient training since it processes tokens in parallel, the ability to capture long range dependencies, and scalability. 
- Concept
    - Broadcasting Rules
        - Each tensor has at least one dimension
        - When iterating over the dimension sizes, starting at the trailing dimension, the dimension sizes must either be equal, one of them is 1, or one of them does not exist
        - If the tensors are broadcastable
        - Prepend 1 to the one with the smaller dimension and the resulting size will have the larger dimension
        - Quantization
        - Reducing the precision of numerical values of the model parameters to decrease the model size and computational requirements. This results in faster inference and lower power consumption 
        - Uses 8-bit integers rather than 32-bit float
    
**Learning of the day (1/25)**
- Learning
    - How is NSP used in LLMs?
    - Next sentence prediction is used in the BERT model along with masked language modeling. NSP’s objective is to determine if the sentence logically follows the other previous sentence presented in the model. This allows the model to understand longer-term dependencies across sentences
    - Explain the concept of Context Window?
    - Context window is the range or len of tokens the model can consider at once when understanding or generating text. Longer context windows require more computational power but it can better understand lengthy or complex texts. Gpt2 had a context window of 1024 tokens. Gpt 4 has 33,000. 
- Concept
    - Byte Pair Encoding (GPT) - Fast and Efficient
        - Works by iteratively merging frequent pairs of chars or subwords
            - Initialization: characters + _
            - Identify patio of symbols next to each other
            - Merge most frequent pair 
            - repeat 
        - Handles out-of-vocabulary (OOV) words
        - Balances Granularity
        - Efficient
        - Language Flexibility
        - There is a vocab size limit (hyperparameter)
    - WordPiece (BERT) - more accurate and context sensitive 
        - Works by merging pairs with based on the maximum likelihood ratio
        - Merge most likely subword pairs in the training data
        - Increase the probability of seeing the token in the corpus  
    - SentencePiece
        - Works on raw text without preprocessing (white space and punctuation preserved)
        - Break words into subwords, not characters
        - Good for NLP, multi language models 

**Learning of the day (1/26)**
- Learning
    - How do you evaluate the performance of LLMs?
    - Intrinsic: how well the model captures what it’s supposed to capture  
        - Perplexity: how much the model is surprised by new data
        - Lower perplexity, better training 
    - Extrinsic how useful the model is for a specific task
        - Text classification
        - Question Answering (QA)
        - Summarization
        - Code Generation 
    - What are some pre-training objectives of LLMs and how do they work? 
        - Masked language modeling
        - Predict correct word within a sentence 
        - Autoregressive modeling (Casual Language Modeling)
        - Predicting the next word given a sequence
        - Seq2Seq, NSP, 
- Concept
    - RNN
    - Allows processing of sequential data by maintaining a hidden state that captures information regarding previous states. Hidden layer has two outputs. One output and one to feed back into the hidden layer. Input to the layer is the concatenation of input from sequence and output from the previous hidden layer. 
    - It is great at sequential data but not good at long-term dependencies. It is computationally expensive for long sequences and prone to exploding/vanishing gradient problem 
    - FeedForward vs Attention Blocks 
        - FeedForward (individual inputs): used for feature transformations and learning representations of individual inputs
        - MLP (multi layer perceptron) is a subset of FFNs. MLP has at least one hidden layer 
        - Attention Blocks (multiple inputs): used to weigh the importance of different tokens and capture long range dependencies. Figure which are important and which are related 


**Learning of the day (1/27)**
- Learning
    - How do generative language models work?
    - Generative language models work by predicting the next work given an input sequence. Based on a list of possible output words, the model chooses the word with the highest probability. The models are auto regressive which means that the output of the model is feedback in as an input to predict the next word until it reaches an EOS token 
    - What is fine-tuning and why is it important?
    - Fine-tuning is taking a pre-trained model created for general and broad range tasks and re-training it for specific tasks such as question answering. New layers and datasets can be added to train the model on a new objective with a loss function reflecting the objective. 
- Concept 
    - LSTM 
    - LSTMs try to overcome the issues of short term memory and vanishing/exploding gradients of vanilla RNNs. It is done by keeping track of a long term memory through the cell state. 
    - There are 3 gates that keep track of the long term memory. The forget gate determines which information should be forgotten from the cell state. The input gate determines which information should be added to the cell state. The output gate (sigmoid) controls what part of the cell state should be used to compute the hidden state and passed onto the next step. 
        - Forget: controls past memory 
        - Input: controls new (short term) memory
        - Output: controls the combination of past and new memory
    - You can think of this process as a teacher telling the student which information is important to retain and what should be forgotten 
        - Cell state update
        - Cell state = Forget gate * prev cell state + input gate * candidate cell state 
    - Hidden state = output gate * cell state 
    - Pros:
        - Handles long-term dependencies
    - Cons:
        - Slower and difficult to train
        - Still not good at very long sequences

**Learning of the day (1/28)**
- Learning
    - What is a token in LLM context
    - Token is the single unit of text that is used as an input to LLMs. Tokens are converted to numerical format and it is used for training the model and for inference. The reason for using tokens is because plain word vocab size is too big and it is not good for generalization of Out Of Vocab words. A common tokenizer is Byte pair encoding which is used for the GPT models 
    - What are some common hardships regarding LLMs?
    - It requires a lot of computational resources, understanding and interpreting the decisions made by the LLMs is difficult due to their complex nature 
- Concept
- GRU (Gated Recurrent Units)
    - A special type of LSTM that is more efficient. It has 2 gates rather than 3
    - Update gate 
    - Forget gate
    - Word2Vec
        - Word embedding technique to to represent ‘words’ as numerical vectors 
    - CBOW: predicts target word based on surrounding text 
    - Skip Grams: predicting surrounding text
    - Layer norm: by row: across features
    - Batch norm: by col: by each feature

**Learning of the day (1/29)**
- Learning
    - What is the advantage of using Transformers over LSTMs?
    - Transformers can process the entire text in parallel which is more efficient than the sequential processing of LSTMs. Transformers have better long-term dependencies through self-attention whereas LSTM can suffer for vanishing gradients. Also, it is more memory efficient since it doesn’t have to store the cell state or hidden state
    - How do LLMs handle OOV words?
    - Through subword tokenization such as BPE and Wordpiece 
- Concept
    - KV Caching 
    - It is used to speed up auto-regressive text generation by eliminating the need to recalculate the attention scores by keeping a memory of cache of keys and values matrix.
    - Since tokens are generated one at a time, each time a new token is generated, it only computes the attention for that token and uses the stored K, Vs for the previous tokens 

**Learning of the day (1/30)**
- Learning
    - Example of alignment problems?
    - Alignment is matching the goals and behavior of the model with human needs and values. 
    - Lack of helpfulness
    - hallucinations 
    - Lack of interpretability 
    - Generating biased or toxic output 
    - What are embedding layers and why are they important?
    - Embedding layers convert words into numerical vectors and capture semantic meaning of the words. Words with similar meaning have similar vector representations. It is used for capturing the semantics of the word and reducing the dimensionality compared to using one-hot encodings. It can also be used to transfer learning by reusing the pre-trained embeddings across different models.  
- Concept 
    - Rotary positional encoding (RoPE)
    - Relative positional embedding + absolute positional embedding 
    - Method for incorporating positional information without explicit embeddings
    - Instead of storing separate position embeddings, the model learns positional relationships through rotation operations.
    - Each (Q, K) pair is rotated by an angle 
    - It is rotated on a 2 dimensions (total pairs dim/2)
    - e^(ix) = cos(x) + i*sin(x)
    - Better long sequence generation and more efficient computation 

**Learning of the day (1/31)**
- Learning
    - How is adaptive softmax useful?
    - Adaptive softmax is a technique used to speed up the softmax computation by clustering more frequent words and rarer words. Rather than computing all words in the vocab separately, rarer words are in deeper clusters which reduces computational time and memory usage 
- Concept
    - RAG (retrieval augmented generation)
    - Generating content through external knowledge retrieval mechanism
    - 1. Query encoding: convert input sentence into embeddings (BERT)
    - 2 models
    - Query encoder 
    - Document encoder 
    - 2. Retriever searches a vector database based on similarity
    - HNSW: Hierarchical Navigable Small World
    - graph-based nearest neighbor search algorithm 
    - Fast, scalable, works in higher dimensions
    - 3. Retrieved document are appended to the original query (augmented input)
    - Kind of like prompt engineering by giving context 
    - 4. Processes the augmented input for generation 


**Learning of the day (2/1)**
- Learning 
    - How does BERT training work?
    - BERT is trained through self-supervised learning
    - two main objectives. Masked Language Modeling and next sentence prediction.
        - MLM is predicting masked word within a sentence
        - NSP is predicting rather or not the sentence is the next sentence of the preceding one.
    - BERT is an encoder only model. 
    - How do you measure the performance of LLMs?
    - Metric such as perplexity which measures how well the model predicts the next token. Lower perplexity, better performance.
        - ROUGE (Recall-Oriented Understudy for Gisting Evaluation) for summarization tasks
        - BLEU (Bilingual Evaluation Understudy) for text generation tasks 
        - Exact Match and F1 Score for question and answering
        - (Massive Multitask Language Understanding) for factual knowledge


**Learning of the day (2/2)**
- Learning
    - What Transfer learning techniques can you use for LLMs?
    - Fine-tuning: using a pre-trained model and training it on a specific task. You can add layers or unfreeze weights 
    - Adapter layers (PEFT): add small trainable layers while keeping most layers frozen: LORA, prefix-tuning, p-tuning
    - Retrieval Augmented Generation (RAG)
    - Feature based transfer learning: using pre-trained model as a feature extractor and training a model on top of the extracted features
    - Multi-Task Learning (MTL): Train on multiple related tasks at once to improve generalization.
    - Knowledge distillation 


**Learning of the day (2/3)**
- Learning
    - What are some techniques for controlling the output of LLMs?
    - Randomness: temperature, Top K sampling, Top P sampling, BEAM, prompt engineering, control tokens


**Learning of the day (2/4)**
- Learning
    - Explain the concept of attention and how it is implemented?
    - Attention allows the model to focus on different tokens of the input sequence by giving weights to each of the words. This is implemented through calculating attention scores which is computed by the dot product of key and query matrices. Words with similar embeddings have a higher dot product value which in turn would result in higher attention scores.  


**Learning of the day (2/5)**
- Learning
    - What are some approaches to reduce the computational cost of LLMs?
    - Compression: Model pruning, distilation, quantization
    - Architecture: MoE, Spares attention, RAG
    - PEFT: LoRA, Flash Attention


**Learning of the day (2/6)**
- Learning
    - What is the difference between encoder vs. decoder models
    - Encoder models understand and extract meaning from the text. Models such as BERT are used, no masking takes place and process all inputs at once so it can use self-attention on both past and future tokens. Used for text classification (sentiment) name entity recognition (NER), semantic similarity test. 
    - Decoder models are used to generate text to predict next tokens. Masking is done for future tokens and inputs are process one token at a time. (auto regressive). It is used for text generation, machine translation, etc.
    - Encoder-Decoder models (seq2seq) are BART and T5 models. Converts one sequence to another and is used for tasks such as summarization and translation.


**Learning of the day (2/7)**
- Learning
    - What’s the difference between wordpiece and BPE
    - They are both subword tokenization methods but BPE iteratively merges based on most frequent pairs. WordPiece merges based on pairs that are most likely to improve the likelihood of the training data. 


**Learning of the day (2/8)**
- Learning
    - How do LLMs handle long-term dependencies in text?
    - Through self-attention mechanism in transformer architecture

**Learning of the day (2/9)**
- Learning
    - What is the difference between local and global attention? 
    - Global attention is where every token in the input sequence can attend to each other whereas local attention is where each token can attend to a window of the input sequence. Global attention is used when full context is critical but it is computationally expensive and not memory efficient

**Learning of the day (2/10)**
- Learning
    - Explain the concept of "few-shot learning" in LLMs and its advantages?
    - Few shot learning means the model can learn and perform on only a few examples. The LLM utilizes the extensive pre-trained knowledge to generalize on smaller instances. Few shot learning reduces the need for additional extensive data and fine-tuning and allows the model to be used for different tasks
- Concept 
    - Temperature
    - If less than 1.0 makes it more confident
    - Divided logits by temperature


**Learning of the day (2/11)**
- Learning 
    - What is the difference between next token prediction (auto regressive) and masked language model?
    - Next token prediction focuses on predicting the word that comes next given an input sequence. Mask language model predicts the what the correct word for the masked token should be within a input sequence 
    - How can you incorporate external knowledge into an LLM?
    - Knowledge graph integration, Retrieval augmented generation, fine-tuning
    - Concept
    - 4 patterns of research papers


**Learning of the day (2/12)**
- Learning
    - What is the difference between monotonic alignment and predictive alignment in transformers?
    - They both refer to different approaches for aligning token representations in sequence modeling. Monotonic alignment ensures that the attention mechanism progresses in a strict order and the model only attends to past tokens. Predictive alignment aligns hidden states with expected future states to predict next step outputs more efficiently. It can be thought of as a teacher guiding a student to focus on the correct features.


**Learning of the day (2/13)**
- Learning
    - What are some challenges associated with deploying LLMs in production?
    - Scalability: can handle large volume of requests
    - Latency: minimizing response time 
    - Resource management: ensuring cost-efficiency


**Learning of the day (2/14)**
- Learning 
    - How do you handle model degradation over time in deployed LLMs?
    - Model degradation happens when the LLM performance declines due to changes in the underlying data distribution. Continuously monitoring the model performance and adding updated data through incremental learning techniques allow the model to learn without losing previous knowledge.


**Learning of the day (2/15)**
- Learning
    - Can you explain how techniques like reinforcement learning from human feedback (RLHF) can be used to improve the quality and safety of LLM outputs, and what are some of the challenges associated with this approach?
    - RLHF is a technique that involves human feedback to align the model output with human preferences. A reward model is created 


**Learning of the day (2/16)**
- Learning
    - List the key components of LSTM?
    - Input gate, output gate, forget gate, cell state - memory


**Learning of the day (2/17)**
- Learning
    - List the variants of RNNs?
    - LSTM, Gated recurrent units, end to end network


**Learning of the day (2/18)**
- Learning
    - What is Autoencoder, name a few applications? 
    - Auto encoder is primarily used for unsupervised learning tasks to compress input data into lower dimensional representation known as bottlenecks or latent space. The decoder reconstructs the data using the compressed representation  
    - Data denoising
    - Dimensionality reduction
    - Image reconstruction
    - Image colorization


**Learning of the day (2/19)**
- Learning
    - What are the components of GAN?
    - Discrimator: aims to discriminate data from humans and generated data. 
    - Generator: aims to create data that is close enough to the human data so it is indistinguishable 

**Learning of the day (2/20)**
- Learning
    - What's the difference between boosting and bagging?
    - Boosting and bagging are both ensemble techniques which combine weak learners to create a more accurate and robust learner. Boosting reduces the bias of the model by using all available data and learning on top of the previous model. Bagging reduces variance by combining models created by bootstrapped samples 

**Learning of the day (2/21)**
- Learning
    - Explain how the ROC curve works?
    - ROC curve is a graphical representation of the trade-off between true positives and false positives. The closer the curve is to the top left, the better the model. AUC (area under the curve) quantifies the performance 


**Learning of the day (2/22)**
- Learning
    - What’s the difference between Type I and Type II error?
    - Type I error is false positives, type II error is false negative. Type 1 error is classifying a condition as true when it is actually false. Type 2 error is not conditioning something as true when it should be conditioned as true 
- Concept
    - Gradient accumulation
    - Is an optimization technique to simulate larger batches in deep learning. Rather than updating model weights after every batch, accumulate gradients over multiple batches before making an update.
    - It handles the memory constraints of GPUs and stabilizes training

**Learning of the day (2/23)**
- Learning
    - What’s the difference between a generative and discriminative model? 
    - A generative model learns the categories of the data whereas a discriminative model learns the distinction between the categories of the model 
    - Concept
    - DDP (Distributed Data Parallel)
    - Parallel computing technique used in pytorch to train across multiple GPUs. It does so by splitting the data and synchronizing the gradients
    - 1. Split data
    - 2. Each GPU does forward and backward 
    - 3. Synchronize gradients by taking average 
    - 4. Update the weights of the model 


**Learning of the day (2/24)**
- Learning
    - Instance-Based Versus Model-Based Learning?
    - Instance-Based: also known as memory based training, the system memorizes the training examples and makes predictions by comparing new inputs to stored instances using similarity metrics. (KNN)
    - Model-Based: builds a generalizable model from training examples by identifying patterns and learning parameters 
    - Concept
    - Overfitting issues
    - Reduce n_embd
    - Reduce n_layer
    - Increase dropout (regularization)
    - Introduce weight decay (l2 regularization)
    - Reduce vocab size
    - Decrease learning rate


**Learning of the day (2/25)**
- Learning
    - When to use a Label Encoding vs. One Hot Encoding?
    - Use one-hot encoding when the categorical features are not ordinal like year and the number of features isn’t huge. Vice versa for label encoding 


**Learning of the day (2/26)**
- Learning
    - What is the difference between LDA (linear discriminant analysis) and PCA (principal component analysis)  for dimensionality reduction?
    - LDA is supervised whereas PCA is unsupervised. LDA maximizes class separability whereas PCA maximizes variance or information retention 
    - Concept
    - Group policy 


**Learning of the day (2/27)**
- Learning
    - What is t-SNE?
    - T-distributed stochastic neighbor embedding is an unsupervised non-linear learning technique for data exploration and visualizing high dimensional data while preserving the structure and relationship between data points. It is used for visualizing clusters or patterns.


**Learning of the day (2/28)**
- Learning 
    - What is the difference between t-SNE and PCA for dimensionality reduction?
    - PCA is a linear method that projects data on a new axis by capturing the maximum variance. t-SNE is a non-linear method that preserves local relationships between points using probability distributions. PCA is good for fast, linear, global structure, t-SNE is good for better, non-linear, local structure 


**Learning of the day (3/1)**
- Learning
    - What is UMAP?
    - Uniform Manifold Approximation and Projection is a dimensionality reduction technique that is useful for visualizing high dimensional data while retaining local and global relationships.
    - First creates a graph that maps similar data points together. It then tries to decrease the dimension while keeping the shape of the initial graph


**Learning of the day (3/2)**
- Learning
    - What is the difference between t-SNE and UMAP for dimensionality reduction?
    - UMAP is better at capturing the global structure whereas t-SNE is better at capturing the local structure. UMAP is graph based whereas t-SNE is probability based. t-SNE is better for high-quality local clustering whereas UMAP is better for scalability and larger datasets. 


**Learning of the day (3/3)**
- Learning
    - What is the difference between Bayesian vs frequentist statistics?
    - Bayesian statistics use prior knowledge and beliefs into the analysis whereas frequentists do not  


**Learning of the day (3/4)**
- Learning
    - How do you handle an unbalanced dataset?
    - Look at appropriate performance metrics such as precision, recall, F1 score
    - Oversample the rare samples or undersample to abundant samples via bootstrapping
    - Secret synthetic examples using methods such as SMOTE (synthetic minority oversampling technique)


**Learning of the day (3/14)**
- Learning 
    - What are some differences you would see between a model that minimizes squared error vs. an absolute error
    - MSE gives higher weight to large errors, therefore it is more useful when trying to avoid large errors. MAE is more robust to outliers but requires more complex computation.
    - MSE is minimized by conditional mean, MAE is minimized by conditional median

**Learning of the day (3/15)**
- Learning 
    - How do you choose K for K-Means Clustering?
    - Use the “elbow method.” By looking at the graph between number of K and explained variation, there should be a sharp change in the y-axis for a certain k
    - Variation explained is the within-cluster sum of squared errors


**Learning of the day (3/16)**
- Learning 
    - How can you make your model more robust to outliers?
    - Add regularization (L1, L2)
    - Use different models (tree based models are usually more robust to outliers)
    - Winsorize data (cap the data)
    - Transform data: do a log transformation when response variable is right skewed
    - Change error metric: from MSE to MAE
    - Remove outliers


**Learning of the day (3/17)**
- Learning
    - Several predictors are correlated during linear regression. What will happen and how do you resolve this issue?
    - First problem is that the coefficients become unstable and have high variance. The second problem is that the p-value may be misleading leading to uncertainty regarding which features are actually relevant to the model. You can resolve the issue by either combining or removing the correlated predictors with high variance inflation factors. For combining, you can use an interaction term, PCA   
- Concept
    - Label Smoothing 
    - Regularization technique used in classification tasks to improve model generalization and prevent overfitting. Instead of assigning a hard one-hot encoded label, it assigns a softened probabilities to the labels 


**Learning of the day (3/18)**
- Learning
    - What is the motivation behind random forests? 
    - It prevents overfitting by allowing more consistent results through bootstrap sampling of data and training on a more diverse dataset
    - Using only a subset of features at each split helps to de-correlate the decision tree by avoiding using very important features at the top every split. 


**Learning of the day (3/19)**
- Learning 
    - How to handle outliers
        - 1. clarify the missing data (random, non-random)
        - 2. Establish a baseline and check if missing data is crucial
        - 3. Impute missing data (using mean/median/nearest neighbor)
        - 4. Check performance with missing data


**Learning of the day (3/20)**
- Learning
    - Logistic regression output is unsatisfactory, what can I do?
    - 1. Normalize features so a particular weight do not dominate the model
    - 2. Address Outliers
    - 3. Use k-fold cross validation and hyperparameter tuning


**Learning of the day (3/21)**
- Learning
    - What is the difference between gradient boosting and random forests?
    - They are both ensemble methods but have different purposes. Gradient boosting uses boosting which is sequential learning to reduce bias. Each tree is trained on the residual errors of the previous tree. Random Forests uses bagging which is parallel learning to reduce variance. Each tree gets a random subset of data and averages the results of many trees. 
    - Gradient boosting is slower and needs careful hyperparameter tuning. It is better for high-accuracy models


**Learning of the day (3/22)**
- Learning        
    - Is 10,000 data points enough to create an accurate model?
    - Define what is an accurate model, what metrics, type of evaluation metric, type of model etc. 
    - Create a baseline model and check performance. Set a clear goal for the model, see if the model can achieve its goal
    - Add more data if necessary
- Concept
    - Inverted Residual Block
    - Expand data, process data, and shrink again 
    - Smaller Model: Faster and fewer calculations 
    - Depthwise Convolution        
    - Each filter applies to only one channel at a time 
    - Each channel (color) gets its own small filter
    - Squeeze Excitation 
    - Focus on the most important parts of the image. 
    - Squeeze (summarize information) - compress to single number
    - Excitation (Recalibrate Weights) - learns how much attention each channel should get
    - Scale (adjust features)
    - Stochastic depth 
    - Makes training faster by randomly skipping layers during training 


**Learning of the day (3/23)**
- Learning
    - Binary classification on a loan model, rejected applicants must be supplied a reason. How would you supply the reasons without digging into the weights of features?
    - We can look at the partial dependence plot to assess how any one feature affects the model’s decision. Shows the marginal effect of each feature on the predicted target 


**Learning of the day (3/24)**
- Learning
    - How would you identify synonyms from a large corpus of words?
    - We can look at word embeddings using algorithms such as word2vec. Words with similar meanings will have similar vector embeddings. We can use k-nearest neighbors to find similar words. However, there are limitations to this model since words that appear in similar contexts that are not necessarily synonyms may show up.
- Concept
    - Layer Norm vs. Batch Norm
    - Batch norm normalizes across the batch dimension, usually used for CNNs
    - Layer Norm normalizes across the feature dimension, usually used in Transformers


**Learning of the day (3/25)**
- Learning
    - Bias variance trade-off
    - If we decrease bias, variance increases due to overfitting. If we reduce variance through generalization, we lose accuracy. Total error is variance plus bias.  


**Learning of the day (3/26)**
- Learning        
    - K-fold cross validation
    - Divide data into k batches. Train data with all data except for one k-batch and test model with the remaining k-data. Use validation to reduce test loss and prevent overfitting


**Learning of the day (3/29)**
- Learning
    - How would you build a lead scoring algorithm to predict whether a prospective company is likely to convert to a customer?
    - Ask clarifying questions. Are building our own, or are we building as a product, are there any business requirements, are we running only within our database
    - Explain the main features used such as firmographic data, marketing activity, sales activity, deals details,
    - Can create a binary classification model using logistic regression 
    - Monitor changes in data and update model accordingly  


**Learning of the day (3/30)**
- Learning
    - How would you create a music recommendation algorithm?
    - Collaborative filtering 
    - Find users with similar tastes based on user interaction. Compute similarity using metrics such as cosine similarity or pearson correlation coefficient
    - Matrix Factorization/ Helps discover latent factor such as type of genre by decomposing a larger matrix into two smaller matrices (user x item)


**Learning of the day (3/31)**
- Learning
    - What does it mean for a function to be convex? What algorithms are not convex?
    - U-shaped function where local minimum is also global minimum. Line segment between any 2 points lies above or on the graph. 2nd derivative >= 0 for all points
    - Neural networks are not convex. Neural Networks can approximate any function which includes non-convex functions. Use algorithms such as Gradient Descent to find global minimum


**Learning of the day (4/1)**
- Learning
    - What is entropy and information gain?
    - Entropy quantifies uncertainty. High Entropy, values close to 1, means high uncertainty and there’s a 50/50 chance for class distribution. Low entropy, values close to 0, have homogenous classes 
    - Information gain is how much entropy is reduced for each split in decision trees. It is used to select the best features


**Learning of the day (4/2)**
- Learning
    - What is L1 and L2 regularization, what is the difference?
    - Both prevent overfitting by shifting the coefficients of the model closer to 0. Difference is the penalty applied to the loss function. L1 or lasso regularization uses the absolute value of the penalty term and reduces the coefficient values to zero. This is helpful when choosing the most important features. L2 or ridge regularization uses the squared value of the penalty term and reduces the coefficient values close but not exactly to zero. This is useful when all features are important but need to control the impact


**Learning of the day (4/3)**
- Learning
- What is gradient descent and why use stochastic gradient descent?
    - Gradient descent is an optimization algorithm to find the global minimum of a loss function by updating the parameters of the model. It uses backpropagation to find the direction of the steepest descent. Stochastic gradient descent updates the model parameters after each individual training sample rather than computing the entire dataset. It converges faster and helps avoid local minimums due to randomness. Mini-batch gradient descent updates parameters each batch such as 32 data points.


**Learning of the day (4/6)**
- Learning 
    - Classifier that produces a score between 0 and 1, we take the square root of that score, how would the ROC curve change?
    - ROC curve plots the true positive rates vs false positive rates. If all the values change, then the classification threshold would change and this would lead to the same true positive and false positive rates. If the function is not monotonically increasing such as a negative function or a stepwise function, the ROC will change


**Learning of the day (4/7)** 
- Learning
    - X is a univariate gaussian random variable, what is the entropy of X?
    - -p*log(p) 
    - Use p for the gaussian function, take integral, should be 1 


**Learning of the day (4/8)**
- Concept
    - JSON (javascript object orientation)
    - dictionary/object: key-value pairs
    - Easy to read and write, language independent


**Learning of the day (4/9)**
- Learning
    - Compare and contrast Gaussian Naive Bayes and logistic regression?
    - Both can be used for classification 
    - GNB assumes features to be independent but requires a small number of data and is easy to implement
    - Logistic regression is not flexible and can fail to capture interactions between features but it has a simple interpretation in terms of class probabilities
    - Logistic regression requires an optimization step and it’s a discriminative classifier whereas GNB is a generative classifier.  
- Concept
    - Generative vs. Discriminative classifier 
    - Generative learns what a spam email and non-spam email looks like 
    - Discrimination learns the boundary that spits between a spam and non-spam email.
    - Generative classifiers need less data and faster to train but have strong assumptions such as feature independence. Focus on how the data was generated  


**Learning of the day (4/10)**
- Learning
    - What loss function is used in k-means clustering? 
    - Within cluster sum of squares (WCSS) or inertia
    - Squared euclidean distance between each point in the cluster and the cluster centroid 
    - Updates clusters and centroid iteratively 


**Learning of the day (4/11)**
- Learning
    - What is the kernel trick in SVM?
    - The idea behind it is that data that cannot be separated by a hyperplane in its current dimension can be linearly separable by projecting onto a higher dimensional space. Instead of calculating the new position of the data points in higher dimension, we use the kernel function to compute the dot product to determine how similar they are if they were to be projected to a higher-dimensional space. This allows SVMs to draw complex boundaries 


**Learning of the day (4/12)**
- Learning
    - Describe how you’d build a model to predict whether a particular use will churn? 
    - Clarify what churn is and how it is calculated. Talk about modeling considerations such as interpretability. If simple interpretation is required, use logistic regression or decision trees. If not, use SVM or neural networks. Collect features that we will use such as demographics, loss patterns, account balance, etc. See if the model is satisfactory using the ROC curve, F1 scores. Monitor and update results periodically. 


**Learning of the day (4/13)**
- Learning
    - Describe the model formulation behind logistic regression. How do you maximize the log-likelihood of a given model?
    - Logistic regression is used for binary classification where the model outputs the probability of being a certain class. Sigmoid function is used for the probability calculation. Minimize the negative log-likelihood function using gradient descent. 
 
**Learning of the day (4/14)**
- Learning
    - What is an ML pipeline, and why is it important?
    - ML pipeline is a structured workflow that automates various steps in machine learning. It is crucial for reproducibility, scalability, automation, and monitoring and maintenance.


**Learning of the day (4/15)**
- Learning
    - What are the key stages of the ML pipeline?
    - Data Ingestion: collecting, processing data
    - Feature Engineering: Selecting and creating meaningful features
    - Model training
    - Model Evaluation 
    - Model Deployment
    - Monitoring and logging


**Learning of the day (4/16)**
- Learning
    - What is versioning and why is it necessary?
    - Model versioning keeps track of different experiment runs of a machine learning model. It allows comparison between different models and reproduces the same model in the future allowing rollback. 


**Learning of the day (4/17)**
- Learning
    - How would you implement a model registry in an ML pipeline?
    - Storing models with metadata such as parameters and datasets.
    - Using a centralized repository with automized model registry using CI/CD pipelines
    - Defining an approval workflow 
    - Tools: MLFlow, KubeFlow, AWS Sagemaker 
- Concept 
    - CI/CD (continuous integration, continuous delivery) 
    - CI: every time code is pushed on github, it is automatically built and tested (ensures main branch is always on working state) 
    - CD: automatically prepares code for release (build -> package -> release) 


**Learning of the day (4/18)**
- Learning
    - How do you monitor a deployed ML model?
    - Track performance metrics, Detect data drifts, set alerts for model anomalies


**Learning of the day (4/20)**
- Learning
    - What types of testing are necessary for an ML pipeline?
    - Unit Testing: verify each individual function works correctly
    - Integration Testing: Ensure interaction between pipelines
    - Regression Testing: verify updating the model doesn’t degrade performance
    - Performance testing: evaluation inference speed and scalability


**Learning of the day (4/21)**
- Learning
    - How does CI/CD work in ML pipelines?
    - Automating model training and validation, running performance checks before deployment, rolling back to previous models 
    - Github Actions, MLFlow, KubeFlow
- Concept
    - Kubernetes
    - Open source container orchestration platform that helps automate deploying, scaling and managing containerized applications (docker) across clusters of machines. Manages containers at scale and auto-restarts dead containers. Handle traffic spikes (load balancing) 
    - ALiBi (relative positional encoding)
    - Attention with Linear Biases. Instead of adding position vectors, add a bias to the attention score so that model prefers to attend to recent tokens more than far ones. Faster to compute, no need for positional embeddings, better at longer sequences. 


**Learning of the day (4/22)**
- Learning
    - What is a canary deployment?
    - Releasing a new model to a subset of users to minimize risk and allow real world monitoring with the possibility of a roll back if performance declines
- Concept 
    - Mixed-Precision Training 
    - Using half precision floats (FP16) instead of full precision floats (FP32). Trade off between numerical stability and memory footprint 
    - Cannot represent floats smaller than 2^-24 and underflows to 0 and no changes to weight values. 
    - Using FP16 activates tensor cores
    - Solution
        - Loss scaling
        - Multiply the gradients to a constant scale factor and shifting the gradients to a value greater than 2 ^-24 
        - Can apply to loss rather than gradients to achieve same output due to chain rule and back prop 
        - FP32 Master Copy of weights 
        - Use FP16 for forward and backward pass, FP32 for the optimizer step 
        - Model memory increases but activation and gradient memory decrease (bigger portion)
        - Mixed Precision Arithmetic 
        - Can do mixed precision thanks to cuda. Read and write in FP16 but compute type of FP32


**Learning of the day (4/23)**
- Learning
    - How do you reproduce a ML pipeline?
    - Versioning code and data, setting random seeds, using containerization
    - What are the biggest challenges in deploying ML models?
    - Scalability, latency, model drift, managing compute costs, security and compliance for protecting sensitive data and meeting regulations 
- Concept 
    - LIMA (Less is more alignment)  
    - Superficial alignment hypothesis
    - A model’s knowledge is almost entirely learned during pretraining 
    - Alignment -> what part of the model to use 
    - Short fine tuning can ruin less of the pretrained knowledge and avoid catastrophic forgetting  
    - small finetuning dataset can be better 


**Learning of the day (4/24)**
- Learning 
    - What are the best practices for scaling ML models?
    - Batch inference: processing data in groups rather than real time
    - microservices architecture: deploying models as independent services 
    - Model caching: storing frequent predictions for quick retrieval 
    - What strategies can be used to optimize ML pipeline?
    - Distributed data processing: (Apache Spark, Hadoop) break up large data tasks into multiple machines
    - Feature store integration: implement a centralized feature store to prevent redundant calculations 
    - Parallel Processing: train models in parallel using GPUs, Cloud
    - Auto-scaling infrastructure: deploy models in kubernetes to leverage auto scaling 




**Learning of the day (4/25)**
- Learning
    - How do you handle long-running ML training jobs efficiently?
    - Checkpointing, using cloud-based spot instances, gradient accumulation: accumulate gradients over multiple mini-batches, Data-pipeline optimization   




**Learning of the day (4/26)**
- Learning
    - How is RAG different from traditional LLMs?
    - Unlike traditional models which rely on pre-trained knowledge, RAG can access external knowledge sources through the retrieval step 


**Learning of the day (4/27)**
- Learning
    - What is the purpose of the max token parameter?
    - Sets an upper limit of how many tokens the model can produce at once. Helps control model API costs 


**Learning of the day (4/28)**
- Learning
    - What is empirical
    - Based on observation or experience rather than theory or pure logic 


**Learning of the day (4/29)**
- Learning
    - What is top P and how does it differ from top-k?
    - Top-p samples the output until the cumulative probability exceeds p whereas top-k samples the fixed k-number of tokens. Top-p is more fluent top-k for more consistency 


**Learning of the day (4/30)**
- Learning 
    - What is the role of the frequency penalty parameter?
    - Frequency parameter penalizes the model from repeating the same tokens too often. It improves lexical variety and reduces redundancy. Similar to presence penalty parameter which penalizes tokens that have already appeared regardless of frequency 


**Learning of the day (5/1)**
- Learning
    - How do stop sequences work, why are they important?
    - Stop sequences are specific tokens or sentences that tell the model to stop generating. They are critical for enforcing the response structure.


**Learning of the day (5/2)**
- Learning
    - How do temperature and top_p work together?
    - You divided the output logits by temperature parameter 
    - Lower temperature with moderate p -> coherent and focused 
    - High temperature and high top p -> more creative output 


**Learning of the day (5/4)**
- Learning
    - Explain how quantization improves LLM efficiency. What are the trade-offs?
    - Faster inference and lower memory by using int8 precision rather than float32
    - Slight accuracy drop
    - Uses ONNX Runtime 
    - Concept
    - ONNX Runtime 
    - Open Neural Network Exchange - high performance inference engine 


**Learning of the day (5/5)**
- Learning
    - What is an inference engine?
    - Inference engine takes a trained model to make predictions based on inputs. They are optimized for speed, efficiency, and hardware acceleration 


**Learning of the day (5/6)**
- Learning
    - What strategies can be used to detect model drift? 
    - Monitor performance metrics such as accuracy, precision 
    - Use statistical tests such as Kolmogorov-Smirnov test to check divergence between data distributions 
    - Build empirical cumulative distribution function and compare vertical distance


**Learning of the day (5/7)**
- Learning
    - How do you ensure security in the ML pipeline?
    - Adversarial testing - test models against adversarial inputs to detect vulnerabilities 
    - Concept
    - WANDA (Weights and activation pruning with norms and discrepancies Awareness)
    - Magnitude based structured pruning method. It uses weight importance derived from activation statistics. It measures the importance of weights by combining both the magnitude of the weights and the activation patterns. See how big the weights are and see which weights are activated when an input is passed in 
    - It is a post training pruning method and requires minimal fine-tuning 


**Learning of the day (5/8)**
- Learning 
    - How does the temperature parameter impact model output?
    - Temperature controls the randomness of the model output. Low temperatures are more deterministic whereas high temperatures are more creative 
    - Concept 
    - LongNet (Dilated Transformers) 1 Billion token context window 
    - Instead of looking at all the tokens, it skips over 2, 4, 8 tokens. Different heads look at different parts of the input token. Some look at close words, some look at far away words  


**Learning of the day (5/9)**
- Learning 
    - How would you debug an LLM that produces overly verbose responses in a chatbot?
    - Reduce temperature, include frequency penalty parameter, add stop sequences 
    - Concept


**Learning of the day (5/10)**
- Learning
    - Deterministic vs stochastic?
    - Deterministic produces the same output given the same input
    - Stochastic includes randomness so the same input can lead to different outputs 


**Learning of the day (5/11)**
- Learning
    - Describe key non-functional requirements (NFRs) for an LLM system.
    - Latency < 1sec
    - Cost: optimize model size 
    - Scalability: Auto-scaling tools such as kubernetes
    - Security: Data Encryption 
    - Reliability: fallback models 
        


**Learning of the day (5/13)**
- Learning
    - Walk through designing a customer support chatbot (real-world case study)?
    - Ask what the requirements are.
    - Frontend: use react based UI
    - Backend: use Fast API, host model on cloud service such as AWS, use RAG pipeline to fetch order details. (pinecone - vector DB). use langchain to manage context (memory, history documents) 
    - Optimization. Use quantized or distilled models. Cache frequently asked prompts 


**Learning of the day (5/14)**
- Learning
    - How would you design a system to handle rate-limiting for LLM API requests? 
    - Token bucket algorithm: track requests per user IP
    - Fallback: serve cached responses or lighter model
    - API gateway: enforce limits via NGINX
- Concept
    - NGINX 
    - Middle man that helps traffic go to the right place
    - Receives requests from users (port 80)
    - Pass those requests to fastAPI
    - Returns the response to the user 


**Learning of the day (5/15)**
- Learning
    - Design a multi-agent LLM system where different models handle different tasks 
    - Router model: classifies queries to appropriate model
    - Orchestrator: langchain 
    - Data processing: JSON 



**Learning of the day (5/16)**
  - Learning
    - How would you reduce GPU costs for a high-traffic LLM app?
    - Model Distillation, quantization. Using the Mixture of Experts model to only activate relevant layers. Cache frequent responses 
  - Concept
    - Serverless GPU 
    - You just send a job such as inference, the cloud provider starts a GPU, runs the job, and shut it down. You only pay for what you use
    - Cost efficient, on-demand usage, and auto-scaling 
    - AWS lambda server


**Learning of the day (5/17)**
  - Learning
    - Design Chat with PDF?
    - 1. Upload pdf
    - 2. Chunk and embed. Divide pdf into paragraphs and vectorize it. Store it to pinecone vector DB 
    - 3. Query. User gives prompt and RAG fetches top chunks using FAISS (facebook ai similarity search) 
    - 4. LLM summarizes and gives output 
  - Concept
    - Async processing 
    - Way for programs to handle multiple tasks at the same time without waiting for each one to finish before starting the next.


**Learning of the day (5/18)**
  - Learning         
    - Scale an LLM app from 100 to 1M users?
    - Ask what does the app do, are responses real-time, what model is used
    - Possible issues are 
        - Compute bottleneck
        - Delayed inference latency
        - Memory and storage 
        - Cost management 
    - For optimization 
        - Use distilled models, quantization, deploy models on vLLM
    - Infrastructure
        - Use kubernetes for horizontal scaling 
        - Set up load balancers
    - Storage
        - Store embeddings in FAISS or pinecone
    - Rate limiting 
        - Add rate limiting to nginx or api gate-way
    - Cost
        - Use spot instances or serverless GPUs
  - Concept
    - Orchestration platform
    - System that automates management, coordination, and scaling of multiple services or components.
    - Web request handling: spawns more instances 
    - Prompt processing: sends prompts to right worker  
    - Text-to-speech job: runs async TTS jobs
    - Logging and analytics: collects logs and usage data


**Learning of the day (5/19)**
  - Learning
    - How does vLLM work?
    - vLLM is a high throughput memory efficient inference engine. 
    - Uses paged-attention which dynamically allocates memory for kv-caching (only uses memory needed rather than a fixed size)
    - Supports dynamic batching: adds new user prompts on the fly rather than fixed schedules. They can also have varying lengths 
    - Supports token streaming: tokens are returned as they are generated


**Learning of the day (5/20)**
  - Learning
    - What are two primary types of LLM evaluation?
    - Reference based metrics: compare generated output against BLEU, ROUGE, METEOR, etc
    - Reference free metrics: tone, coherence, structure
- Concept 
    - N-grams
    - Comparing two pieces of text by looking at how many small word groups they have in common. It measures how similar texts are.
    - BLEU score 


**Learning of the day (5/22)**
  - Learning
    - How does BERTScore work?
    - Bert score calculates the concise similarity between token embeddings. 


**Learning of the day (5/23)**
  - Learning
    - What are reference-free metrics, and when are they useful?
    - Evaluates output without needing reference text making it ideal to evaluate tone, open-ended generation
    - Regex-based checks, Deterministic validation: check if output is in correct format


**Learning of the day (5/24)**
  - Learning
    - How are Contextual Precision and Recall applied in RAG systems?
    - Contextual Precision: measures how relevant the retrieved documents are
    - Contextual Precision: Assess rather enough information was retrieved 


**Learning of the day (5/25)**
  - Learning
    - Which metric detects hallucinations effectively?
    - HallucinationMetric flags content that deviates from the provided context or source.


**Learning of the day (5/26)**
  - Learning
    - Why is BLEU limited, and when does it fall short?
    - BLEU (Bilingual Evaluation Understudy) focus on n-gram matches missing semantic nuances and struggles with paraphrased or creative output


**Learning of the day (5/27)**
  - Learning
    - Which Python libraries support BLEU, ROUGE, and METEOR?
    - Huggingface evaluate library 


**Learning of the day (5/28)**
  - Learning 
    - What are the drawbacks of embedding-based metrics?
    - Ignore syntax, overlook factual accuracy, inflate similarity from shared vocabulary 


**Learning of the day (5/29)**
  - Learning
    - What are Agentic Design Patterns?
    - Frameworks that enable to agents autonomously reason, act, self-improve using memory, tools, reflection, and multi-agent frameworks 


**Learning of the day (5/30)**
  - Learning
    - Name 3 common agentic patterns
    - ReAct (reasoning and acting), CodeAct (dynamic code and generation), agentic RAG (rag with with memory and tools)


**Learning of the day (5/31)** 
  - Learning
    - What is the role of the ReAct agent?
    - It enables the agent to alternate between reasoning (LLM-based) and performing actions (external tools/APIs) dynamic and context-aware decision making 


**Learning of the day (6/1)**
  - Learning
    - How does the Model Context Protocol (MCP) support Modern Tool Use?
    - MCP defines how context is represented and passed to a model. MCP uses structured tool_calls in messages which enables automatic routing to the correct tool. 
    - Tells the model which tools to use in, structured message, and keeps memory and tools organized. Separates thinking and doing. 




**Learning of the day (6/2)**
  - Learning
    - What purpose does the Self-Reflection pattern serve?
    - It enables agents to critique their own output, look for errors, iteratively refine answers
    - How does a Multi-Agent Workflow improve efficiency?
    - By distributing tasks among specific agents, enabling parallelism 


**Learning of the day (6/3)**
  - Learning
    - How would you design a customer support agent for a financial institution using Agentic Design Patterns?
    - I would use a ReAct agent to handle customer queries and perform actions. Add tools with API integrations, Self reflect on answers and Agentic RAG to retrieve policy documents and past conversations 


**Learning of the day (6/4)**
  - Learning
    - Why is memory (checkpointer) critical in agentic design? 
    - It enables long-term coherence, recovery from interruptions, records intermediate states so it doesn’t forget the previous context, it also stores which tools are called, what the output of the tool call was and what to do next


**Learning of the day (6/5)**
  - Learning
    - Preventing infinite reflection loops?
    - Use critique thresholds, capped iteration loops.
  - Concept
    - CORS (Cross-Origin Resource Sharing)
    - Security feature built into web browsers that controls which websites can make requests. Protects from malicious sites 


**Learning of the day (6/6)**
  - Learning
    - SSL/TLS certificates?
    - Certbot is a free, open-source tool that helps you automatically get and renew SSL/TLS certificates from Let's Encrypt, so your website can use HTTPS.
    - SSL/TLS certificates are digital certificates that provide secure, encrypted communication between a user's browser and a website server.
        - SSL (Secure Sockets Layer) - older
        - TLS (Transport Layer Security) - modern

**Learning of the day (6/7)**
  - Learning
    - What’s the difference between “tool use” and “tool learning” in agentic systems?
    - Tool use refers to invoking predefined tools or APIs.
    - Tool learning enables agents to autonomously select or even learn how to use new tools during runtime.


**Learning of the day (6/8)**
  - Learning
    - How do you handle context window limitations in long-running agentic tasks?
    - Use vector databases.
    - Implement episodic memory compression 
        - Have AI remember past conversations but summarize older conversations to reduce memory
    - Use summarization patterns to reduce context load. 

**Learning of the day (6/9)**
- Learning
    - Improving model interactivity 
    - Think about how a human will interact 
        - turn-taking (interruptions)
        - Voice activity detection
    - Real time streaming rather than audio upload
        - Low latency. Transcribe as audio comes in
        - Enable background response 
        - Real-Time:
            - Use low latency model
            - Break data into chunks and process each chunk. Use caching for previous input 
        - Integrate context tracking 
            - Model remembers previous conversations 
            - Use Vector DB to store semantic embeddings 
        - Have model understand emotions 
        - Gather audio samples with labeled emotions
            - Have ‘better’ model label emotions
            - Use dataset with emotions (metadata)
            - Train model to classify correct emotion 
            - Maybe use MoE: one for emotion, one for context 


**Learning of the day (6/10)**
- Learning
- How to set agent Entry point?
- Goal is to create an action classifier 
- First, need an supervised or labeled dataset with a question and agent pair 
    - This dataset can be synthesized through a more advanced model
    - Or you can use human labeling
- Next, you will train a classifier model where it will select the correct call based on the context (cross entropy loss)
- Use PPO reinforcement learning where the base model is the policy and the reward is choosing the correct tool. Use K-L divergence to keep baseline model intact 

**Learning of the day (6/11)**
- Learning
- Test-time reinforcement learning 
- learning or adaptation happens during inference or evaluation rather than a separate training phase
- Allows the model to continue improving based on the feedback it receives during use 
- Benefits: reduces the need for human labels or reward models 
- It receives feedback by 
    - Self-evaluation: confidence based
    - External signals: human-feedback
    - Environment-derived: failed a task 
- How can you optimize performance in multi-agent frameworks?
    - Asynchoronos task execution
    - Caching frequently used data 
    - Agent prioritization


**Learning of the day (6/12)**
- Learning
- Fully Sharded Data Parallel (FSDP) is a distributed training method designed to efficiently train large-scale deep learning models by sharding (splitting) the model's parameters,
- 1. Sharding
    - Each GPU only holds a shard of the model
- 2. Sharing gradients and optimizer states 
    - Gradients and optimizer states are shared (each GPU has a component of the gradients and optimizer states - momentum/weight decay and these info are shared when needed) 
    - Same input batch is broadcasted to all GPUs
- 3. Communication overlap
    - Overlaps communication with computation during all-gather
- 4. All gather at Forward pass:
    - During the forward pass, FSDP temporarily gathers the required shards so the layers can be computed 
- 5. All Reduce at Backward pass
-   During backward pass, gradients are reduce-scattered to distribute them 

- How do you mitigate prompt injection in agentic systems?
    - Input/output validation
    - Role-based tool access
    - Instruction hardening: make it harder to bypass or override models behavior 


**Learning of the day (6/13)**
- Learning 
    - Decoding strategies
    - Greedy: picks highest probability each step
    - Beam: maintains top-k sequences each step
    - Top-k: randomly selects from top-k probable tokens
    - Top-p: randomly selects from tokens where their cumulative probability exceeds p

- How do you prevent over/underfitting in generative models?
    - Over: Early stopping, dropout, noise injection, regularization (weight decay) 
    - Under: increase model depth, layers, training steps 


**Learning of the day (6/14)**
- Learning
- How do prepare text data for LLM pretraining
    - 1. Collect data from the web 
    - 2. Clean and normalize
        - Bad encodings, extra spaces
    - 3. Filter low-quality content
    - 4. Deduplicate
    - 5. Tokenize
    - 6. Batching through data loaders 

- What is the role of the latent space in generative models?
    - Latent space is a compressed representation of the input data. Encoder maps a higher dimensional data into the latent space and the decoder takes the data from the latent space and expands it back to the higher dimension 

**Learning of the day (6/15)**
- Learning 
- Preparing fine-tuning dataset
- Format data by task
    - Text classification 
    - Text-label pair 
    - Instruction tuning (system, User, assistant messages)
- Watch out for inconsistent formatting, Need very high quality data, make sure each input-output pair matches perfectly
- Have LLM oversee the dataset to check for any issues or errors 

- What are some limitations of current generative models?
    - Hallucination and factual inconsistency
    - High compute and environmental costs
    - Poor performance in low-resource domains 

**Learning of the day (6/16)**
- Learning 
- Magistral: mistral model - reasoning better by only using RL (no SFT)
- How would you vertically scale your rapGPT 
    - 1. Use GPU instance
    - 2. Optimization using vLLM or ONNX (high performance inference engine)
    - 3. Replace raw FastAPI with optimized model servers such as NVIDIA triton inference server
    - 4. Use batch inference and cache frequent values. Use Async requests   


**Learning of the day (6/17)**
- Learning 
- How would you horizontally scale your rapGPT
    - Containerize the model server that has the FastAPI backend and model
    - Run multiple instances using kubernetes
    - Use a load balancer like NGINX
    - Deploy on cloud  
- MOE model
    - activates subset of the neural network for each input based on the gating network
    - benefits:
        - Efficient, Scalable, Specialization, Parallelism 
    - Downsides: 
        - routing complexity, uneven utilization, infrastructure requirements (Good GPUs)

**Learning of the day (6/18)**
- Learning 
- Test Time Scaling
    - Enhancing a model’s performance at inference time without changing model weights or retraining
- Strategies
    - 1. Parallel Sampling Methods: Generate multiple answers and pick the best one  
        - Best-of-N (BoN): Generate N outputs, pick the best.
        - Beam search: Explore top-k high-probability sequences.
        - Tree search: Build a decision tree of steps/outputs.
    - 2. Result Verification and Merging 
        - When multiple outputs are generated, how do we choose the final answer?
            - Voting (majority)
            - Scoring (pick the one with best self-evaluation)
            - List-wise methods (rank all outputs and merge)
    - 3. Sequential Revision: Reflection and Self-Refinement
        - Reflecting only when performance drops works better than reflecting at every step.
    - 4. Multi-Agent Collaborative Sampling
        - multiple agents generate different candidate solutions in parallel, then merging them.

**Learning of the day (6/20)**
- Learning
    - What is a Memory Leak? What are the common causes, and How do you handle them?
    - Memory leak happens when the program holds onto the memory that it no longer needs. This can slow down your program or cache it. Even though python has automatic garbage collection, memory leaks can happen due to leftover references to objects, unused global variables, circular references, cache that is not cleared, etc. Important to clean and remove unused objects, limit cache size. Use tools like trace malloc.

**Learning of the day (6/21)**
- Learning
    - Class Methods vs Static Methods?
    - Both class method and static method doesn't rely on on instance-specific data. Class method is defined with @classmethod and take cls as its first argument, which refers to the class. It can access and modify class-level data and is used for factory methods - creating object in a specific way. Static method is defined with @staticmethod and doesn't take in self or class. It behaves like a regular function but is kept inside class for logical organization. Static methods doesn't need class or instance    
    - Class Methods vs Instance Methods
    - Instance methods (def) take in self as an argument and work on instance specific data (has the same class but different instances). Class methods modifies the entire class-level data 

**Learning of the day (6/23)**
- Learning
- How do you access a parent class method?
    - By Inheriting the parent class in the child class, then accessing the parent method in child class.
    - By Inheriting the parent class in the child class, and using super().

**Learning of the day (6/24)**
- Learning
- How do you debug Python code in production systems?
    - Monitoring logs 
    - Enable metrics and alerts using tools such as Prometheus 
    - Replicate in staging: reproduce production like env in local 

 **Learning of the day (6/25)**
- Learning
- List vs Tuple
    - Lists: Mutable, two blocks of memory, element can be modified easily 
    - Tuple: Immutable, one block of memory, element cannot be changed 
- Array vs Lis
    - Array: imported for numpy, size cannot be resized, uniform data type, optimized for arithmetic 
- How is memory managed in python?
    - Memory management is handled by the Python private heap space. User doesn't have access 
    - Python has built in garbage collector which recycles all unused memory 

 **Learning of the day (6/26)**
- Learning
- How do you achieve multi-threading in python?
    - Python has a construct called the Global Interpreter Lock (GIL).
    - The GIL makes sure that only one of your threads can execute at any one time
    - true parallel execution of threads is limited when working with CPU-bound tasks.
- What are lambda functions?
    - anonymous one line function which returns an object 
    - use where functions are passed as arguments, such as with map(), filter(), or sorted().
- What is pickling and unpickling?
    - Pickling is the process of converting a Python object (such as a list, dictionary, or custom object) into a byte stream.
    - Unpickling is the reverse process: converting the byte stream back into the original Python object.
    - This is commonly used for storing trained machine learning models, caching data, or persisting Python objects between program runs. 
- What advantages do NumPy arrays offer over (nested) Python lists?
    - NumPy array is faster and you get a lot built-in functions, can do “vectorized” operations 