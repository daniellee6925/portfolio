# 📘 **Learning of the Day**

# research 
   1. Current LLMs are not great and following rules
   2. Possible improvements are
      1. test-time steering 
      2. supervised fine-tuning.
# Learning
   1. Regularization: prevents overfitting by adding a penalty term to the loss function 
   2. L1: Lasso -> when many features are useless 
   3. L2: ridge -> all features are useful but need to control the impact


---
# 📘 **Learning of the Day**

4. Learning
   1. Supervised learning: train model with labeled data: regression
   2. Un-supervised learning: clustering, PCA


---
# 📘 **Learning of the Day**

   1. What is gradient descent?
   2. Optimization algorithm to minimize loss function by iteratively updating the model parameters in the opposite direction of the gradient of the loss function
   3. ADAM: adaptive learning rates
      1. Too low: converge too slowly
      2. Too high: overshoot the minimum
   4. Stochastic vs Batch
      1. Batch: entire dataset
      2. Stochastic: randomly selected subset of the dataset


---
# 📘 **Learning of the Day**

   1. How do you combat the curse of dimensionality? 
   2. Exponential increase in computational effort as the number of dimension increases
      1. Harder to find patterns
      2. overfitting 
      3. Computational complexity 
   3. Feature Selection, PCA, t-SNE, UMAP, LLE (locally linear embedding)
      1. LLE: project high-dimensional data into a lower-dimensional space while preserving the local relationships between data points
      2. t-SNE: maps high-dimensional data onto a lower-dimensional space
      3. PCA: identifies direction in which data varies the most
   1. PPO (Preferred Policy Optimization)
      1. Used for reward model
      2. Policy function: prob of choosing action at a certain state
      3. Reward function: expected reward of action at a certain state (Q)
      4. Back propagation using negative loss function -> gradient ascent (global maximum)


---
# 📘 **Learning of the Day**

   1. Why use RELU over Sigmoid?
      1. Computational efficiency
      2. Reduce likelihood of vanishing gradient
      3. Sparsity (neuron inputs are negative and less activated) -> network is lighter
   1. Vanishing Gradient
      1. gradients used to update the network weights become extremely small during backpropagation
      2. prevents the network from learning effectively in deeper layers
   2. Solutions
      1. Use ReLU
      2. weight initialization strategies like Xavier (1/input+output - Sigmoid/tanh) initialization or He(kaiming) initialization (2/n - ReLU)
         1. Scaled variance - balances signal across layers so activations don’t get too large or too small
   3. Exploding Gradient
      1. gradients calculated during backpropagation become excessively large
      2. drastic weight updates, potentially causing the model to jump across the loss landscape and fail to converge to an optimal solution
      3. Happens when there are many layers and using activation functions such as Sigmoid: Jump in weights, NaN
   4. Solution
      1. Use optimizer such as ADAM to control learning rate
      2. Use RNN (LSTM) 
      3. Gradient clipping: limit size of gradients from becoming too large 
4. Neural Network from Scratch
   1. Learning rate decay: decreasing the learning rate over time/steps 
      1. Good way is to program a decay rate
   2. Momentum: creates a rolling average of gradients and uses this average with the unique gradient each step
      1. Uses previous update’s direction to influence the next update’s direction 
      2. With momentum, more likely to pass through local minimums


---
# 📘 **Learning of the Day**

   1. Describe how convolution works?
   2. a convolution operation is applied to the input image using a small matrix called a kernel or filter.
   3. The kernel slides over the image in small steps, called strides, and performs element-wise multiplications with the corresponding elements of the image and then sums up the results -> this result is called a feature map 
   4. Benefits
      1. Robust to noise
      2. Reduces memory and computational needs (reduce dimensionality) 
      3. Feature learning: automatically learn features from raw data
      4. Transfer learning - can be pretrained 
   5. Disadvantages
      1. Require a large amount of labeled data
      2. Prone to overfitting 
   1. DPO: Instead of optimizing reward function, optimize the optimal policy


---
# 📘 **Learning of the Day**

   1. What is data normalization and why do we need it
   2. Data normalization is a process of rescaling the input data so that it fits within a specific range, usually between 0 and 1. It is calculated by subtracting the mean and dividing by the standard deviance. Data normalization ensures all features are weighted equally and assures better convergence during back propagation
   1. There is a token to specify the start and end of sentence 
      1. SOS: Start of Sentence
      2. EOS: End of Sentence
4. Neural Networks
   1. decay  -> reduce learning rate
   2. Momentum -> creates a moving average of gradients and uses this with the unique gradient at each step
      1. More likely to pass through local minimums 
   3. Adagrad (Adaptive Gradient) & RMSprop -> normalize parameter updates by keeping history of previous updates
      1. Bigger the update is in a certain direction, the smaller updates are made further in training
      2. Learning rates for parameters with smaller gradients decrease slowly and vice versa 
      3. Less likely to overshoot global minimum
      4. RMSprop: divide by the square root of gradients squared 
   4. ADAM
      1. Combination of Momentum and RMSprop
      2. Parameters beta1, beta2,
         1. Learning rate: base learning rate
         2. Decay: rate of decay
         3. Epsilon: prevents dividing by zero for RMSprop
         4. Beta_1: how much weight to give to momentum
         5. Beta_2: how much weight to give to cache


---
# 📘 **Learning of the Day**

   1. Why do we max pool in classification CNNs?
      1. Max pooling reduces computations since it feature maps become smaller
      2. enhances invariance to object position and orientation within an image
      3. Potentially prevents overfitting by focusing on the most prominent features in a local region
3. Neural Network
   1. Regularization
      1. Calculate a penalty to add to the loss value
      2. Penalize model for large weights and biases
         1. Generally better to have many neurons contribute rather than a select few
      3. L1: absolute value
      4. L2: squared value 
      5. Hyperparameter Lambda
         1. Delicate how much impact we want the regularization penalty to carry
         2. Higher the lambda, higher the penalty 
4. Concept
   1. Negative Sampling 
      1. Solves computational program 
      2. Minimize the probability that randomly sampled word-context  (2~20) pairs are observed rather than the entire dictionary


---
# 📘 **Learning of the Day**

   1. What is the significance of residual networks? 
   2. Introduce residual learning to address challenges such as vanishing gradient problem
   3. Uses skip connections that bypass one or more layers which helps network focus learning the differences, making optimization easier 
   4. Allowed direct feature access from previous layers: features from previous layers are retained and important information is not lost. 
   1. Encoder -> context vector -> decoder
      1. Encoder combines words in the sequence and extract the “meaning”
      2. This “meaning” is stored in the context vector 
      3. The context vector is inputted to the decoder and similar words are outputted based on similar vectors (which means similar words)
4. Neural Networks        
   1. Dropout
      1. Disables some neurons, while other are unchanged
         1. Disables during every forward pass, learn to make predictions with only random neurons remaining
         2. Disabled mean setting the output values to zeros
      2. Prevent becoming too dependent on any neuron
      3. Prevent co-adoption: neurons depend on output of other neurons and doesn’t learn function on their own
      4. Aims to combat overfitting


---
# 📘 **Learning of the Day**

   1. What is Batch Normalization and how does it work?
   2. Batch Normalization makes training faster and more stable by normalizing the inputs to each layer. It’s like resetting each layer ensure the inputs are neither too big nor too small
   3. Subtract mean and divided by STD
   4. Introduces the shift and scale learnable parameters
   5. Faster convergence and reduces overfitting
   1. Recurrent Neural Networks
   2. RNNs have an internal memory (hidden state) that captures information about prior inputs in the sequence. This means that it keeps earlier data to help understand the next output
   3. The same weights are used across all time steps and reduces the number of parameters
   4. Problems:
      1. vanishing/exploding gradients
      2. Short-term memory: forget what happened long time ago
   5. Fixes
      1. LSTM: Long short term memory
      2. GRU: Gated Recurrent Unit


---
# 📘 **Learning of the Day**

   1. Why would you use many small convolutional kernels such as 3x3 rather than a few large ones?
   2. smaller kernels allow less parameters and computations for the same receptive field and spatial content
   3. Will be using more filters and more activation functions and have a more distinct mapping functions 
   1. Attention:
      1. Communication mechanism between tokens 
      2. No notion of space, which is why we need to position encoding 
      3. Each example across batches is completely independent.(They do not talk to each other
      4. Encoder block: allow all tokens to communicate
         1. ex) used for sentiment analysis
      5. Decoder block: mask future tokens 
         1. Autoregressive settings: predictions made sequentially based on previous inputs or outputs 
      6. In Self attention, all K, Q, V comes from the same source 
      7. In Cross Attention, K, Q, V can come from different nodes
         1. ex) use Keys to pull information from different nodes (Q, V)
      8. Scaling (divide by sqrt(head_size) is used to prevent softmax values from being too extreme 
   2. Logit
      1. raw, unnormalized output of a neural network layer 
      2. Before activation function
      3. Used as an input for softmax
   3. Batches
      1. Batch size = number of samples processed at once
      2. Parallel computation - efficient
   4. Self Attention
      1. Keys:what do i contain?
      2. Queries: what do I look for?
      3. Affinity: dot product between key and query
      4. In Self attention, all K, Q, V comes from the same source 
      5. In Cross Attention, K, Q, V can come from different nodes
         1. ex) use Keys to pull information from different nodes (Q, V)


---
# 📘 **Learning of the Day**

   1. Why do we need a validation set and test set? What is the difference between them? 
   2. We divide the data into 3 sets, Training, Validation, and Test
   3. The training set is used to learn and fit the model parameters
   4. The validation set is used to tune the hyperparameters of the model.
   5. The test set is used to measure how well the data performs on unseen data. 
   1. Residual Connections (skip connections)
      1. Intuition: "Remember what I said earlier, and add this new piece”
      2. Secure original information is retained and not lost
      3. How: adding input directly to the output before activation 
      4. Used in Transformer Blocks 
      5. Benefits
         1. Reduce vanishing gradient 
         2. Preserves information
         3. Faster convergence


---
# 📘 **Learning of the Day**

   1. What is stratified cross-validation and when should we use it?
   2. Cross validation is the process of dividing the data in train sets and validation sets.
   3. Stratified CV splits the data categories into the same ratios.
   4. LOOCV is used by using one set as the validation set and using the rest as the train set. 
   5. CV is used for hyperparameter tuning to prevent overfitting


---
# 📘 **Learning of the Day**

   1. Why do ensembles typically have higher scores than individual models?
   2. Ensemble uses multiple models for a single prediction. The key idea is to have each model make different errors so the other model can compensate for it by making the right prediction. 
   3. It also helps with overfitting by reducing the impact of a single model’s biases.


---
# 📘 **Learning of the Day**

   1. What is an imbalanced dataset? Can you list some ways to deal with it?
   2. Different proportions of data in each category 
      1. oversampling/undersampling 
      2. Data augmentation 
      3. Using appropriate metrics such as precision, recall, f1 score 
   1. Pytorch Steps
      1. Set to train()
      2. Logits = model(X_train)
      3. Preds = softmax
      4. Calculate loss
      5. Calculate accuracy
      6. Set zero_grad 
      7. Backward step
      8. Optimizer step
      9. Set to eval()  
      10. Inference mode


---
# 📘 **Learning of the Day**

   1. What is data augmentation?
   2. Technique for synthesizing new data by modifying existing data such that the target is unchanged or changed in a known way 
   3. Computer vision:
      1. Resize, reshape, rotate, add noise, modify colors, deform
   1. Auto-encoder
      1. Encoder -> compresses data into lower dimension
         1. This can be done by capturing similar data structures
      2. Latent space ->  bottleneck that enforces dimensionality reduction
      3. Decoder -> reconstructs original data from latent representation
      4. Usage
         1. Dimensionality reduction 
         2. Data denoising
         3. Anomaly Detection
         4. Data generation


---
# 📘 **Learning of the Day**

   1. Precision, Recall, F1 Score?
   2. Precision: percentage of true positives among all labeled as positives 
   3. Recall: percentage of true positives identified among all positives 
   4. F1 Score: weighted average of precision and recall 
   1. PyTorch
      1. DataLoader: stored data into batches based on transforms
      2. Predicted values from models are logit


---
# 📘 **Learning of the Day**

   1. Define Learning Rate?
   2. Hyperparameter that adjusts how much we are updating the parameters of the model with respect to the loss gradient
   3. If the learning rate is too high, it may not converge to the global minimum. If learning rate is too low, it may take too long to converge
   4. We can apply adjustments such as decay (L2 regularization) and momentum to the learning rate


---
# 📘 **Learning of the Day**

   1. What is momentum?
   2. Momentum adjusts the learning rate appropriately by keeping track of the past gradients by calculating the moving average. Momentum allows the optimizer to avoid getting stuck at local minimums
3. Web/PDF scraping 
   1. PDF: Import fitz (PyMuPDF)
   2. Web Scrapping: Jina_ai


---
# 📘 **Learning of the Day**

   1. What is the difference between Batch Gradient Descent and Stochastic Gradient Descent?
   2. Batch Gradient Descent computes the gradients for the entire dataset whereas Stochastic Gradient Descent computes gradient using a single sample 
   3. Batch Gradient Descent is better for smaller datasets since it has a slow computational time and memory usage in exchange for smoother convergence
   4. There is also a mini-batch gradient descent witches uses one-batch 


---
# 📘 **Learning of the Day**

   1. What is the vanishing gradient?
   2. As we add more hidden layers, backpropagation becomes less useful in passing information to deeper layers and the gradients become extremely small compared to the weights which prevents the network from learning. We can reduce this impact by using residual networks or ReLU activation function 
   1. LoRA (low rank adaptation)
   2. Reduces the number of parameters that needs to be trained during fine tuning
   3. Original weights are frozen, only low rank updates are learned 
      1. Delta W = A * B
      2. Small low rank decomposition
         1. Linear combinations of set of smaller vectors (dimensionality reduction)
         2. Hyperparameters: alpha, rank
            1. Alpha: scale parameter, how much impact given to pretrained weights
            2. Rank: controls low rank size 
            3. Usually, alpha = r 
   4. Efficient, scalable, reusable


---
# 📘 **Learning of the Day**

   1. What are dropouts?
   2. Dropout is a regularization technique which disables a percentage of neurons at random to prevent overfitting. This means that during training, certain neurons are not used which allows all neurons to learn features and reduce co-adaptation
   1. BERT (Bidirectional encoder representation of transformers)
   2. Designed to generate contextualized word embeddings
      1. Not next word prediction
      2. Masked language model: predict word inside sentence
      3. Next sentence prediction
   3. Fine tuned by adding task specific layers
   4. Bidirectional: captures context on both sides at once (no masking)        
   5. Wordpiece tokenizer 
   6. CLS token (Classification)
      1. Representative summary of the entire input sentence (contextualized embedding)
      2. CLS token can attend to all other tokens in the sentence, making it ideal for gather context


---
# 📘 **Learning of the Day**

   1. What are the components of GAN (Generative Adversarial Network)?
      1. Generator: create synthetic data that cannot be distinguished from real data
      2. Discriminator: determine whether or not data is real
      3. Goal: create synthetic data that cannot be distinguished as fake
   1. Stable Diffusion
      1. text- to-image generative model
      2. Diffusion process: add random noise to images 
         1. Model learns to reverse this process by removing noise
         2. Uses U-Net 
            1. Encoder -> Bottleneck -> Decoder 
            2. Skip connections between encoder, decoder helps preservation of image details 
            3. Predicts noise present in the image and subtract the noise from the input
      3. Text-to-image Translation
         1. Uses a latent diffusion model (LDM)
            1. Lower dimensional latent space (computationally efficient)
         2. Transformer-based text encoder to understand textual input


---
# 📘 **Learning of the Day**

   1. What is the transformer architecture?
   2. Transformer architecture has an encoder and decoder, each with 3 main layers
   3. The first layer is the embedding layer which converts tokens into vectors. Positional embeddings are added to capture the order of the tokens
   4. The second layer is multi-head attention. Attention scores are calculated by comparing queries to keys and it determines how much focus each token should receive from the current token. Multi-head attention allows the model to capture relationships simultaneously. 
   5. The last layer is the feedforward network which applies the non-linear transformations, 
   6. The encoder and decoder communicate through the cross attention mechanism where the encoder can be seen as the ‘comprehension’ module which understands and processes the input and the decoder can be seen as the ‘generation’ module which produces the output 
   7. The main advantages of the transformer architecture compared to RNNs are efficient training since it processes tokens in parallel, the ability to capture long range dependencies, and scalability. 
   1. Broadcasting Rules
      1. Each tensor has at least one dimension
      2. When iterating over the dimension sizes, starting at the trailing dimension, the dimension sizes must either be equal, one of them is 1, or one of them does not exist
   2. If the tensors are broadcastable
      1. Prepend 1 to the one with the smaller dimension and the resulting size will have the larger dimension
   3. Quantization
      1. Reducing the precision of numerical values of the model parameters to decrease the model size and computational requirements. This results in faster inference and lower power consumption 
      2. Uses 8-bit integers rather than 32-bit float


---
# 📘 **Learning of the Day**

   1. How is NSP used in LLMs?
   2. Next sentence prediction is used in the BERT model along with masked language modeling. NSP’s objective is to determine if the sentence logically follows the other previous sentence presented in the model. This allows the model to understand longer-term dependencies across sentences
   3. Explain the concept of Context Window?
   4. Context window is the range or len of tokens the model can consider at once when understanding or generating text. Longer context windows require more computational power but it can better understand lengthy or complex texts. Gpt2 had a context window of 1024 tokens. Gpt 4 has 33,000. 
   1. Byte Pair Encoding (GPT) - Fast and Efficient
      1. Works by iteratively merging frequent pairs of chars or subwords
         1. Initialization: characters + _
         2. Identify patio of symbols next to each other
         3. Merge most frequent pair 
         4. repeat 
      2. Handles out-of-vocabulary (OOV) words
      3. Balances Granularity
      4. Efficient
      5. Language Flexibility
      6. There is a vocab size limit (hyperparameter)
   2. WordPiece (BERT) - more accurate and context sensitive 
      1. Works by merging pairs with based on the maximum likelihood ratio
         1. Merge most likely subword pairs in the training data
         2. Increase the probability of seeing the token in the corpus  
   3. SentencePiece
      1. Works on raw text without preprocessing (white space and punctuation preserved)
      2. Break words into subwords, not characters
      3. Good for NLP, multi language models


---
# 📘 **Learning of the Day**

   1. How do you evaluate the performance of LLMs?
      1. Intrinsic: how well the model captures what it’s supposed to capture  
         1. Perplexity: how much the model is surprised by new data
         2. Lower perplexity, better training 
      2. Extrinsic how useful the model is for a specific task
         1. Text classification
         2. Question Answering (QA)
         3. Summarization
         4. Code Generation 
   2. What are some pre-training objectives of LLMs and how do they work? 
      1. Masked language modeling
         1. Predict correct word within a sentence 
      2. Autoregressive modeling (Casual Language Modeling)
         1. Predicting the next word given a sequence
      3. Seq2Seq, NSP, 
   1. RNN
   2. Allows processing of sequential data by maintaining a hidden state that captures information regarding previous states. Hidden layer has two outputs. One output and one to feed back into the hidden layer. Input to the layer is the concatenation of input from sequence and output from the previous hidden layer. 
   3. It is great at sequential data but not good at long-term dependencies. It is computationally expensive for long sequences and prone to exploding/vanishing gradient problem 
   4. FeedForward vs Attention Blocks 
   5. FeedForward (individual inputs): used for feature transformations and learning representations of individual inputs
      1. MLP (multi layer perceptron) is a subset of FFNs. MLP has at least one hidden layer 
   6. Attention Blocks (multiple inputs): used to weigh the importance of different tokens and capture long range dependencies. Figure which are important and which are related


---
# 📘 **Learning of the Day**

   1. How do generative language models work?
   2. Generative language models work by predicting the next work given an input sequence. Based on a list of possible output words, the model chooses the word with the highest probability. The models are auto regressive which means that the output of the model is feedback in as an input to predict the next word until it reaches an EOS token 
   3. What is fine-tuning and why is it important?
   4. Fine-tuning is taking a pre-trained model created for general and broad range tasks and re-training it for specific tasks such as question answering. New layers and datasets can be added to train the model on a new objective with a loss function reflecting the objective. 
   1. LSTM 
   2. LSTMs try to overcome the issues of short term memory and vanishing/exploding gradients of vanilla RNNs. It is done by keeping track of a long term memory through the cell state. 
   3. There are 3 gates that keep track of the long term memory. The forget gate determines which information should be forgotten from the cell state. The input gate determines which information should be added to the cell state. The output gate (sigmoid) controls what part of the cell state should be used to compute the hidden state and passed onto the next step. 
      1. Forget: controls past memory 
      2. Input: controls new (short term) memory
      3. Output: controls the combination of past and new memory
   4. You can think of this process as a teacher telling the student which information is important to retain and what should be forgotten 
   5. Cell state update
      1. Cell state = Forget gate * prev cell state + input gate * candidate cell state 
      2. Hidden state = output gate * cell state 
   6. Pros:
      1. Handles long-term dependencies
   7. Cons:
      1. Slower and difficult to train
      2. Still not good at very long sequences


---
# 📘 **Learning of the Day**

   1. What is a token in LLM context
   2. Token is the single unit of text that is used as an input to LLMs. Tokens are converted to numerical format and it is used for training the model and for inference. The reason for using tokens is because plain word vocab size is too big and it is not good for generalization of Out Of Vocab words. A common tokenizer is Byte pair encoding which is used for the GPT models 
   3. What are some common hardships regarding LLMs?
   4. It requires a lot of computational resources, understanding and interpreting the decisions made by the LLMs is difficult due to their complex nature 
   1. GRU (Gated Recurrent Units)
   2. A special type of LSTM that is more efficient. It has 2 gates rather than 3
      1. Update gate 
      2. Forget gate
   3. Word2Vec
      1. Word embedding technique to to represent ‘words’ as numerical vectors 
         1. CBOW: predicts target word based on surrounding text 
         2. Skip Grams: predicting surrounding text
   4. Layer norm: by row: across features
   5. Batch norm: by col: by each feature


---
# 📘 **Learning of the Day**

   1. What is the advantage of using Transformers over LSTMs?
   2. Transformers can process the entire text in parallel which is more efficient than the sequential processing of LSTMs. Transformers have better long-term dependencies through self-attention whereas LSTM can suffer for vanishing gradients. Also, it is more memory efficient since it doesn’t have to store the cell state or hidden state
   3. How do LLMs handle OOV words?
   4. Through subword tokenization such as BPE and Wordpiece 
   1. KV Caching 
      1. It is used to speed up auto-regressive text generation by eliminating the need to recalculate the attention scores by keeping a memory of cache of keys and values matrix.
      2. Since tokens are generated one at a time, each time a new token is generated, it only computes the attention for that token and uses the stored K, Vs for the previous tokens


---
# 📘 **Learning of the Day**

   1. Example of alignment problems?
   2. Alignment is matching the goals and behavior of the model with human needs and values. 
      1. Lack of helpfulness
      2. hallucinations 
      3. Lack of interpretability 
      4. Generating biased or toxic output 
   3. What are embedding layers and why are they important?
   4. Embedding layers convert words into numerical vectors and capture semantic meaning of the words. Words with similar meaning have similar vector representations. It is used for capturing the semantics of the word and reducing the dimensionality compared to using one-hot encodings. It can also be used to transfer learning by reusing the pre-trained embeddings across different models.  
   1. Rotary positional encoding (RoPE)
      1. Relative positional embedding + absolute positional embedding 
      2. Method for incorporating positional information without explicit embeddings
      3. Instead of storing separate position embeddings, the model learns positional relationships through rotation operations.
         1. Each (Q, K) pair is rotated by an angle 
         2. It is rotated on a 2 dimensions (total pairs dim/2)
         3. e^(ix) = cos(x) + i*sin(x)
      4. Better long sequence generation and more efficient computation


---
# 📘 **Learning of the Day**

   1. How is adaptive softmax useful?
   2. Adaptive softmax is a technique used to speed up the softmax computation by clustering more frequent words and rarer words. Rather than computing all words in the vocab separately, rarer words are in deeper clusters which reduces computational time and memory usage 
   1. RAG (retrieval augmented generation)
   2. Generating content through external knowledge retrieval mechanism
   3. 1. Query encoding: convert input sentence into embeddings (BERT)
      1. 2 models
         1. Query encoder 
         2. Document encoder 
   4. 2. Retriever searches a vector database based on similarity
      1. HNSW: Hierarchical Navigable Small World
         1. graph-based nearest neighbor search algorithm 
         2. Fast, scalable, works in higher dimensions
   5. 3. Retrieved document are appended to the original query (augmented input)
      1. Kind of like prompt engineering by giving context 
   6. 4. Processes the augmented input for generation


---
# 📘 **Learning of the Day**

   1. How does BERT training work?
      1. BERT is trained through self-supervised learning
      2. two main objectives. Masked Language Modeling and next sentence prediction.
      3. MLM is predicting masked word within a sentence
      4. NSP is predicting rather or not the sentence is the next sentence of the preceding one.
      5. BERT is an encoder only model. 
   2. How do you measure the performance of LLMs?
   3. Metric such as perplexity which measures how well the model predicts the next token. Lower perplexity, better performance.
   4. ROUGE (Recall-Oriented Understudy for Gisting Evaluation) for summarization tasks
   5. BLEU (Bilingual Evaluation Understudy) for text generation tasks 
   6. Exact Match and F1 Score for question and answering
   7. (Massive Multitask Language Understanding) for factual knowledge


---
# 📘 **Learning of the Day**

   1. What Transfer learning techniques can you use for LLMs?
   2. Fine-tuning: using a pre-trained model and training it on a specific task. You can add layers or unfreeze weights 
   3. Adapter layers (PEFT): add small trainable layers while keeping most layers frozen: LORA, prefix-tuning, p-tuning
   4. Retrieval Augmented Generation (RAG)
   5. Feature based transfer learning: using pre-trained model as a feature extractor and training a model on top of the extracted features
   6. Multi-Task Learning (MTL): Train on multiple related tasks at once to improve generalization.
   7. Knowledge distillation


---
# 📘 **Learning of the Day**

   1. What are some techniques for controlling the output of LLMs?
   2. Randomness: temperature, Top K sampling, Top P sampling, BEAM, prompt engineering, control tokens


---
# 📘 **Learning of the Day**

   1. Explain the concept of attention and how it is implemented?
   2. Attention allows the model to focus on different tokens of the input sequence by giving weights to each of the words. This is implemented through calculating attention scores which is computed by the dot product of key and query matrices. Words with similar embeddings have a higher dot product value which in turn would result in higher attention scores.


---
# 📘 **Learning of the Day**

   1. What are some approaches to reduce the computational cost of LLMs?
   2. Compression: Model pruning, distilation, quantization
   3. Architecture: MoE, Spares attention, RAG
   4. PEFT: LoRA, Flash Attention


---
# 📘 **Learning of the Day**

   1. What is the difference between encoder vs. decoder models
   2. Encoder models understand and extract meaning from the text. Models such as BERT are used, no masking takes place and process all inputs at once so it can use self-attention on both past and future tokens. Used for text classification (sentiment) name entity recognition (NER), semantic similarity test. 
   3. Decoder models are used to generate text to predict next tokens. Masking is done for future tokens and inputs are process one token at a time. (auto regressive). It is used for text generation, machine translation, etc.
   4. Encoder-Decoder models (seq2seq) are BART and T5 models. Converts one sequence to another and is used for tasks such as summarization and translation.


---
# 📘 **Learning of the Day**

   1. What’s the difference between wordpiece and BPE
   2. They are both subword tokenization methods but BPE iteratively merges based on most frequent pairs. WordPiece merges based on pairs that are most likely to improve the likelihood of the training data.


---
# 📘 **Learning of the Day**

   1. How do LLMs handle long-term dependencies in text?
   2. Through self-attention mechanism in transformer architecture


---
# 📘 **Learning of the Day**

   1. What is the difference between local and global attention? 
   2. Global attention is where every token in the input sequence can attend to each other whereas local attention is where each token can attend to a window of the input sequence. Global attention is used when full context is critical but it is computationally expensive and not memory efficient


---
# 📘 **Learning of the Day**

   1. Explain the concept of "few-shot learning" in LLMs and its advantages?
   2. Few shot learning means the model can learn and perform on only a few examples. The LLM utilizes the extensive pre-trained knowledge to generalize on smaller instances. Few shot learning reduces the need for additional extensive data and fine-tuning and allows the model to be used for different tasks
   1. Temperature
      1. If less than 1.0 makes it more confident
      2. Divided logits by temperature


---
# 📘 **Learning of the Day**

   1. What is the difference between next token prediction (auto regressive) and masked language model?
   2. Next token prediction focuses on predicting the word that comes next given an input sequence. Mask language model predicts the what the correct word for the masked token should be within a input sequence 
   3. How can you incorporate external knowledge into an LLM?
   4. Knowledge graph integration, Retrieval augmented generation, fine-tuning
   1. 4 patterns of research papers


---
# 📘 **Learning of the Day**

   1. What is the difference between monotonic alignment and predictive alignment in transformers?
   2. They both refer to different approaches for aligning token representations in sequence modeling. Monotonic alignment ensures that the attention mechanism progresses in a strict order and the model only attends to past tokens. Predictive alignment aligns hidden states with expected future states to predict next step outputs more efficiently. It can be thought of as a teacher guiding a student to focus on the correct features.


---
# 📘 **Learning of the Day**

   1. What are some challenges associated with deploying LLMs in production?
   2. Scalability: can handle large volume of requests
   3. Latency: minimizing response time 
   4. Resource management: ensuring cost-efficiency


---
# 📘 **Learning of the Day**

   1. How do you handle model degradation over time in deployed LLMs?
   2. Model degradation happens when the LLM performance declines due to changes in the underlying data distribution. Continuously monitoring the model performance and adding updated data through incremental learning techniques allow the model to learn without losing previous knowledge.


---
# 📘 **Learning of the Day**

   1. Can you explain how techniques like reinforcement learning from human feedback (RLHF) can be used to improve the quality and safety of LLM outputs, and what are some of the challenges associated with this approach?
   2. RLHF is a technique that involves human feedback to align the model output with human preferences. A reward model is created


---
# 📘 **Learning of the Day**

   1. List the key components of LSTM?
   2. Input gate, output gate, forget gate, cell state - memory


---
# 📘 **Learning of the Day**

   1. List the variants of RNNs?
   2. LSTM, Gated recurrent units, end to end network


---
# 📘 **Learning of the Day**

   1. What is Autoencoder, name a few applications? 
   2. Auto encoder is primarily used for unsupervised learning tasks to compress input data into lower dimensional representation known as bottlenecks or latent space. The decoder reconstructs the data using the compressed representation  
   3. Data denoising
   4. Dimensionality reduction
   5. Image reconstruction
   6. Image colorization


---
# 📘 **Learning of the Day**

def backtrack(candidate):
    if find_solution(candidate):
        output(candidate)
        return
    
    # iterate all possible candidates.
    for next_candidate in list_of_candidates:
        if is_valid(next_candidate):
            # try this partial candidate solution
            place(next_candidate)
            # given the candidate, explore further.
            backtrack(next_candidate)
            # backtrack
            remove(next_candidate)
   1. What are the components of GAN?
   2. Discrimator: aims to discriminate data from humans and generated data. 
   3. Generator: aims to create data that is close enough to the human data so it is indistinguishable


---
# 📘 **Learning of the Day**

   1. What's the difference between boosting and bagging?
   2. Boosting and bagging are both ensemble techniques which combine weak learners to create a more accurate and robust learner. Boosting reduces the bias of the model by using all available data and learning on top of the previous model. Bagging reduces variance by combining models created by bootstrapped samples


---
# 📘 **Learning of the Day**

   1. Explain how the ROC curve works?
   2. ROC curve is a graphical representation of the trade-off between true positives and false positives. The closer the curve is to the top left, the better the model. AUC (area under the curve) quantifies the performance


---
# 📘 **Learning of the Day**

   1. What’s the difference between Type I and Type II error?
   2. Type I error is false positives, type II error is false negative. Type 1 error is classifying a condition as true when it is actually false. Type 2 error is not conditioning something as true when it should be conditioned as true 
   1. Gradient accumulation
   2. Is an optimization technique to simulate larger batches in deep learning. Rather than updating model weights after every batch, accumulate gradients over multiple batches before making an update.
   3. It handles the memory constraints of GPUs and stabilizes training


---
# 📘 **Learning of the Day**

   1. What’s the difference between a generative and discriminative model? 
   2. A generative model learns the categories of the data whereas a discriminative model learns the distinction between the categories of the model 
   1. DDP (Distributed Data Parallel)
   2. Parallel computing technique used in pytorch to train across multiple GPUs. It does so by splitting the data and synchronizing the gradients
      1. 1. Split data
      2. 2. Each GPU does forward and backward 
      3. 3. Synchronize gradients by taking average 
      4. 4. Update the weights of the model


---
# 📘 **Learning of the Day**

   1. Instance-Based Versus Model-Based Learning?
   2. Instance-Based: also known as memory based training, the system memorizes the training examples and makes predictions by comparing new inputs to stored instances using similarity metrics. (KNN)
   3. Model-Based: builds a generalizable model from training examples by identifying patterns and learning parameters 
   1. Overfitting issues
      1. Reduce n_embd
      2. Reduce n_layer
      3. Increase dropout (regularization)
      4. Introduce weight decay (l2 regularization)
      5. Reduce vocab size
      6. Decrease learning rate


---
# 📘 **Learning of the Day**

1. Kadane’s algo: similar to sliding windows. Check current and see if the left of current is negative. If it is negative, drop the value. Update global max within each loop
   2. Circular array. Max(max, total - min)
   1. When to use a Label Encoding vs. One Hot Encoding?
   2. Use one-hot encoding when the categorical features are not ordinal like year and the number of features isn’t huge. Vice versa for label encoding


---
# 📘 **Learning of the Day**

   1. What is the difference between LDA (linear discriminant analysis) and PCA (principal component analysis)  for dimensionality reduction?
   2. LDA is supervised whereas PCA is unsupervised. LDA maximizes class separability whereas PCA maximizes variance or information retention 
   1. Group policy


---
# 📘 **Learning of the Day**

   1. What is t-SNE?
   2. T-distributed stochastic neighbor embedding is an unsupervised non-linear learning technique for data exploration and visualizing high dimensional data while preserving the structure and relationship between data points. It is used for visualizing clusters or patterns.


---
# 📘 **Learning of the Day**

   1. What is the difference between t-SNE and PCA for dimensionality reduction?
   2. PCA is a linear method that projects data on a new axis by capturing the maximum variance. t-SNE is a non-linear method that preserves local relationships between points using probability distributions. PCA is good for fast, linear, global structure, t-SNE is good for better, non-linear, local structure


---
# 📘 **Learning of the Day**

   1. What is UMAP?
   2. Uniform Manifold Approximation and Projection is a dimensionality reduction technique that is useful for visualizing high dimensional data while retaining local and global relationships.
   3. First creates a graph that maps similar data points together. It then tries to decrease the dimension while keeping the shape of the initial graph


---
# 📘 **Learning of the Day**

   1. What is the difference between t-SNE and UMAP for dimensionality reduction?
   2. UMAP is better at capturing the global structure whereas t-SNE is better at capturing the local structure. UMAP is graph based whereas t-SNE is probability based. t-SNE is better for high-quality local clustering whereas UMAP is better for scalability and larger datasets.


---
# 📘 **Learning of the Day**

   1. What is the difference between Bayesian vs frequentist statistics?
   2. Bayesian statistics use prior knowledge and beliefs into the analysis whereas frequentists do not


---
# 📘 **Learning of the Day**

   1. How do you handle an unbalanced dataset?
   2. Look at appropriate performance metrics such as precision, recall, F1 score
   3. Oversample the rare samples or undersample to abundant samples via bootstrapping
   4. Secret synthetic examples using methods such as SMOTE (synthetic minority oversampling technique)


---
# 📘 **Learning of the Day**

   1. What are some differences you would see between a model that minimizes squared error vs. an absolute error
   2. MSE gives higher weight to large errors, therefore it is more useful when trying to avoid large errors. MAE is more robust to outliers but requires more complex computation.
   3. MSE is minimized by conditional mean, MAE is minimized by conditional median


---
# 📘 **Learning of the Day**

   1. How do you choose K for K-Means Clustering?
   2. Use the “elbow method.” By looking at the graph between number of K and explained variation, there should be a sharp change in the y-axis for a certain k
   3. Variation explained is the within-cluster sum of squared errors


---
# 📘 **Learning of the Day**

   1. How can you make your model more robust to outliers?
   2. Add regularization (L1, L2)
   3. Use different models (tree based models are usually more robust to outliers)
   4. Winsorize data (cap the data)
   5. Transform data: do a log transformation when response variable is right skewed
   6. Change error metric: from MSE to MAE
   7. Remove outliers


---
# 📘 **Learning of the Day**

   1. Several predictors are correlated during linear regression. What will happen and how do you resolve this issue?
   2. First problem is that the coefficients become unstable and have high variance. The second problem is that the p-value may be misleading leading to uncertainty regarding which features are actually relevant to the model. You can resolve the issue by either combining or removing the correlated predictors with high variance inflation factors. For combining, you can use an interaction term, PCA   
   1. Label Smoothing 
   2. Regularization technique used in classification tasks to improve model generalization and prevent overfitting. Instead of assigning a hard one-hot encoded label, it assigns a softened probabilities to the labels


---
# 📘 **Learning of the Day**

   1. What is the motivation behind random forests? 
   2. It prevents overfitting by allowing more consistent results through bootstrap sampling of data and training on a more diverse dataset
   3. Using only a subset of features at each split helps to de-correlate the decision tree by avoiding using very important features at the top every split.


---
# 📘 **Learning of the Day**

   1. How to handle outliers
   2. 1. clarify the missing data (random, non-random)
   3. 2. Establish a baseline and check if missing data is crucial
   4. 3. Impute missing data (using mean/median/nearest neighbor)
   5. 4. Check performance with missing data


---
# 📘 **Learning of the Day**

   1. Logistic regression output is unsatisfactory, what can I do?
   2. 1. Normalize features so a particular weight do not dominate the model
   3. 2. Address Outliers
   4. 3. Use k-fold cross validation and hyperparameter tuning


---
# 📘 **Learning of the Day**

   1. What is the difference between gradient boosting and random forests?
   2. They are both ensemble methods but have different purposes. Gradient boosting uses boosting which is sequential learning to reduce bias. Each tree is trained on the residual errors of the previous tree. Random Forests uses bagging which is parallel learning to reduce variance. Each tree gets a random subset of data and averages the results of many trees. 
   3. Gradient boosting is slower and needs careful hyperparameter tuning. It is better for high-accuracy models


---
# 📘 **Learning of the Day**

   1. Is 10,000 data points enough to create an accurate model?
   2. Define what is an accurate model, what metrics, type of evaluation metric, type of model etc. 
   3. Create a baseline model and check performance. Set a clear goal for the model, see if the model can achieve its goal
   4. Add more data if necessary
   1. Inverted Residual Block
      1. Expand data, process data, and shrink again 
      2. Smaller Model: Faster and fewer calculations 
   2. Depthwise Convolution        
      1. Each filter applies to only one channel at a time 
      2. Each channel (color) gets its own small filter
   3. Squeeze Excitation 
      1. Focus on the most important parts of the image. 
         1. Squeeze (summarize information) - compress to single number
         2. Excitation (Recalibrate Weights) - learns how much attention each channel should get
         3. Scale (adjust features)
   4. Stochastic depth 
      1. Makes training faster by randomly skipping layers during training


---
# 📘 **Learning of the Day**

   1. Binary classification on a loan model, rejected applicants must be supplied a reason. How would you supply the reasons without digging into the weights of features?
   2. We can look at the partial dependence plot to assess how any one feature affects the model’s decision. Shows the marginal effect of each feature on the predicted target


---
# 📘 **Learning of the Day**

   1. How would you identify synonyms from a large corpus of words?
   2. We can look at word embeddings using algorithms such as word2vec. Words with similar meanings will have similar vector embeddings. We can use k-nearest neighbors to find similar words. However, there are limitations to this model since words that appear in similar contexts that are not necessarily synonyms may show up.
   1. Layer Norm vs. Batch Norm
   2. Batch norm normalizes across the batch dimension, usually used for CNNs
   3. Layer Norm normalizes across the feature dimension, usually used in Transformers


---
# 📘 **Learning of the Day**

   1. Bias variance trade-off
   2. If we decrease bias, variance increases due to overfitting. If we reduce variance through generalization, we lose accuracy. Total error is variance plus bias.


---
# 📘 **Learning of the Day**

   1. K-fold cross validation
      1. Divide data into k batches. Train data with all data except for one k-batch and test model with the remaining k-data. Use validation to reduce test loss and prevent overfitting


---
# 📘 **Learning of the Day**

   1. How would you build a lead scoring algorithm to predict whether a prospective company is likely to convert to a customer?
   2. Ask clarifying questions. Are building our own, or are we building as a product, are there any business requirements, are we running only within our database
   3. Explain the main features used such as firmographic data, marketing activity, sales activity, deals details,
   4. Can create a binary classification model using logistic regression 
   5. Monitor changes in data and update model accordingly


---
# 📘 **Learning of the Day**

   1. How would you create a music recommendation algorithm?
   2. Collaborative filtering 
   3. Find users with similar tastes based on user interaction. Compute similarity using metrics such as cosine similarity or pearson correlation coefficient
   4. Matrix Factorization/ Helps discover latent factor such as type of genre by decomposing a larger matrix into two smaller matrices (user x item)


---
# 📘 **Learning of the Day**

   1. What does it mean for a function to be convex? What algorithms are not convex?
   2. U-shaped function where local minimum is also global minimum. Line segment between any 2 points lies above or on the graph. 2nd derivative >= 0 for all points
   3. Neural networks are not convex. Neural Networks can approximate any function which includes non-convex functions. Use algorithms such as Gradient Descent to find global minimum


---
# 📘 **Learning of the Day**

   1. What is entropy and information gain?
   2. Entropy quantifies uncertainty. High Entropy, values close to 1, means high uncertainty and there’s a 50/50 chance for class distribution. Low entropy, values close to 0, have homogenous classes 
   3. Information gain is how much entropy is reduced for each split in decision trees. It is used to select the best features


---
# 📘 **Learning of the Day**

   1. What is L1 and L2 regularization, what is the difference?
   2. Both prevent overfitting by shifting the coefficients of the model closer to 0. Difference is the penalty applied to the loss function. L1 or lasso regularization uses the absolute value of the penalty term and reduces the coefficient values to zero. This is helpful when choosing the most important features. L2 or ridge regularization uses the squared value of the penalty term and reduces the coefficient values close but not exactly to zero. This is useful when all features are important but need to control the impact


---
# 📘 **Learning of the Day**

   1. What is gradient descent and why use stochastic gradient descent?
   2. Gradient descent is an optimization algorithm to find the global minimum of a loss function by updating the parameters of the model. It uses backpropagation to find the direction of the steepest descent. Stochastic gradient descent updates the model parameters after each individual training sample rather than computing the entire dataset. It converges faster and helps avoid local minimums due to randomness. Mini-batch gradient descent updates parameters each batch such as 32 data points.


---
# 📘 **Learning of the Day**

   1. Classifier that produces a score between 0 and 1, we take the square root of that score, how would the ROC curve change?
   2. ROC curve plots the true positive rates vs false positive rates. If all the values change, then the classification threshold would change and this would lead to the same true positive and false positive rates. If the function is not monotonically increasing such as a negative function or a stepwise function, the ROC will change


---
# 📘 **Learning of the Day**

   1. X is a univariate gaussian random variable, what is the entropy of X?
   2. -p*log(p) 
   3. Use p for the gaussian function, take integral, should be 1


---
# 📘 **Learning of the Day**

2. 2. Learning
   1. Create a model to calculate the propensity of a customer to purchase the product?
   2. Create a dataset with the variable of interest such as purchased/not purchased with related information such as age, gender, income, etc. We can build a logistic regression model for a simple straightforward model. We can also use neural networks to capture more complex relationships between the inputs but it would require a large amount of data to perform well. Another option would be a tree-based model.  
   1. JSON (javascript object orientation)
   2. dictionary/object: key-value pairs
   3. Easy to read and write, language independent


---
# 📘 **Learning of the Day**

   1. Compare and contrast Gaussian Naive Bayes and logistic regression?
   2. Both can be used for classification 
   3. GNB assumes features to be independent but requires a small number of data and is easy to implement
   4. Logistic regression is not flexible and can fail to capture interactions between features but it has a simple interpretation in terms of class probabilities
   5. Logistic regression requires an optimization step and it’s a discriminative classifier whereas GNB is a generative classifier.  
   1. Generative vs. Discriminative classifier 
   2. Generative learns what a spam email and non-spam email looks like 
   3. Discrimination learns the boundary that spits between a spam and non-spam email.
   4. Generative classifiers need less data and faster to train but have strong assumptions such as feature independence. Focus on how the data was generated


---
# 📘 **Learning of the Day**

   1. What loss function is used in k-means clustering? 
   2. Within cluster sum of squares (WCSS) or inertia
   3. Squared euclidean distance between each point in the cluster and the cluster centroid 
   4. Updates clusters and centroid iteratively


---
# 📘 **Learning of the Day**

   1. What is the kernel trick in SVM?
   2. The idea behind it is that data that cannot be separated by a hyperplane in its current dimension can be linearly separable by projecting onto a higher dimensional space. Instead of calculating the new position of the data points in higher dimension, we use the kernel function to compute the dot product to determine how similar they are if they were to be projected to a higher-dimensional space. This allows SVMs to draw complex boundaries


---
# 📘 **Learning of the Day**

   1. Describe how you’d build a model to predict whether a particular use will churn? 
   2. Clarify what churn is and how it is calculated. Talk about modeling considerations such as interpretability. If simple interpretation is required, use logistic regression or decision trees. If not, use SVM or neural networks. Collect features that we will use such as demographics, loss patterns, account balance, etc. See if the model is satisfactory using the ROC curve, F1 scores. Monitor and update results periodically.


---
# 📘 **Learning of the Day**

   1. Describe the model formulation behind logistic regression. How do you maximize the log-likelihood of a given model?
   2. Logistic regression is used for binary classification where the model outputs the probability of being a certain class. Sigmoid function is used for the probability calculation. Minimize the negative log-likelihood function using gradient descent.


---
# 📘 **Learning of the Day**

   1. What is an ML pipeline, and why is it important?
   2. ML pipeline is a structured workflow that automates various steps in machine learning. It is crucial for reproducibility, scalability, automation, and monitoring and maintenance.


---
# 📘 **Learning of the Day**

   1. What are the key stages of the ML pipeline?
   2. Data Ingestion: collecting, processing data
   3. Feature Engineering: Selecting and creating meaningful features
   4. Model training
   5. Model Evaluation 
   6. Model Deployment
   7. Monitoring and logging


---
# 📘 **Learning of the Day**

   1. What is versioning and why is it necessary?
   2. Model versioning keeps track of different experiment runs of a machine learning model. It allows comparison between different models and reproduces the same model in the future allowing rollback.


---
# 📘 **Learning of the Day**

   1. How would you implement a model registry in an ML pipeline?
   2. Storing models with metadata such as parameters and datasets.
   3. Using a centralized repository with automized model registry using CI/CD pipelines
   4. Defining an approval workflow 
   5. Tools: MLFlow, KubeFlow, AWS Sagemaker 
   1. CI/CD (continuous integration, continuous delivery) 
   2. CI: every time code is pushed on github, it is automatically built and tested (ensures main branch is always on working state) 
   3. CD: automatically prepares code for release (build -> package -> release)


---
# 📘 **Learning of the Day**

   1. How do you monitor a deployed ML model?
   2. Track performance metrics, Detect data drifts, set alerts for model anomalies


---
# 📘 **Learning of the Day**

   1. What types of testing are necessary for an ML pipeline?
   2. Unit Testing: verify each individual function works correctly
   3. Integration Testing: Ensure interaction between pipelines
   4. Regression Testing: verify updating the model doesn’t degrade performance
   5. Performance testing: evaluation inference speed and scalability


---
# 📘 **Learning of the Day**

   1. How does CI/CD work in ML pipelines?
   2. Automating model training and validation, running performance checks before deployment, rolling back to previous models 
   3. Github Actions, MLFlow, KubeFlow
   1. Kubernetes
   2. Open source container orchestration platform that helps automate deploying, scaling and managing containerized applications (docker) across clusters of machines. Manages containers at scale and auto-restarts dead containers. Handle traffic spikes (load balancing) 
   3. ALiBi (relative positional encoding)
   4. Attention with Linear Biases. Instead of adding position vectors, add a bias to the attention score so that model prefers to attend to recent tokens more than far ones. Faster to compute, no need for positional embeddings, better at longer sequences.


---
# 📘 **Learning of the Day**

   1.  What is a canary deployment?
   2. Releasing a new model to a subset of users to minimize risk and allow real world monitoring with the possibility of a roll back if performance declines
   1. Mixed-Precision Training 
   2. Using half precision floats (FP16) instead of full precision floats (FP32). Trade off between numerical stability and memory footprint 
   3. Cannot represent floats smaller than 2^-24 and underflows to 0 and no changes to weight values. 
   4. Using FP16 activates tensor cores
   5. Solution
      1. Loss scaling
         1.  Multiply the gradients to a constant scale factor and shifting the gradients to a value greater than 2 ^-24 
         2. Can apply to loss rather than gradients to achieve same output due to chain rule and back prop 
      2. FP32 Master Copy of weights 
         1. Use FP16 for forward and backward pass, FP32 for the optimizer step 
         2. Model memory increases but activation and gradient memory decrease (bigger portion)
      3. Mixed Precision Arithmetic 
         1. Can do mixed precision thanks to cuda. Read and write in FP16 but compute type of FP32


---
# 📘 **Learning of the Day**

   1. How do you reproduce a ML pipeline?
   2. Versioning code and data, setting random seeds, using containerization
   3. What are the biggest challenges in deploying ML models?
   4. Scalability, latency, model drift, managing compute costs, security and compliance for protecting sensitive data and meeting regulations 
   1. LIMA (Less is more alignment)  
   2. Superficial alignment hypothesis
   3. A model’s knowledge is almost entirely learned during pretraining 
      1. Alignment -> what part of the model to use 
      2. Short fine tuning can ruin less of the pretrained knowledge and avoid catastrophic forgetting  
   4. small finetuning dataset can be better


---
# 📘 **Learning of the Day**

   1. What are the best practices for scaling ML models?
      1. Batch inference: processing data in groups rather than real time
      2. microservices architecture: deploying models as independent services 
      3. Model caching: storing frequent predictions for quick retrieval 
   2. What strategies can be used to optimize ML pipeline?
      1. Distributed data processing: (Apache Spark, Hadoop) break up large data tasks into multiple machines
      2. Feature store integration: implement a centralized feature store to prevent redundant calculations 
      3. Parallel Processing: train models in parallel using GPUs, Cloud
      4. Auto-scaling infrastructure: deploy models in kubernetes to leverage auto scaling


---
# 📘 **Learning of the Day**

   1. How do you handle long-running ML training jobs efficiently?
   2. Checkpointing, using cloud-based spot instances, gradient accumulation: accumulate gradients over multiple mini-batches, Data-pipeline optimization


---
# 📘 **Learning of the Day**

   1. How is RAG different from traditional LLMs?
   2. Unlike traditional models which rely on pre-trained knowledge, RAG can access external knowledge sources through the retrieval step


---
# 📘 **Learning of the Day**

   1. What is the purpose of the max token parameter?
   2. Sets an upper limit of how many tokens the model can produce at once. Helps control model API costs


---
# 📘 **Learning of the Day**

   1. What is empirical
   2. Based on observation or experience rather than theory or pure logic


---
# 📘 **Learning of the Day**

   1. What is top P and how does it differ from top-k?
   2. Top-p samples the output until the cumulative probability exceeds p whereas top-k samples the fixed k-number of tokens. Top-p is more fluent top-k for more consistency


---
# 📘 **Learning of the Day**

   1. What is the role of the frequency penalty parameter?
   2. Frequency parameter penalizes the model from repeating the same tokens too often. It improves lexical variety and reduces redundancy. Similar to presence penalty parameter which penalizes tokens that have already appeared regardless of frequency


---
# 📘 **Learning of the Day**

   1. How do stop sequences work, why are they important?
   2. Stop sequences are specific tokens or sentences that tell the model to stop generating. They are critical for enforcing the response structure.


---
# 📘 **Learning of the Day**

   1. How do temperature and top_p work together?
   2. You divided the output logits by temperature parameter 
   3. Lower temperature with moderate p -> coherent and focused 
   4. High temperature and high top p -> more creative output


---
# 📘 **Learning of the Day**



---
# 📘 **Learning of the Day**

   1. Explain how quantization improves LLM efficiency. What are the trade-offs?
   2. Faster inference and lower memory by using int8 precision rather than float32
   3. Slight accuracy drop
   4. Uses ONNX Runtime 
   1. ONNX Runtime 
   2. Open Neural Network Exchange - high performance inference engine


---
# 📘 **Learning of the Day**

   1. What is an inference engine?
   2. Inference engine takes a trained model to make predictions based on inputs. They are optimized for speed, efficiency, and hardware acceleration


---
# 📘 **Learning of the Day**

   1. What strategies can be used to detect model drift? 
   2. Monitor performance metrics such as accuracy, precision 
   3. Use statistical tests such as Kolmogorov-Smirnov test to check divergence between data distributions 
      1. Build empirical cumulative distribution function and compare vertical distance


---
# 📘 **Learning of the Day**

   1. How do you ensure security in the ML pipeline?
   2. Adversarial testing - test models against adversarial inputs to detect vulnerabilities 
   1. WANDA (Weights and activation pruning with norms and discrepancies Awareness)
   2. Magnitude based structured pruning method. It uses weight importance derived from activation statistics. It measures the importance of weights by combining both the magnitude of the weights and the activation patterns. See how big the weights are and see which weights are activated when an input is passed in 
   3. It is a post training pruning method and requires minimal fine-tuning


---
# 📘 **Learning of the Day**

   1. How does the temperature parameter impact model output?
   2. Temperature controls the randomness of the model output. Low temperatures are more deterministic whereas high temperatures are more creative 
   1. LongNet (Dilated Transformers) 1 Billion token context window 
   2. Instead of looking at all the tokens, it skips over 2, 4, 8 tokens. Different heads look at different parts of the input token. Some look at close words, some look at far away words


---
# 📘 **Learning of the Day**

1. For backtracking, to prevent duplicates, range over (index, len(candidates))
   2. append(candidate[i])
   1. How would you debug an LLM that produces overly verbose responses in a chatbot?
   2. Reduce temperature, include frequency penalty parameter, add stop sequences 


---
# 📘 **Learning of the Day**

   1. Deterministic vs stochastic?
   2. Deterministic produces the same output given the same input
   3. Stochastic includes randomness so the same input can lead to different outputs


---
# 📘 **Learning of the Day**

   2. How would you optimize LLM inference for low latency and high throughput?
   3. Quantization, KV caching, caching frequent prompts, using distilled models, batching to process multiple requests in parallel, use triton, vLLM


---
# 📘 **Learning of the Day**

   1. Describe key non-functional requirements (NFRs) for an LLM system.
   2. Latency < 1sec
   3. Cost: optimize model size 
   4. Scalability: Auto-scaling tools such as kubernetes
   5. Security: Data Encryption 
   6. Reliability: fallback models


---
# 📘 **Learning of the Day**

   1. Walk through designing a customer support chatbot (real-world case study)?
   2. Ask what the requirements are.
   3. Frontend: use react based UI
   4. Backend: use Fast API, host model on cloud service such as AWS, use RAG pipeline to fetch order details. (pinecone - vector DB). use langchain to manage context (memory, history documents) 
   5. Optimization. Use quantized or distilled models. Cache frequently asked prompts


---
# 📘 **Learning of the Day**

   1. How would you design a system to handle rate-limiting for LLM API requests? 
   2. Token bucket algorithm: track requests per user IP
   3. Fallback: serve cached responses or lighter model
   4. API gateway: enforce limits via NGINX
   1. NGINX 
      1. Middle man that helps traffic go to the right place
      2. Receives requests from users (port 80)
      3. Pass those requests to fastAPI
      4. Returns the response to the user


---
# 📘 **Learning of the Day**

   1. Design a multi-agent LLM system where different models handle different tasks 
   2. Router model: classifies queries to appropriate model
   3. Orchestrator: langchain 
   4. Data processing: JSON


---
# 📘 **Learning of the Day**

   1. How would you reduce GPU costs for a high-traffic LLM app?
   2. Model Distillation, quantization. Using the Mixture of Experts model to only activate relevant layers. Cache frequent responses 
   3. Concept
   1. Serverless GPU 
   2. You just send a job such as inference, the cloud provider starts a GPU, runs the job, and shut it down. You only pay for what you use
   3. Cost efficient, on-demand usage, and auto-scaling 
   4. AWS lambda server


---
# 📘 **Learning of the Day**

   1. Design Chat with PDF?
   2. 1. Upload pdf
   3. 2. Chunk and embed. Divide pdf into paragraphs and vectorize it. Store it to pinecone vector DB 
   4. 3. Query. User gives prompt and RAG fetches top chunks using FAISS (facebook ai similarity search) 
   5. 4. LLM summarizes and gives output 
   3. Concept
   1. Async processing 
   2. Way for programs to handle multiple tasks at the same time without waiting for each one to finish before starting the next.


---
# 📘 **Learning of the Day**

   1. Scale an LLM app from 100 to 1M users?
   2. Ask what does the app do, are responses real-time, what model is used
   3. Possible issues are 
   1. Compute bottleneck
   2. Delayed inference latency
   3. Memory and storage 
   4. Cost management 
   4. For optimization 
   1. Use distilled models, quantization, deploy models on vLLM
   5. Infrastructure
   1. Use kubernetes for horizontal scaling 
   2. Set up load balancers
   6. Storage
   1. Store embeddings in FAISS or pinecone
   7. Rate limiting 
   1. Add rate limiting to nginx or api gate-way
   8. Cost
   1. Use spot instances or serverless GPUs
   3. Concept
   1. Orchestration platform
   2. System that automates management, coordination, and scaling of multiple services or components.
   1. Web request handling: spawns more instances 
   2. Prompt processing: sends prompts to right worker  
   3. Text-to-speech job: runs async TTS jobs
   4. Logging and analytics: collects logs and usage data


---
# 📘 **Learning of the Day**

   1. How does vLLM work?
   2. vLLM is a high throughput memory efficient inference engine. 
   3. Uses paged-attention which dynamically allocates memory for kv-caching (only uses memory needed rather than a fixed size)
   4. Supports dynamic batching: adds new user prompts on the fly rather than fixed schedules. They can also have varying lengths 
   5. Supports token streaming: tokens are returned as they are generated


---
# 📘 **Learning of the Day**

   1. What are two primary types of LLM evaluation?
   1. Reference based metrics: compare generated output against BLEU, ROUGE, METEOR, etc
   2. Reference free metrics: tone, coherence, structure
   3. Concept 
   1. N-grams
   1. Comparing two pieces of text by looking at how many small word groups they have in common. It measures how similar texts are.
   2. BLEU score


---
# 📘 **Learning of the Day**

1. Learning
   1. How does BERTScore work?
   2. Bert score calculates the concise similarity between token embeddings.


---
# 📘 **Learning of the Day**

1. What are reference-free metrics, and when are they useful?
   1. Evaluates output without needing reference text making it ideal to evaluate tone, open-ended generation
   2. Regex-based checks, Deterministic validation: check if output is in correct format


---
# 📘 **Learning of the Day**

1. Learning
   1. How are Contextual Precision and Recall applied in RAG systems?
   2. Contextual Precision: measures how relevant the retrieved documents are
   3. Contextual Precision: Assess rather enough information was retrieved


---
# 📘 **Learning of the Day**

1. Learning
   1. Which metric detects hallucinations effectively?
   2. HallucinationMetric flags content that deviates from the provided context or source.


---
# 📘 **Learning of the Day**

1. Learning
   1. Why is BLEU limited, and when does it fall short?
   2. BLEU (Bilingual Evaluation Understudy) focus on n-gram matches missing semantic nuances and struggles with paraphrased or creative output


---
# 📘 **Learning of the Day**

1. Learning
   1. Which Python libraries support BLEU, ROUGE, and METEOR?
   2. Huggingface evaluate library


---
# 📘 **Learning of the Day**

1. Learning 
   1. What are the drawbacks of embedding-based metrics?
   2. Ignore syntax, overlook factual accuracy, inflate similarity from shared vocabulary


---
# 📘 **Learning of the Day**

   1. What are Agentic Design Patterns?
   2. Frameworks that enable to agents autonomously reason, act, self-improve using memory, tools, reflection, and multi-agent frameworks


---
# 📘 **Learning of the Day**

   1. Name 3 common agentic patterns
   2. ReAct (reasoning and acting), CodeAct (dynamic code and generation), agentic RAG (rag with with memory and tools)


---
# 📘 **Learning of the Day**

   1. What is the role of the ReAct agent?
   2. It enables the agent to alternate between reasoning (LLM-based) and performing actions (external tools/APIs) dynamic and context-aware decision making


---
# 📘 **Learning of the Day**

   1. How does the Model Context Protocol (MCP) support Modern Tool Use?
   2. MCP defines how context is represented and passed to a model. MCP uses structured tool_calls in messages which enables automatic routing to the correct tool. 
   3. Tells the model which tools to use in, structured message, and keeps memory and tools organized. Separates thinking and doing.


---
# 📘 **Learning of the Day**

   1. What purpose does the Self-Reflection pattern serve?
   2. It enables agents to critique their own output, look for errors, iteratively refine answers
   3. How does a Multi-Agent Workflow improve efficiency?
   4. By distributing tasks among specific agents, enabling parallelism


---
# 📘 **Learning of the Day**

   1. How would you design a customer support agent for a financial institution using Agentic Design Patterns?
   2. I would use a ReAct agent to handle customer queries and perform actions. Add tools with API integrations, Self reflect on answers and Agentic RAG to retrieve policy documents and past conversations


---
# 📘 **Learning of the Day**

1. Why is memory (checkpointer) critical in agentic design? 
   2. It enables long-term coherence, recovery from interruptions, records intermediate states so it doesn’t forget the previous context, it also stores which tools are called, what the output of the tool call was and what to do next


---
# 📘 **Learning of the Day**

1. Preventing infinite reflection loops?
   2. Use critique thresholds, capped iteration loops.

   1. CORS (Cross-Origin Resource Sharing)
   2. Security feature built into web browsers that controls which websites can make requests. Protects from malicious sites


---
# 📘 **Learning of the Day**

1. SSL/TLS certificates?
   2. Certbot is a free, open-source tool that helps you automatically get and renew SSL/TLS certificates from Let's Encrypt, so your website can use HTTPS.
   3. SSL/TLS certificates are digital certificates that provide secure, encrypted communication between a user's browser and a website server.
   1. SSL (Secure Sockets Layer) - older
   2. TLS (Transport Layer Security) - modern
---
