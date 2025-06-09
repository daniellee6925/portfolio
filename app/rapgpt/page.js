'use client';

import React from 'react';
import Navbar from '../components/Navbar'; 
import Footer from '../components/Footer'; 



const RapGPTDocs = () => {
  return (
    <div>
      <Navbar />
    <main className="max-w-6xl mx-auto px-6 py-12 text-gray-800 pt-[7%]">
      <h1 className="text-4xl font-bold text-red-800 mb-8">RapGPT: Rap Lyric Generator</h1>

      <Section title="Overview">
        <p className="mb-4">
          RapGPT is a pre-trained & fine-tuned GPT-2 model designed to generate original rap lyrics similar to Eminem$&apos;s Lyric style based on user prompts.
          It is trained on a decoder-only transformer architecture and built on a slightly modified GPT-2 configuration.
          The project was iteratively optimized to run on CPU-only environments due to resource constraints, making it accessible for lightweight deployment.
        </p>
        <p>
          This project was started as a hands-on initiative to gain end-to-end experience in large language model (LLM) development, covering data collection, pre-training, parameter-efficient fine-tuning (PEFT), evaluation, and deployment. While model performance is constrained by limited compute resources, the focus was on building practical skills across the full LLM pipeline.
        </p>

        <li className='mt-4'>
          <strong>GitHub Repository:</strong>{' '}
          <a
            href="https://github.com/daniellee6925/rapGPT2.0"
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 underline"
          >
            github.com/daniellee6925/rapGPT2.0
          </a>
        </li>
        <li>
          <strong>Live Site:</strong>{' '}
          <a
            href="https://www.eminemgpt.com"
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 underline"
          >
            www.eminemgpt.com
          </a>
        </li>
      </Section>

      <Section title="Data Preparation">
        <ul className="list-disc list-inside">
          <li><strong>Source:</strong> <a href="https://genius.com/" className="text-blue-600 underline" target="_blank">Genius.com</a></li>
          <li><strong>Cleaning:</strong> Removed metadata, repetition tags, and extra whitespaces.</li>
          <li><strong>Tokenization:</strong> GPT-2&apos;s <code>tiktoken</code>-based Byte Pair Encoding (BPE).</li>
        </ul>
      </Section>

      <section className="mb-6">
        <h2 className="text-xl font-semibold mb-2">Dataset Summary</h2>
        <ul className="list-disc list-inside text-gray-700 space-y-1">
          <li><strong>Total Words:</strong> 23,009,521</li>
          <li><strong>Total Songs:</strong> 17,273</li>
          <li><strong>Total Artists:</strong> 226</li>
          <li><strong>Total Tokens:</strong> 25,367,203</li>
        </ul>
      </section>

      <Section title = "Pretraining" className="my-8">
      <p className="mb-4">
        The model was pretrained using a <strong>decoder-only Transformer architecture</strong> based on <code className="bg-gray-100 px-1 py-0.5 rounded">GPT-2</code>, with modifications to reduce size and optimize for <code className="bg-gray-100 px-1 py-0.5 rounded">CPU-based</code> environments.
      </p>
      <section className="space-y-4">
          <h2 className="text-2xl font-semibold text-gray-800">Training Optimization Techniques</h2>
      <ul className="list-disc list-inside space-y-4">
        <li>
          <strong>Model Selection:</strong> While larger parameter <code className="bg-gray-100 px-1 py-0.5 rounded">(774M)</code> models were trained, the base GPT-2 variant <code className="bg-gray-100 px-1 py-0.5 rounded">(124M)</code> was chosen to strike a balance between performance and computational efficiency, enabling inference on a CPU-only instance.
        </li>
        <li>
          <strong>Tokenization:</strong> Used <code>tiktoken</code> for its faster encoding <code className="bg-gray-100 px-1 py-0.5 rounded">(2x-5x)</code> compared to equivalent huggingface encoding and lower memory usage.
        </li>
        <li>
          <strong>Mixed Precision Training:</strong> Mixed precision training using PyTorch’s autocast context manager was implemented. This allowed for faster training steps and reduced memory usage, enabling larger batch sizes and more efficient training. <code className="bg-gray-100 px-1 py-0.5 rounded">bfloat16</code> was used instead of <code className="bg-gray-100 px-1 py-0.5 rounded">float16</code> to enable mixed-precision training and inference on a CPU instance, as <code className="bg-gray-100 px-1 py-0.5 rounded">bfloat16</code>maintains the same exponent range as <code className="bg-gray-100 px-1 py-0.5 rounded">bfloat32</code>. Since <code className="bg-gray-100 px-1 py-0.5 rounded">bfloat16</code> support 8 exponential bits, scaling (GradScaler) was not required to prevent uverflow.
        </li>
        <li>
          <strong>Gradient Accumulation:</strong> Enabled to effectively simulate larger batch sizes and stabilize training without requiring additional memory overhead.
        </li>
        <li>
          <strong>Distributed Data Parallel:</strong> While Distributed Data Parallel <code className="bg-gray-100 px-1 py-0.5 rounded">(DDP)</code> was  implemented to enable scalable multi-GPU training, it was ultimately not utilized due to the lack of access to multi-GPU instances during development. 
        </li>
        </ul>
        </section>


        <section className="space-y-4 mt-4">
          <h2 className="text-2xl font-semibold text-gray-800">Training Stability Techniques</h2>
        <ul className="list-disc list-inside space-y-2">
          <li>
            <strong>Weight Initialization:</strong> Model weights were initialized using standard GPT-2 initialization-normal distribution with a standard deviation of <code>0.02</code>, promoting convergence stability early in training.
          </li>
          <li>
            <strong>Gradient Clipping:</strong> Used to prevent exploding gradients by capping gradient norms during backpropagation.
          </li>
          <li>
            <strong>Learning Rate Scheduling:</strong> <em>cosine decay</em> scheduler was used to gradually reduce the learning rate.
          </li>
        </ul>
        </section>

        <section className="space-y-4 mt-4">
          <h2 className="text-2xl font-semibold text-gray-800">Training Process</h2>
        <li>
          A custom PyTorch loop loads batches, computes forward and backward passes, and evaluates loss every 500 steps using a held-out validation split.
        </li>

         <li>
          <strong>Hyperparameters:</strong>
          <ul className="list-disc list-inside ml-6">
            <li>Batch Size: 8 (per step)</li>
            <li>Block Size (context length): 1024 tokens</li>
            <li>Learning Rate: 6e-4 (AdamW optimizer)</li>
            <li>Training Steps: 4 epochs</li>
            <li>Hardware: AWS EC2 g4dn.xlarge & RTX 4080</li>
          </ul>
        </li>
        </section>
      <p className="mt-4">
        Overall, the pretraining process reflects practical design choices made to maximize learning outcomes under limited hardware while keeping the model architecture extensible for future improvements.
      </p>
    </Section>

      <Section title="Fine-Tuning (LoRA)">
         <p className="mb-4">
          To reduce computational cost and memory usage during fine-tuning, this project uses a custom{' '}
          <strong>Low-Rank Adaptation (LoRA)</strong> implementation applied to both{' '}
          <strong>linear</strong> and <strong>embedding</strong> layers. 
        </p>

        <p className="mb-4">
          <strong>Fine-Tuning Dataset:</strong> The model was fine-tuned exclusively on a curated set of Eminem&apos;s lyrics, consisting of approximately <code>213,680</code> tokens. This dataset was used to guide the model toward generating lyrics in a style consistent with the artist.
        </p>
        <p className="mb-4">
          <strong>Training Configuration:</strong> The fine-tuning loop followed similar architecture and hyperparameter conventions as the main pretraining process, including the use of gradient accumulation, cosine learning rate decay, and mixed precision training, ensuring a consistent and efficient optimization pipeline.
        </p>

        <ul className="list-disc list-inside mb-4">
          <li><strong>Target Modules:</strong> Attention + MLP layers</li>
          <li><strong>Excluded Modules:</strong> Final LayerNorm</li>
          <li><strong>Rank:</strong> 8</li>
          <li><strong>Alpha:</strong> 8</li>
          <li><strong>RSLoRA used:</strong> stabilized fine-tuning with rank based scaling factor. Ideal for small datasets and robust at low-rank settings</li>
        </ul>
        <p className="mb-4">
        <strong>LoRA Rank/Alpha Selection:</strong> I chose a relatively low LoRA rank/alpha value of <code>8</code> to suit the characteristics of this project. Since the fine-tuning task is not highly complex and is performed on a small dataset within a low-resource environment , a lower alpha helps maintain training stability without overwhelming the base model.
        </p>
        <p className="text-gray-700 mb-4">
          <strong>Training Efficiency:</strong> Only <strong>1.06%</strong> of the total model parameters were updated during fine-tuning,
          significantly reducing the computational load while preserving performance.
        </p>
      </Section>

      <Section title="Optimization">
        <h3 className="text-xl font-semibold mt-4">KV Caching</h3>
        <p>
          Key-Value caching improves generation speed by storing previously computed attention values,
          avoiding recomputation for earlier tokens during inference.
        </p>
        <p className='mt-4'>
        <strong>KV-Caching Memory Usage:</strong> For a max context length of 1024 tokens and using <code>bfloat16</code>, the model consumes approximately <strong>16 MB</strong> of memory for caching key/value tensors during inference.
        </p>
        <h2 className="text-xl font-semibold mt-4">KV-Cache Memory Usage Estimation</h2>
          <ul className="space-y-1 text-sm text-gray-700 mt-3">
            <li><strong>Batch Size:</strong>1 </li>
            <li><strong>Layers:</strong>8 </li>
            <li><strong>Heads:</strong>8 </li>
            <li><strong>Context Length:</strong>1024</li>
            <li><strong>Head Dim:</strong>64</li>
            <li><strong>Dtype:</strong> bfloat16</li>
            <li><strong>Memory Usage=</strong><code className="bg-gray-100 px-1 py-0.5 rounded">1 × 8 × 8 × 1024 × 64 × 2 × 2 bytes = 16 MB</code></li>
          </ul>
        <h3 className="text-xl font-semibold mt-4">Quantization</h3>
        <p className="mb-4">
          To reduce model size and improve inference speed, <strong>dynamic quantization</strong> was applied using PyTorch’s built-in API. This optimization allows the model to run efficiently on CPU without retraining.
        </p>
        <p className="mb-4">
          Although <strong>int8</strong> quantization is typically preferred for maximum compression, it significantly 
          degraded the model&apos;s output quality in this case. As a result, <strong><code>float16</code></strong> was used instead, 
          which preserved model performance while still achieving an estimated <strong>20% reduction in inference time</strong>.
        </p>
        <ul className="list-disc list-inside space-y-2">
          <li><strong>Method:</strong> <code>torch.quantization.quantize_dynamic</code></li>
          <li><strong>Scope:</strong> Applied to <code>nn.Linear</code> layers</li>
          <li><strong>Precision:</strong> <code>float16</code> (half precision)</li>
          <li><strong>Goal:</strong> Decrease memory footprint and enable faster CPU inference</li>
          <li><strong>Result:</strong> Achieved lower latency and smaller model size for deployment on a CPU-only instance</li>
        </ul>
        <h2 className="text-xl font-semibold mt-4 mb-4">Dynamic Quantization Summary</h2>
          <p className="mb-2">
            Dynamic quantization reduces model size and improves inference speed by converting certain weights—
             in <code>Linear</code> layers—from 32-bit floats to lower precision types like <code>int8</code> or 
            <code> float16</code> at runtime.
          </p>
          <p className="mb-2">
            It uses <strong>asymmetric quantization</strong>, where the scale (α) and zero-point (β) are
            <strong> automatically calculated</strong> based on the <strong>distribution of model weights</strong>.
          </p>
      </Section>

      <Section title="Evaluation">
        <p className="mb-4">
          To assess the quality and stylistic relevance of generated rap lyrics, a combination of a
          <strong> DistilBERT-based classifier</strong> and <strong>cosine similarity metrics</strong> was used.
        </p>

        <p className="mb-4 text-gray-700">
          <strong>Performance Impact:</strong> The average cosine similarity score before fine-tuning was approximately <strong>0.34</strong>,
          indicating limited stylistic alignment. After fine-tuning on Eminem&apos;s lyrics, the average score increased to around <strong>0.71</strong>,
          resulting in a significant improvement in stylistic relevance and generation quality.
        </p>

        <h3 className="text-xl font-semibold mb-2">Evaluation Method</h3>
        <ul className="list-disc pl-6 mb-4 space-y-2">
          <li>
            <strong>Embedding Generation:</strong> Each generated lyric is converted into a semantic vector using a
            pre-trained <code>distilbert-base-uncased</code> model.
          </li>
          <li>
            <strong>Reference Comparison:</strong> The generated vectors are compared against embeddings of
            real rap lyrics from the training set or manually curated references.
          </li>
          <li>
            <strong>Cosine Similarity:</strong> Calculates similarity between vectors—higher scores indicate
            better alignment with real rap lyric style.
          </li>
          <li>
            <strong>Thresholding / Labeling (Optional):</strong> A fine-tuned classifier can assign quality labels (e.g., 
            authentic-style vs off-style) or confidence scores to each generation.
          </li>
        </ul>

        <h3 className="text-xl font-semibold mb-2">Example Results</h3>
        <div className="overflow-x-auto rounded-lg border border-gray-300">
          <table className="min-w-full text-sm text-left table-auto">
            <thead className="bg-gray-100">
              <tr>
                <th className="py-2 px-4 border-b">Prompt</th>
                <th className="py-2 px-4 border-b">Cosine Similarity</th>
                <th className="py-2 px-4 border-b">Comment</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-t">
                <td className="py-2 px-4">Chasin&apos; dreams in the rain</td>
                <td className="py-2 px-4">0.82</td>
                <td className="py-2 px-4">Strong stylistic match</td>
              </tr>
              <tr className="border-t">
                <td className="py-2 px-4">I got Loyalty, got royalty inside my DNA</td>
                <td className="py-2 px-4">0.48</td>
                <td className="py-2 px-4">Off-style / Kendrik Lamar</td>
              </tr>
              <tr className="border-t">
                <td className="py-2 px-4">It feels so empty without me</td>
                <td className="py-2 px-4">0.92</td>
                <td className="py-2 px-4">Excellent stylistic match</td>
              </tr>
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="Deployment">
      <p className="text-gray-700 mb-4">
        <strong>rapGPT</strong> is deployed as a full-stack application, consisting of a FastAPI backend and a React/Next.js frontend.
      </p>
      <ul className="list-disc list-inside text-gray-700 mb-4">
        <li>
          <strong>Backend:</strong> Powered by <code className="bg-gray-100 px-1 rounded">FastAPI</code>, hosted on an <code className="bg-gray-100 px-1 rounded">AWS EC2 c5.large</code> instance (CPU-only).
        </li>
        <li>
          <strong>Frontend:</strong> Built with <code className="bg-gray-100 px-1 rounded">React</code> and <code className="bg-gray-100 px-1 rounded">Next.js</code>, deployed via <a href="https://vercel.com" target="_blank" rel="noopener noreferrer" className="text-blue-600 underline">Vercel</a>
        </li>
      </ul>
    </Section>
    </main>
    <Footer/>
    </div>
  );
}

function Section({ title, children }) {
  return (
    <section className="mb-12">
      <h2 className="text-2xl font-bold border-b border-gray-300 pb-2 mb-4">{title}</h2>
      {children}
    </section>
  );
}


export default RapGPTDocs;
