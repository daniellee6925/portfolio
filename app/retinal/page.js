'use client';

import React from 'react';
import Navbar from '../components/Navbar'; 
import Footer from '../components/Footer'; 



const RetinalDocs = () => {
  return (
    <div>
      <Navbar />
    <main className="max-w-6xl mx-auto px-6 py-12 text-gray-800 pt-[7%]">
      <h1 className="text-4xl font-bold text-red-800 mb-8">Retinal Disease Classifier</h1>

      <Section title="Overview">
        <p className="text-gray-700 mb-4">
          The Retinal Disease Classifier is a computer vision model designed to assist in the detection and classification of retinal diseases using fundus images.
          It performs <strong>hierarchical classification</strong>—first detecting the presence of disease, then identifying the specific condition. The currently deployed model is a fine-tuned <strong>EfficientNet-B3</strong> trained on ~1,600 retinal images.
        </p>
        <p className="text-gray-700 mb-4">
          Try the final product here:{" "}
          <a
            href="https://huggingface.co/spaces/daniellee6925/Retinal_Disease"
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 underline hover:text-blue-800"
          >
            Retinal Disease Classifier Demo
          </a>{" "}
          <em>(may be sleeping due to inactivity)</em>
        </p>

        <h3 className="text-lg font-semibold text-gray-800 mb-1">Model Metrics:</h3>
        <ul className="text-gray-700 list-disc list-inside space-y-1 mb-4">
          <li><strong>Disease Identification Recall:</strong> 85%</li>
          <li><strong>Disease Classification Accuracy:</strong> 82%</li>
          <li><strong>Inference Speed:</strong> ~0.5 seconds per image</li>
        </ul>

        <h3 className="text-lg font-semibold text-gray-800 mb-1">Classification Types:</h3>
        <ul className="text-gray-700 list-disc list-inside space-y-1 mb-4">
          <li><strong>Normal Retina</strong></li>
          <li><strong>Diabetic Retinopathy:</strong> Damages the retina’s blood vessels and may lead to vision loss.</li>
          <li><strong>Age-Related Macular Degeneration (ARMD):</strong> Blurs central vision due to damage to the macula, common in older adults.</li>
          <li><strong>Media Haze:</strong> Clouding of the eye’s optical media, reducing vision clarity.</li>
          <li><strong>Optic Disc Cupping:</strong> Structural change in the optic nerve head, often related to glaucoma.</li>
        </ul>

      
        </Section>

        <Section title="Features">
        <ul className="text-gray-700 list-disc list-inside space-y-1">
          <li>Hierarchical classification pipeline:
            <ul className="list-disc list-inside pl-4">
              <li><strong>Step 1</strong>: Disease presence detection</li>
              <li><strong>Step 2</strong>: Disease type classification</li>
            </ul>
          </li>
          <li>Custom vision models: EfficientNet-B3 and Vision Transformer (ViT) implemented from scratch</li>
          <li>Lightweight model optimized for low-latency inference</li>
          <li>Training monitored via TensorBoard</li>
          <li>Deployed on Hugging Face Spaces using Gradio for live demo</li>
        </ul>
      </Section>

      <Section title='Data Preparation'>
        <p className="text-gray-700">
          The dataset was downloaded from{" "}
          <a
            href="https://ieee-dataport.org/your-dataset-link"
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 underline hover:text-blue-800"
          >
            IEEE DataPort
          </a>. 
        </p>

        <p>
          The dataset was filtered to include only the five relevant conditions used in this model. The data was manually organized and split into <strong>training</strong>, <strong>validation</strong>, and <strong>test</strong> sets to ensure balanced class distribution.
        </p>

        <h4 className="text-md font-semibold text-gray-800 mb-4 mt-4">Data Summary</h4>
        <div className="overflow-x-auto">
          <table className="min-w-full border border-gray-300 text-sm text-left text-gray-700">
            <thead className="bg-gray-100 text-gray-800 font-semibold">
              <tr>
                <th className="px-4 py-2 border-b">Acronym</th>
                <th className="px-4 py-2 border-b">Full Name</th>
                <th className="px-4 py-2 border-b">Training</th>
                <th className="px-4 py-2 border-b">Validation</th>
                <th className="px-4 py-2 border-b">Test</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="px-4 py-2 border-b">NORMAL</td>
                <td className="px-4 py-2 border-b">Normal Retina</td>
                <td className="px-4 py-2 border-b">516</td>
                <td className="px-4 py-2 border-b">134</td>
                <td className="px-4 py-2 border-b">134</td>
              </tr>
              <tr>
                <td className="px-4 py-2 border-b">DR</td>
                <td className="px-4 py-2 border-b">Diabetic Retinopathy</td>
                <td className="px-4 py-2 border-b">375</td>
                <td className="px-4 py-2 border-b">132</td>
                <td className="px-4 py-2 border-b">124</td>
              </tr>
              <tr>
                <td className="px-4 py-2 border-b">ARMD</td>
                <td className="px-4 py-2 border-b">Age-Related Macular Degeneration</td>
                <td className="px-4 py-2 border-b">100</td>
                <td className="px-4 py-2 border-b">38</td>
                <td className="px-4 py-2 border-b">31</td>
              </tr>
              <tr>
                <td className="px-4 py-2 border-b">MH</td>
                <td className="px-4 py-2 border-b">Media Haze</td>
                <td className="px-4 py-2 border-b">316</td>
                <td className="px-4 py-2 border-b">92</td>
                <td className="px-4 py-2 border-b">100</td>
              </tr>
              <tr>
                <td className="px-4 py-2">ODC</td>
                <td className="px-4 py-2">Optic Disc Cupping</td>
                <td className="px-4 py-2">281</td>
                <td className="px-4 py-2">72</td>
                <td className="px-4 py-2">91</td>
              </tr>
            </tbody>
          </table>
        </div>
        <h4 className="text-lg font-semibold text-gray-800 mt-4">Image Preprocessing</h4>
          <p className="text-gray-700 mb-4">
            The following default image transforms were applied to match the expectations of the pretrained models:
          </p>
          <ul className="list-disc list-inside text-gray-700">
            <li><strong>Resize:</strong> 300x300 (EffNet) / 224x224 (ViT)</li>
            <li><strong>Normalization:</strong> Mean = [0.485, 0.456, 0.406], Std = [0.229, 0.224, 0.225]</li>
            <li><strong>ToTensor:</strong> Converts PIL images to PyTorch tensors</li>
            <li><strong>Optional Augmentations:</strong> RandomHorizontalFlip, ColorJitter (during training only)</li>
          </ul>
        </Section>

        <Section title="Model Training">
          <p className="text-gray-700 mb-4">
            The primary goal of this project was to strike a balance between <strong>model accuracy</strong> and <strong>inference efficiency</strong>,
            particularly for potential deployment in resource-constrained environments.
          </p>

          <h4 className="text-md font-semibold text-gray-800 mb-2">Architectures Explored</h4>
          <ul className="list-disc list-inside text-gray-700">
            <li><strong>EfficientNet-B3:</strong> A compact yet powerful CNN-based architecture known for speed and accuracy.</li>
            <li><strong>Vision Transformer (ViT):</strong> A transformer-based image model with strong feature extraction capabilities.</li>
          </ul>

          <p className="text-gray-700 mt-4">
            Both models were <strong>implemented from scratch</strong> to experiment with pretraining on the dataset. However, due to the limited size of
            the dataset (~1,600 images), pretraining from scratch resulted in underwhelming performance. As a result, both models were fine-tuned from
            their pretrained versions.
          </p>

          <p className="text-gray-700 mb-4">
            While the ViT architecture slightly outperformed EfficientNet-B3 in terms of raw accuracy, EfficientNet was selected for deployment due to
            its significantly lower inference time and resource requirements.
          </p>

          <div className="mt-4">
            <h3 className="text-lg font-semibold text-gray-800">Model Comparison</h3>
            <p className="text-gray-700">
              Below is a comparison of key metrics between different architectures explored for the retinal disease classification task:
            </p>

            <div className="overflow-x-auto mt-4">
              <table className="table-auto w-full border-collapse border border-gray-300 text-sm text-left">
                <thead className="bg-gray-100">
                  <tr>
                    <th className="border border-gray-300 px-4 py-2">Model</th>
                    <th className="border border-gray-300 px-4 py-2">Accuracy</th>
                    <th className="border border-gray-300 px-4 py-2">Inference Time</th>
                    <th className="border border-gray-300 px-4 py-2">Model Size</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="border border-gray-300 px-4 py-2">EfficientNet-B3</td>
                    <td className="border border-gray-300 px-4 py-2">82%</td>
                    <td className="border border-gray-300 px-4 py-2">~0.5 sec</td>
                    <td className="border border-gray-300 px-4 py-2">~12M parameters</td>
                  </tr>
                  <tr>
                    <td className="border border-gray-300 px-4 py-2">EfficientNet-B4</td>
                    <td className="border border-gray-300 px-4 py-2">84%</td>
                    <td className="border border-gray-300 px-4 py-2">~0.85 sec</td>
                    <td className="border border-gray-300 px-4 py-2">~19M parameters</td>
                  </tr>
                  <tr>
                    <td className="border border-gray-300 px-4 py-2">Vision Transformer (ViT)</td>
                    <td className="border border-gray-300 px-4 py-2">85%</td>
                    <td className="border border-gray-300 px-4 py-2">~0.9 sec</td>
                    <td className="border border-gray-300 px-4 py-2">~22M parameters</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

        <section id="weight-balancing" className="my-12">
          <h2 className="text-xl font-bold mb-4">Weight Balancing</h2>

          <p className="mb-4">
            During initial training, the model exhibited a strong bias toward <strong>classifying most disease cases as Diabetic Retinopathy (DR)</strong>, due to class imbalance in the dataset.
            This resulted in poor performance on underrepresented classes like ARMD and Optic Disc Cupping.
          </p>
          <p className="mb-4">
            To address the <strong>class imbalance</strong> present in the dataset—such as fewer samples of ARMD and Optic Disc Cupping compared to Normal or DR—
            <strong> class weighting</strong> was applied during training. This ensured that minority classes contributed proportionally to the loss function, helping the model avoid overfitting to majority classes.
          </p>

          <p className="mb-4">
            <strong>Strategy:</strong>
            <ul className="list-disc ml-6 mt-2">
              <li>Calculated weights inversely proportional to class frequencies in the training set.</li>
              <li>Applied these weights to the cross-entropy loss function.</li>
              <li>Improved recall and F1-score for underrepresented diseases like ARMD and ODC.</li>
            </ul>
          </p>

          <p className="mb-4">
            <strong>Formula Used:</strong><br />
            <code className="bg-gray-100 p-1 rounded block mt-1">
              Weight<sub>i</sub> = Total Training Samples / (Number of Classes × Samples in Class i)
            </code>
          </p>

          <p className="mb-2"><strong>Example Weights:</strong></p>
          <div className="overflow-x-auto">
            <table className="min-w-full table-auto border border-collapse border-gray-300">
              <thead className="bg-gray-100">
                <tr>
                  <th className="border border-gray-300 px-4 py-2 text-left">Class</th>
                  <th className="border border-gray-300 px-4 py-2 text-left">Training Samples</th>
                  <th className="border border-gray-300 px-4 py-2 text-left">Weight (normalized)</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="border border-gray-300 px-4 py-2">NORMAL</td>
                  <td className="border border-gray-300 px-4 py-2">516</td>
                  <td className="border border-gray-300 px-4 py-2">0.37</td>
                </tr>
                <tr>
                  <td className="border border-gray-300 px-4 py-2">DR</td>
                  <td className="border border-gray-300 px-4 py-2">375</td>
                  <td className="border border-gray-300 px-4 py-2">0.51</td>
                </tr>
                <tr>
                  <td className="border border-gray-300 px-4 py-2">ARMD</td>
                  <td className="border border-gray-300 px-4 py-2">100</td>
                  <td className="border border-gray-300 px-4 py-2">1.92</td>
                </tr>
                <tr>
                  <td className="border border-gray-300 px-4 py-2">MH</td>
                  <td className="border border-gray-300 px-4 py-2">316</td>
                  <td className="border border-gray-300 px-4 py-2">0.60</td>
                </tr>
                <tr>
                  <td className="border border-gray-300 px-4 py-2">ODC</td>
                  <td className="border border-gray-300 px-4 py-2">281</td>
                  <td className="border border-gray-300 px-4 py-2">0.67</td>
                </tr>
              </tbody>
            </table>
          </div>

          <p className="mt-4">
            This approach significantly enhanced the model’s ability to identify <strong>less frequent retinal diseases</strong>, particularly within the hierarchical classification system.
          </p>
        </section>

        <section id="training-config">
          <h2 className="text-xl font-bold mb-4">Training Configuration</h2>
          <ul className="list-disc ml-6 space-y-2 text-base">
            <li><strong>Optimizer:</strong> AdamW</li>
            <li><strong>Learning Rate:</strong> 1e-4</li>
            <li><strong>Epochs:</strong> 10</li>
            <li><strong>Loss Function:</strong> Cross Entropy Loss</li>
            <li><strong>Hardware:</strong> Trained on NVIDIA RTX 4080</li>
          </ul>
        </section>
        </Section>

        <Section title='Evaluation Method'>
          <p className="text-base mb-4">
            The model’s performance was assessed using two primary evaluation metrics to ensure both sensitivity and specificity in a medical context:
          </p>

          <ul className="list-disc ml-6 space-y-3 text-base">
            <li>
              <strong>Recall (Disease Detection):</strong> 
              &nbsp;This metric evaluates the model’s ability to detect the presence of any retinal disease, regardless of type. High recall is crucial in healthcare applications to minimize false negatives—cases where a disease is present but the model fails to identify it. 
              <br />
              <span className="font-semibold">Result:</span> 85% Recall on the validation set.
            </li>

            <li>
              <strong>Accuracy (Disease Classification):</strong> 
              &nbsp;Measures the correctness of disease type predictions, given that a disease has been detected. It reflects the model’s precision in distinguishing among different retinal disease categories.
              <br />
              <span className="font-semibold">Result:</span> 82% Classification Accuracy across all disease classes.
            </li>
          </ul>

          <p className="text-base mt-4">
            These metrics were computed on the <strong>validation set</strong> after training completion.
          </p>

          <section className="mt-10">
            <h2 className="text-2xl font-semibold mb-4">Class-wise Performance Metrics</h2>
            <p className="text-base mb-4">
              To better understand the model’s classification behavior, the following is the calculated precision, recall, and F1-score for each disease class using the validation set. 
            </p>

            <div className="overflow-x-auto">
              <table className="min-w-full table-auto border border-gray-300 text-left text-sm">
                <thead className="bg-gray-100">
                  <tr>
                    <th className="px-4 py-2 border">Class</th>
                    <th className="px-4 py-2 border">Precision</th>
                    <th className="px-4 py-2 border">Recall</th>
                    <th className="px-4 py-2 border">F1-Score</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-t">
                    <td className="px-4 py-2 border">Normal Retina</td>
                    <td className="px-4 py-2 border">0.91</td>
                    <td className="px-4 py-2 border">0.89</td>
                    <td className="px-4 py-2 border">0.90</td>
                  </tr>
                  <tr className="border-t">
                    <td className="px-4 py-2 border">Diabetic Retinopathy (DR)</td>
                    <td className="px-4 py-2 border">0.76</td>
                    <td className="px-4 py-2 border">0.88</td>
                    <td className="px-4 py-2 border">0.81</td>
                  </tr>
                  <tr className="border-t">
                    <td className="px-4 py-2 border">Age-Related Macular Degeneration (ARMD)</td>
                    <td className="px-4 py-2 border">0.74</td>
                    <td className="px-4 py-2 border">0.68</td>
                    <td className="px-4 py-2 border">0.71</td>
                  </tr>
                  <tr className="border-t">
                    <td className="px-4 py-2 border">Media Haze (MH)</td>
                    <td className="px-4 py-2 border">0.79</td>
                    <td className="px-4 py-2 border">0.75</td>
                    <td className="px-4 py-2 border">0.77</td>
                  </tr>
                  <tr className="border-t">
                    <td className="px-4 py-2 border">Optic Disc Cupping (ODC)</td>
                    <td className="px-4 py-2 border">0.81</td>
                    <td className="px-4 py-2 border">0.73</td>
                    <td className="px-4 py-2 border">0.77</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>
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


export default RetinalDocs;
