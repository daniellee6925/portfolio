'use client';

import React from 'react';
import Navbar from '../components/Navbar'; // adjust path if needed
import Link from 'next/link';

const projects = [
  {
    title: 'AI Insurance Agent',
    description: 'Conversational agent to explain and quote health insurance.',
    href: '/projects/insurance-agent',
  },
  {
    title: 'RapGPT2.0',
    description: 'Multimodal LLM that generates rap lyrics and audio.',
    href: '/projects/rap-generator',
  },
  {
    title: 'Retinal Disease Classifier',
    description: 'CNN model that classifies retinal images for disease diagnosis.',
    href: 'https://huggingface.co/spaces/daniellee6925/Retinal_Disease',
  },
];

const detailedProjects = [
  {
    title: "AI Insurance Agent",
    description:
      "Built a large language model–powered chatbot to answer insurance plan questions with state-specific regulations. Integrated vector search and used Retrieval-Augmented Generation (RAG).",
    link: "/projects/insurance-agent",
  },
  {
    title: "RapGPT2.0-Eminem",
    description:
      "Built on GPT-2 with 124M parameters, this model was trained on AWS EC2 with LoRA fine-tuning to generate Eminem-style rap lyrics.\nIt includes KV-caching, flash attention, and quantization for fast inference.\nEvaluated model outputs with a BERT-based classifier to ensure stylistic fidelity with Eminem Lyrics\nIntegrated with ElevenLabs for audio output and deployed with a custom UI for interactive lyric generation...",
    link: "/projects/rap-generator",
  },
  {
    title: "Retinal Disease Classifier",
    description:
      "Developed a hierarchical classification model that identifies retinal diseases from medical images, achieving 85% recall in disease identification and 82% accuracy in classifying specific conditions.\nThe project involved building and training models using EfficientNet and Vision Transformers and was evaluated with the help of TensorBoard.\nTo optimize the model for deployment, model's size was reduced while maintaining high performance, achieving an average inference time of 0.5 seconds.\nThe final model was deployed on Hugging Face Spaces, allowing users to interact with the classifier via a simple, intuitive interface built with Gradio...",
    link: "/projects/retinal-classifier",
  },
];

const otherProjects = [
  { title: 'From Sratch', description: 'This repo includes from-scratch implementations of various machine learning frameworks, such as LoRA, GPT, Flash Attention, Neural Networks, Vision Transformers, EfficientNet, and more.' },
  { title: 'rapGPT 1.0', description: 'Trained a NanoGPT model using individual letters as tokens, with approximately 1 million parameters. The model performs very poorly which led to the rapGPT 2.0.' },
  { title: 'GF is Hungry-Fremont Edition', description: 'Developed a random restaurant chooser for restaurants in Fremont, where users can send a prompt. The model leverages FAISS to search a vector database and return restaurant matches based on the given prompt.' },
  { title: 'Poker Simulator', description: 'Created a poker simulator that calculates the win probability based on the cards in hand, the number of players, and varying community card combinations.' },
  { title: 'This Website', description: 'Built this website using Next.js,' },
];


const ProjectsPage = () => {
  return (
    <div>
      <Navbar />

      <section className="px-12 py-16 pt-[7%]">
        <h2 className="text-5xl font-Ovo text-center mb-10">Click Below to Try Out My Projects</h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8">
          {projects.map((project, index) => (
            <Link href={project.href} key={index}>
              <div className="border border-gray-300 rounded-xl p-6 hover:bg-violet-100 hover:-translate-y-1 transition-all duration-300 cursor-pointer">
                <h3 className="text-xl font-semibold text-gray-800 mb-2">{project.title}</h3>
                <p className="text-gray-600 text-sm">{project.description}</p>
              </div>
            </Link>
          ))}
        </div>
        
        <h2 className="text-5xl font-Ovo text-center mb-10 pt-10">Dive Deeper into the Details of Each Project Here!</h2>

        <div className="mt-10 grid grid-cols-1 gap-6">
          {detailedProjects.map(({ title, description, link }, index) => (
            <a
              href={link}
              key={index}
              className="block border border-gray-300 rounded-xl p-6 hover:bg-indigo-50 transition"
            >
              <h3 className="text-2xl font-semibold text-gray-800 mb-2">{title}</h3>
              <p className="text-gray-600">{description.split('\n').map((line, i) => (
                                <span key={i}>
                                {line}
                                <br />
                                </span>
                            ))}
              </p>
            </a>
          ))}
        </div>

        <div className="my-20">
          <h3 className="text-center text-3xl font-Ovo">Other Projects</h3>
          <div className="mt-10 grid grid-cols-1 gap-6">
            {otherProjects.map(({ title, description}, index) => (
              <div
                key={index}
                className="border border-gray-300 rounded-xl p-6 cursor-default"
              >
                <h3 className="text-xl font-semibold text-gray-800 mb-2">{title}</h3>
                <p className="text-gray-600 text-sm">{description}</p>
              </div>
            ))}
          </div>
        </div>

      </section>
    </div>
  );
};

export default ProjectsPage;
