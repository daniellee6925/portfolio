'use client';

import React from 'react';
import Navbar from '../components/Navbar'; 



const InsuranceDocs = () => {
  return (
    <div>
      <Navbar />
    <main className="max-w-6xl mx-auto px-6 py-12 text-gray-800 pt-[7%]">
      <h1 className="text-4xl font-bold text-red-800 mb-8">Insurance Agent</h1>

      <Section title="Overview">
        <p className='text-3xl font-bold'>
          This project is currently in development
        </p>
        <p className="text-gray-700 mt-4">
          The AI Insurance Agent is an interactive assistant designed to help users navigate and select health insurance plans. Built with <strong>LangGraph</strong>, this agent leverages a combination of <strong>retrieval-augmented generation (RAG)</strong>, <strong>function-calling</strong>, and <strong>decision-oriented dialogue flows</strong> to provide personalized, accurate insurance guidance.
        </p>

        <div>
          <h3 className="text-xl font-semibold text-gray-800 mt-4 mb-2">Core Features:</h3>
          <ul className="list-disc list-inside text-gray-700 space-y-2">
            <li>
              <strong>RAG-Powered Q&amp;A:</strong> Search and summarize complex policy documents to answer user queries in plain language.
            </li>
            <li>
              <strong>Plan Comparison Tool:</strong> Dynamically compares multiple insurance plans based on user criteria (e.g., premiums, deductibles, out-of-pocket max).
            </li>
            <li>
              <strong>Actuarial Value Calculator:</strong> Uses built-in actuarial formulas to estimate the value of each plan for the user.
            </li>
            <li>
              <strong>Preference-Based Plan Recommendation:</strong> A scripted, chatbot-style flow that suggests plans aligned with a user’s budget, provider preferences, coverage needs, and risk tolerance.
            </li>
          </ul>
        </div>


      </Section>
    </main>
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


export default InsuranceDocs;
