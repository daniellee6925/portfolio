'use client';

import React from 'react';
import Navbar from '../components/Navbar'; 
import Footer from '../components/Footer'; 
import Image from 'next/image'
import { assets } from '@/assets/assets'


const InsuranceDocs = () => {
  return (
    <div>
      <Navbar />
    <main className="max-w-6xl mx-auto px-6 py-12 text-gray-800 pt-[7%]">
      <h1 className="text-4xl font-bold text-red-800 mb-8">AI Insurance Agent</h1>

      <Section title="Overview">
        <p className='text-xl font-bold'>
          *Development is ongoing for this project.
        </p>
          <p className="text-gray-700 mb-4 mt-4">
          This application is a <strong>multi-agent</strong> system built to assist users in navigating <strong>health insurance-related</strong> topics. It routes tasks and questions to specialized agents to ensure accurate, specific, and professional responses. The system follows a <strong>supervisor framework</strong>, where a central Supervisor Agent coordinates the workflow by assigning tasks to the appropriate specialized agents based on context.
        </p>

        <h3 className='font-bold mb-2 mt-4'>System Configuration</h3>
        <li><strong>Primary LLM</strong>: gpt-4o</li>
        <li><strong>Architecture</strong>: state-based graph</li>
        <li><strong>Search Tool</strong>: Tavily Search Tool</li>
        <li><strong>Memory Checkpointer</strong>: SQLiteCheckpointer (for development only)</li>
        <li><strong>Vector DB</strong>: Pinecone</li>
        <li><strong>Output Validation</strong>: Pydantic schemas & structured output parsing</li>


        <p className="text-gray-700 mb-4 mt-4">
          The <strong>main purpose</strong> of this application is to help users make informed decisions when selecting a health insurance plan and to maximize the benefits of their current policy. It ensures that consumers understand the products they are considering, are aware of their rights, and receive clear, personalized guidance throughout the decision-making process.
        </p>

        <strong>GitHub Repository:</strong>{' '}
        <a
          href="https://github.com/daniellee6925/Insurance_Agent"
          target="_blank"
          rel="noopener noreferrer"
          className="text-blue-600 underline"
        >
          github.com/daniellee6925/Insurance_Agent
        </a>

      </Section>
      <Section title = "Specialized Agents">
          <ul className="list-disc pl-6 text-gray-700">
            <li><strong>Supervsior Agent</strong>: Orchestrates the workflow by routing tasks to the appropriate specialized agents.</li>
            <li><strong>Enhancer Agent</strong>: Rephrases vague or unclear questions.</li>
            <li><strong>Validator Agent</strong>: Checks the quality and correctness of Researcher agent responses.</li>
            <li><strong>Policy Expert (RAG) Agent</strong>: Retrieves insurance policy information using a retrieval-augmented approach.</li>
            <li><strong>Research Agent</strong>: Finds up-to-date data from reliable web sources.</li>
            <li><strong>Actuary Agent</strong>: Calculates actuarial values for health insurance plans.</li>
            <li><strong>Consultant Agent</strong>: Recommends the best insurance plan based on user preferences and risk profile.</li>
            <li><strong>Underwriting Agent</strong>: Assesses user risk by asking key health-related questions.</li>
            <li><strong>Verifier Agent</strong>: Ensures risk assesment from the Underwriting Agent is reasonable.</li>
          </ul>

      </Section>
      <Section title = "Framework Diagram">
        <div className=''>
              <Image src={assets.framework} alt="framework diagram" className='w-full'/>
        </div>

      </Section>
      <Section title = "Agent input and outputs">
        <table className="table-auto w-full border border-gray-300 text-left text-sm">
          <thead className="bg-gray-100">
            <tr>
              <th className="px-4 py-2 border">Agent</th>
              <th className="px-4 py-2 border">Role</th>
              <th className="px-4 py-2 border">Input</th>
              <th className="px-4 py-2 border">Output</th>
              <th className="px-4 py-2 border">Tools</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="px-4 py-2 border">Supervisor</td>
              <td className="px-4 py-2 border">Routes queries</td>
              <td className="px-4 py-2 border">User query</td>
              <td className="px-4 py-2 border">Agent route</td>
              <td className="px-4 py-2 border">Router logic (LLM-based)</td>
            </tr>
            <tr>
              <td className="px-4 py-2 border">Enhancer</td>
              <td className="px-4 py-2 border">Rephrases ambiguous questions</td>
              <td className="px-4 py-2 border">User query</td>
              <td className="px-4 py-2 border">Clarified query</td>
              <td className="px-4 py-2 border">Prompt template + LLM</td>
            </tr>
            <tr>
              <td className="px-4 py-2 border">Policy Expert (RAG) Agent</td>
              <td className="px-4 py-2 border">Provides policy info</td>
              <td className="px-4 py-2 border">Clarified query</td>
              <td className="px-4 py-2 border">Retrieved info</td>
              <td className="px-4 py-2 border">VectorDB + RAG chain</td>
            </tr>
            <tr>
              <td className="px-4 py-2 border">Research Agent</td>
              <td className="px-4 py-2 border">Fetches real-time info</td>
              <td className="px-4 py-2 border">Search Query</td>
              <td className="px-4 py-2 border">Latest info with citation</td>
              <td className="px-4 py-2 border">Web search (Tavily)</td>
            </tr>
            <tr>
              <td className="px-4 py-2 border">Validator</td>
              <td className="px-4 py-2 border">Reflects and improves responses</td>
              <td className="px-4 py-2 border">Research Agent output</td>
              <td className="px-4 py-2 border">Refined output/agent route</td>
              <td className="px-4 py-2 border">Self-reflection prompt</td>
            </tr>
            <tr>
              <td className="px-4 py-2 border">Actuary Agent</td>
              <td className="px-4 py-2 border">Computes actuarial value</td>
              <td className="px-4 py-2 border">Plan details</td>
              <td className="px-4 py-2 border">Actuarial value</td>
              <td className="px-4 py-2 border">Python formula</td>
            </tr>
            <tr>
              <td className="px-4 py-2 border">Consultant Agent</td>
              <td className="px-4 py-2 border">Recommends plans</td>
              <td className="px-4 py-2 border">User preferences</td>
              <td className="px-4 py-2 border">Suggested plan</td>
              <td className="px-4 py-2 border">Ranking + prompt</td>
            </tr>
            <tr>
              <td className="px-4 py-2 border">Underwriting Agent</td>
              <td className="px-4 py-2 border">Assesses risk</td>
              <td className="px-4 py-2 border">User profile</td>
              <td className="px-4 py-2 border">Risk score</td>
              <td className="px-4 py-2 border">LLM-based</td>
            </tr>
            <tr>
              <td className="px-4 py-2 border">Verifier Agent</td>
              <td className="px-4 py-2 border">Self-reflects risk assesment</td>
              <td className="px-4 py-2 border">Risk score and reason</td>
              <td className="px-4 py-2 border">Risk score/agent route</td>
              <td className="px-4 py-2 border">Self-reflection prompt</td>
            </tr>
          </tbody>
        </table>
      </Section>
      <Section title = "Supervsior Agent Framework">
        <p>
          The Supervisor Agent acts as the central coordinator. It routes user
          questions to the most appropriate specialized agent based on the context
          and content of the query. This ensures efficient and accurate handling of
          insurance-related topics by leveraging the strengths of specialized agents.
        </p>
        <p>
            <strong>Example User Input:</strong> I want to understand which health plan fits my family.
        </p>
        <p className="mt-2">
            <strong>Supervisor Agent Output:</strong> Routing query to the Consultant Agent for personalized plan recommendations.
        </p>
        <div className=''>
              <Image src={assets.supervisor} alt="supervisor diagram" className='w-full'/>
        </div>

      </Section>
      <Section title = "State-based Architecture">
        <p className="text-gray-700 mb-4">
          This multi-agent system is designed around a <strong>state-based architecture</strong>, 
          where a shared <code>state</code> object is passed between agents throughout the workflow.
          The state captures the evolving context of the user&apos;s interaction — including their inputs, 
          intermediary results, decisions made by agents, and messages exchanged.
        </p>
        <p className="text-gray-700 mb-4">
          Each agent reads from and writes to this state, enabling them to:
        </p>
        <ul className="list-disc list-inside text-gray-700 mb-4 space-y-1">
          <li>Access relevant background information or previous outputs</li>
          <li>Perform their specialized task based on up-to-date context</li>
          <li>Append their response or decision back into the workflow</li>
          <li>Enable coordination across agents without redundancy</li>
        </ul>
        <h3 className='text-xl font-bold mb-2 mt-8'>Sample Flow Chart</h3>
        <p className="text-gray-700 mb-4">
          The following example demonstrates the system assisting the user select their preferred plan:
        </p>
        <div className=''>
              <Image src={assets.flowchart} alt="flowchart diagram" className='w-full'/>
        </div>

      </Section>
      <Section title = "Policy Expert RAG Agent (Retrieval-Augmented Generation)">
        <p className="text-gray-700 mb-4">
          The RAG Agent is responsible for providing accurate and detailed answers based on real-world health insurance policy documents. 
          It retrieves relevant context from a custom-built vector database containing documents from major health insurers such as 
          <strong> UnitedHealthcare, Blue Shield of California,</strong> and <strong>Kaiser Permanente</strong>.
        </p>
        <p className="text-gray-700 mb-4">
          To ensure the quality and precision of its responses, the RAG Agent employs a <strong>self-reflection mechanism</strong> that:
        </p>
        <ul className="list-disc list-inside text-gray-700 mb-4 space-y-1">
          <li>Rephrases unclear or overly broad user queries to improve retrieval effectiveness</li>
          <li>Grades the relevance of retrieved documents before forming a response</li>
          <li>Filters out weak or unrelated context to ensure concise, accurate answers</li>
        </ul>
        <h3 className='text-xl font-bold mb-2 mt-8'>Sample Flow Chart</h3>
        <div className=''>
              <Image src={assets.rag} alt="rag diagram" className='w-full'/>
        </div>
        <p>
            <strong>Example User Input:</strong> I want to understand my drug benefits for the following Kaiser health plan.
        </p>
        <p className="mt-2">
            <strong>RAG Agent Output:</strong> For Kaiser plan XYZ, the copay of the generic drugs is $25. This means that you will pay $25 out of pocket for each prescription of a covered generic drug, regardless of the actual cost of the medication, and the insurance will cover the rest...
        </p>

      </Section>
      <Section title = "Underwriting Agent (+Verifier Agent)">
        <p className="text-gray-700 mb-4">
          The Underwriting Agent evaluates the user&apos;s risk profile to assign an appropriate risk classification. This classification is used 
          to determine coverage eligibility, premium pricing, and the most suitable health insurance options for the user.
        </p>

        <h3 className="text-lg font-semibold text-gray-800 mb-2">Risk Classifications</h3>
        <ul className="list-disc list-inside text-gray-700 mb-4 space-y-1">
          <li><strong>Low Risk</strong> – Excellent health, no chronic conditions, healthy lifestyle</li>
          <li><strong>Moderate Risk</strong> – Minor conditions (e.g., mild asthma) with good control, no major risk factors</li>
          <li><strong>High Risk</strong> – Serious conditions (e.g., diabetes, heart disease), or risky lifestyle behaviors</li>
          <li><strong>Major Risk</strong> – Conditions that are very costly (e.g., terminal illness or recent hospitalization)</li>
        </ul>

        <h3 className="text-lg font-semibold text-gray-800 mb-2">Factors Considered</h3>
        <p className="text-gray-700">
          The following info regarding the user will be asked by the LLM and stored in the state for risk assesment.
        </p>
        <ul className="list-disc list-inside text-gray-700 mb-4 space-y-1">
          <li>Age</li>
          <li>Gender</li>
          <li>Height & Weight</li>
          <li>Tobacco Use</li>
          <li>Alcohol Consumption</li>
          <li>Pre-existing Conditions</li>
          <li>Current Medications</li>
          <li>Occupation</li>
          <li>Risky Activities (e.g., skydiving, scuba diving)</li>
          <li>Recent Hospitalizations</li>
          <li>Family Medical History</li>
        </ul>

        <h3 className='text-xl font-bold mb-2 mt-8'>Sample Flow Chart</h3>
        <div className=''>
              <Image src={assets.underwriter} alt="underwriter diagram" className='w-full'/>
        </div>
        <p className="text-gray-700 mb-4">
          After generating the risk classification, the output is passed to the <strong>Validator Agent</strong>, which verifies whether the 
          classification and reasoning align logically with the provided health profile. If the validation passes, the system proceeds to 
          the Consultant Agent for plan recommendation. If validation fails, the Underwriting Agent re-assesses the profile.
        </p>
      </Section>
      
      <Section title = "Researcher Agent (+Validator Agent)">
        <p className="text-gray-700 mb-4">
          The Researcher Agent is responsible for conducting live-search on the web regarding user queries. It uses the <strong>Tavily Search API</strong> to retrieve real-time, up-to-date information.
        </p>

        <h3 className="text-lg font-semibold text-gray-800 mb-2">Role and Functionality</h3>
        <ul className="list-disc list-inside text-gray-700 mb-4 space-y-1">
          <li>Breaks down multi-part or vague questions into researchable tasks</li>
          <li>Compiles findings into a concise, accurate, and readable explanation</li>
          <li>Acts as a fallback when the initial agent response is insufficient</li>
        </ul>

        <h3 className="text-lg font-semibold text-gray-800 mb-2">Input and Output</h3>
        <ul className="list-disc list-inside text-gray-700 mb-4 space-y-1">
          <li><strong>Input:</strong> User’s question that requires up-to-date information</li>
          <li><strong>Output:</strong> Structured insight and citation derived from multiple sources</li>
        </ul>

        <h3 className="text-lg font-semibold text-gray-800">Validation</h3>
        <p className="text-gray-700 mb-4">
          The output of the Researcher Agent is passed to the <strong>Validator Agent</strong>, which reviews the relevance, completeness, and
          clarity of the answer. If the Validator approves, the result is returned to the user. If not, the task may be reassigned or refined for
          better accuracy.
        </p>
        <h3 className='text-xl font-bold mb-2 mt-8'>Sample Flow Chart</h3>
        <div className=''>
              <Image src={assets.researcher} alt="researcher diagram" className='w-full'/>
        </div>

      </Section>


      <Section title = "Acturay Agent">
        <p className="text-gray-700 mb-4">
          The Actuary Agent is responsible for calculating the actuarial value of a health insurance plan. <strong>Actuarial value (AV)</strong> represents the 
          percentage of total average costs for covered benefits that a plan will cover. This helps users compare the generosity of different 
          insurance plans and understand the financial implications of their choices.
        </p>

        <h3 className="text-lg font-semibold text-gray-800 mb-2">What It Calculates</h3>
        <ul className="list-disc list-inside text-gray-700 mb-4 space-y-1">
          <li>Estimated out-of-pocket costs based on plan details</li>
          <li>Coverage percentage of medical expenses</li>
          <li>Comparison across metal tiers (Bronze, Silver, Gold, Platinum)</li>
        </ul>

        <h3 className="text-lg font-semibold text-gray-800 mb-2">Input Parameters</h3>
        <ul className="list-disc list-inside text-gray-700 mb-4 space-y-1">
          <li>Deductibles</li>
          <li>Out-of-pocket maximums</li>
          <li>Coinsurance rates</li>
          <li>Copayment amounts</li>
          <li>Service utilization assumptions</li>
        </ul>

        <p className="text-gray-700 mb-4">
          The Actuary Agent uses a deterministic formula or simulation-based approach to derive the actuarial value. This value is presented 
          to the user in an easy-to-understand format, aiding in financial planning and plan comparison. The agent <strong>ensures transparency</strong> in how much the insurance company pays versus how much the <strong>consumer is expected to pay</strong>.
        </p>

        <p className="text-gray-700 mb-4">
          *The actuarial value calculator is currently undergoing updates. 
        </p>

      </Section>

      <Section title="Consultant Agent">
          <p className="text-gray-700 mb-4">
            The Consultant Agent specializes in helping users choose the most suitable health insurance plan based on their unique needs,
            preferences, and eligibility. It combines decision logic with prompt-driven reasoning to deliver tailored recommendations.
          </p>

          <h3 className="text-lg font-semibold text-gray-800 mb-2">Role and Functionality</h3>
          <ul className="list-disc list-inside text-gray-700 mb-4 space-y-1">
            <li>Analyzes user-provided preferences such as budget, coverage needs, providers, and prescription requirements</li>
            <li>Compares available plan options (e.g., HMO, PPO, EPO, HDHP) and ranks them by suitability</li>
            <li>Generates a concise explanation of the recommended plan, including trade-offs and fit rationale</li>
            <li>Guides users in understanding key policy components (deductibles, copays, out-of-pocket limits, etc.)</li>
            <li>Ensures the recommendation aligns with any constraints identified by the underwriting or actuarial agents</li>
          </ul>

          <h3 className="text-lg font-semibold text-gray-800 mb-2">Input and Output</h3>
          <ul className="list-disc list-inside text-gray-700 mb-4 space-y-1">
            <li><strong>Input:</strong> User preferences (budget, provider network, expected care needs, location, etc.)</li>
            <li><strong>Output:</strong> Recommended insurance plan with explanation and rationale</li>
          </ul>

          <h3 className="text-lg font-semibold text-gray-800 mb-2">This Agent is currently being developed</h3>
      </Section>

      <Section title="Deployment">
      <p className="text-gray-700 mb-4">
        <strong>Insurance Agent</strong> is being developed as a full-stack application. It will use FastAPI for backend and a React/Next.js for frontend.
      </p>
      <ul className="list-disc list-inside text-gray-700 mb-4">
        <li>
          <strong>Backend:</strong> Powered by <code className="bg-gray-100 px-1 rounded">FastAPI</code>, hosted on an <code className="bg-gray-100 px-1 rounded">AWS EC2</code> instance.
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
      <h2 className="text-3xl font-bold border-b border-gray-300 pb-2 mb-4">{title}</h2>
      {children}
    </section>
  );
}


export default InsuranceDocs;
