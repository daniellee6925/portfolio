'use client';

import { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import Navbar from '../components/Navbar'; 
import Footer from '../components/Footer'; 

export default function LearningPage() {
  const [content, setContent] = useState('');

  useEffect(() => {
    fetch('/research.md')
      .then(res => res.text())
      .then(setContent);
  }, []);

  return (
    <div>
      <Navbar /> 
    <div className="prose max-w-5xl mx-auto p-4 pt-[7%]">
      <h1 className="text-2xl font-bold mb-4">Paper Reivews</h1>
      <p className="text-gray-700 mb-4">
        The <strong>Research Reviews</strong> section captures my takeaways from reading academic papers.
        It serves as a personal archive to solidify my understanding, a resource for others looking to grasp key insights without reading full papers.
      </p>
      <p className="text-bold text-blue-700 mb-4">
        <strong>*You can open the original paper from arXiv by clicking the title of each paper</strong>
      </p>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={{
        ul: ({ node, ...props }) => (
          <ul className="list-disc ml-6" {...props} />
        ),
        li: ({ node, ...props }) => (
          <li className="ml-2" {...props} />
        ),
        p: ({ node, ...props }) => (
          <p className="mb-2" {...props} />
        )
      }}>{content}</ReactMarkdown>
    </div>
     <Footer/>
    </div>
  );
}
