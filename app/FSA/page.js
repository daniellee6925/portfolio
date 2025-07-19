'use client';

import { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import Navbar from '../components/Navbar'; 
import Footer from '../components/Footer'; 

export default function LearningPage() {
  const [content, setContent] = useState('');

  useEffect(() => {
    fetch('/FSA.md')
      .then(res => res.text())
      .then(setContent);
  }, []);

  return (
    <div>
      <Navbar /> 
    <div className="prose max-w-5xl mx-auto p-4 pt-[7%]">
      <h1 className="text-2xl font-bold mb-4">4 FSA exams in one sitting </h1>
      <p className="text-gray-700 mb-6">
        The <strong>FSA</strong> section documents what I’ve learned each day while studying FSA exams. The first exam is GH 101
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
