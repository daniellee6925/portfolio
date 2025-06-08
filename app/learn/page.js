'use client';

import { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import Navbar from '../components/Navbar'; 

export default function LearningPage() {
  const [content, setContent] = useState('');

  useEffect(() => {
    fetch('/learning_of_the_day.md')
      .then(res => res.text())
      .then(setContent);
  }, []);

  return (
    <div>
      <Navbar /> 
    <div className="prose max-w-5xl mx-auto p-4 pt-[7%]">
      <h1 className="text-2xl font-bold mb-4">Learning of the Day</h1>
      <p className="text-gray-700 mb-6">
        The <strong>Learning of the Day</strong> section documents what I’ve learned each day while studying machine learning. 
        It serves as a reference for myself when I need to recall concepts, a guide for others who are also starting out in ML, 
        and a record of my ongoing progress and improvement.
      </p>
      <ReactMarkdown>{content}</ReactMarkdown>
    </div>
    </div>
  );
}
