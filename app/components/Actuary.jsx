import React from 'react';

const Actuary = () => {
  return (
    <div className="px-[12%] py-10">
      <h2 className="text-4xl font-semibold mb-6">What is an Actuary?</h2>
      <p className="text-gray-700 text-lg leading-relaxed mb-6">
        For those who are not familiar with what an actuary is, an actuary is a professional who applies mathematics, statistics, and probability theory to analyze and manage risk. 
        To become an <strong>Associate of the Society of Actuaries (ASA)</strong>, I had to pass a series of professional exams. Some of the topics include probability, statistics, and predictive analytics, followed by more specialized exams in topics like maximum likelihood estimation, bayesian inference and credibility, and risk measures (e.g., VaR, TVaR) and capital modeling.
      </p>
      <p className="text-gray-700 text-lg leading-relaxed mb-6">
        A <strong>health actuary</strong> specializes in analyzing the financial risks associated with healthcare by using statistics, data modeling, and predictive analytics. This discipline emphasizes working with large datasets, building regression models, and understanding real-world complexities like patient behavior and regulatory policy. These same analytical and technical skills—especially in modeling, data wrangling, and decision-making under uncertainty—are directly transferable to large language model (LLM) engineering. Actuaries' ability to extract insights from complex systems and structure them into quantifiable solutions provides a valuable mindset for building intelligent, reliable AI models..
      </p>
    </div>
  );
};

export default Actuary;
