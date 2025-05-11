import React from 'react';

const Education = () => {
  return (
    <section id="education" className="w-full px-[12%] py-16 bg-white">
      <h2 className="text-4xl font-Ovo text-center mb-2">Education & Credentials</h2>

      <div className="flex flex-col gap-8 sm:gap-12">
        <div className="border-l-4 border-violet-600 pl-4">
          <h3 className="text-xl font-semibold">M.S. in Computer Science (in progress)</h3>
          <p className="text-gray-600">Georgia Institute of Technology</p>
          <p className="text-sm text-gray-500">Expected 2026</p>
        </div>

        <div className="border-l-4 border-violet-600 pl-4">
          <h3 className="text-xl font-semibold">B.A. in Data Science</h3>
          <p className="text-gray-600">University of California, Berkeley</p>
          <p className="text-sm text-gray-500">Graduated 2023</p>
        </div>

        <div className="border-l-4 border-violet-600 pl-4">
          <h3 className="text-xl font-semibold">Associate of the Society of Actuaries (ASA)</h3>
          <p className="text-gray-600">Society of Actuaries</p>
          <p className="text-sm text-gray-500">Earned: 2025</p>
        </div>
      </div>
    </section>
  );
};

export default Education;
