import { assets, toolsData } from '@/assets/assets'
import { infoList } from '@/assets/assets'
import React from 'react'
import Image from 'next/image'

const About = () => {
  return (
    <div id='about' className='w-full px-[12%] py-10 scroll-mt-20'>
      <h4 className='text-center mb-2 text-2xl font-Ovo'>Introduction</h4>
      <h2 className='text-center text-6xl font-Ovo'>About Me</h2>

        <div className='flex w-full flex-col lg:flex-row items-center gap-20 my-20'>
            <div className='w-64 sm:w-80 rounded-3xl max-w-none'>
                <Image src={assets.user_image} alt='user' className='w-full rounded-3xl'/>
            </div>
            <div className='flex-1'>
                <ul className="mb-6 list-disc list-inside space-y-2 text-gray-700">
                    <li>
                        <strong>Who I am:</strong> A health actuary (ASA) with a strong foundation in mathematics, statistics, and data science.
                    </li>
                    <li>
                        <strong>Where I work:</strong> Blue Shield of California.
                    </li>
                    <li>
                        <strong>Education:</strong> M.S. in Computer Science (in progress) from Georgia Tech, and a B.A. in Data Science from UC Berkeley.
                    </li>

                    <li>
                        <strong>Projects:</strong> AI insurance agent, multimodal rap lyric generator, and a retinal disease classifier.
                    </li>
                    <li>
                        <strong>Passion:</strong> Creating agentic AI that makes healthcare and insurance more transparent and consumer-friendly.
                    </li>
                    </ul>

                <ul className='grid grid-cols-1 sm:grid-cols-3 gap-6 max-w-2xl'>
                    {infoList.map(({ icon, title, description, href }, index) => (
                        <li key={index}>
                        <a
                            href={href}
                            className='block border-[0.5px] border-gray-400 rounded-xl p-6
                            cursor-pointer hover:bg-violet-100 hover:-translate-y-1 duration-500'
                        >
                            <Image src={icon} alt={title} className='w-7' />
                            <h3 className='my-4 font-semibold text-gray-700'>{title}</h3>
                            <p className='text-gray-600 text-sm'>
                            {description.split('\n').map((line, i) => (
                                <span key={i}>
                                {line}
                                <br />
                                </span>
                            ))}
                            </p>
                        </a>
                        </li>
                    ))}
                </ul>

            </div>
        </div>
    </div>
  )
}

export default About
