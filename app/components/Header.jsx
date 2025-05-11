import { assets } from '@/assets/assets'
import React from 'react'
import Image from 'next/image'

const Header = () => {
  return (
    <div className='w-full max-w-3xl text-center mx-auto min-h-screen flex flex-col 
        items-center justify-center gap-4'>
        <h3 className='flex items-end gap-2 text-7xl mb-3 font-Ovo'>
                Hi! I'm Daniel Lee <Image src={assets.hand_icon} alt = ''
                className = 'w-12'/>
        </h3>
        <h1 className='text-3xl text-5xl font-Ovo'>
            Actuary driven to solve real-world problems with LLMs.
        </h1>
            <p className='max-w-xl mx-auto font-Ovo'>
                Health actuary and aspiring ML engineer using LLMs to build consumer-friendly AI solutions for healthcare and insurance.
            </p>
            <div className='flex flex-col sm:flex-row items-center gap-4 mt-4'>
                <a href="#contact" 
                className='px-10 py-3 border border-white rounded-full bg-black text-white flex items-center gap-2'
                >talk to me <Image src = {assets.right_arrow_white} alt = ''
                className = "w-4"/>
                </a>
                <a href="/sample-resume.pdf" download rel="noopener noreferrer"
                className='px-10 py-3 border rounded-full border-gray-500 flex items-center gap-2'>
                    my resume <Image src = {assets.download_icon} alt = ''
                className = "w-4"/>
                </a>
            </div>
    </div>
  )
}

export default Header
