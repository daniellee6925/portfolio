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
            <div className='flex flex-col sm:flex-row items-center gap-4 mt-4'>
                <a href="#contact" 
                className='px-10 py-3 border border-white rounded-full bg-black text-white flex items-center gap-2 hover:bg-violet-900 hover:-translate-y-1 duration-500'
                >Chat with Me<Image src = {assets.right_arrow_white} alt = ''
                className = "w-4"/>
                </a>
                <a href="/sample-resume.pdf" download rel="noopener noreferrer"
                className='px-10 py-3 border rounded-full border-gray-500 flex items-center gap-2 hover:bg-violet-200 hover:-translate-y-1 duration-500'>
                    my resume <Image src = {assets.download_icon} alt = ''
                className = "w-4"/>
                </a>
            </div>
    </div>
  )
}

export default Header
