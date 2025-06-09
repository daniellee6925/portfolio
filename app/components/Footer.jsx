import React from 'react'
import Image  from 'next/image'
import { assets } from '@/assets/assets'

const footer = () => {
  return (
    <div className='mt-2'>
      <div className='text-center sm:flex items-center justify-between border-t border-gray-400 mx-[10%] mt-10 py-6'>
        <p>@ 2025 Daniel Lee. All Rights Reserved.</p>
        <ul className='flex items-center gap-10 justify-center mt-4 sm:mt-0'>
            <li><a target="_blank" href="https://github.com/daniellee6925" className='hover:text-indigo-700'>GitHub</a></li>
            <li><a target="_blank" href="https://www.linkedin.com/in/daniel-lee2023/" className='hover:text-indigo-700'>LinkedIn</a></li>
            <li><a target="_blank" href="mailto:daniellee6925@gmail.com" className='hover:text-indigo-700'>Email</a></li>
        </ul>
      </div>
    </div>
  )
}

export default footer
