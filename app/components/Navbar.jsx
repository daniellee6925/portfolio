"use client";
import { assets } from '@/assets/assets'
import React, {useEffect, useRef, useState} from 'react'
import Image from 'next/image'
import Link from 'next/link';

const Navbar = () => {
    const [isScroll, setIsScroll] = useState(false)
    const sideMenuRef = useRef();
    
    const openMenu = () =>{
      sideMenuRef.current.style.transform = 'translateX(-16rem)'
    }
    const closeMenu = () =>{
      sideMenuRef.current.style.transform = 'translateX(16rem)'
    }

    useEffect(()=>{
      window.addEventListener('scroll', ()=>{
        if(scrollY > 50){
          setIsScroll(true)
        }else{
          setIsScroll(false)
        }
      })
    },[])

  return (
    <>
    <div className='fixed top-0 right-0 w-11/12 -z-10 translate-y-[-80%]'>
      <Image src={assets.header_bg_color} alt="" className='w-full'/>
    </div>
      <nav className={'w-full fixed px-5 lg:px-8 xl:px-[8%] py-4 flex items-center justify-between z-50 ${isScroll ? "bg-white bg-opacity-50 backdrop-blur-lg shadow-sm" : ""}'}>
        <a href ="/">
            <Image src = {assets.logo} alt = ""  className='w-40 cursor-pointer mr-14'/> 
        </a>
        <ul className={'hidden md:flex items-center gap-6 lg:gap-8 rounded-full px-12 py-3 bg-white shadow-sm bg-opacity-50'}>
            <li><Link className="font-Ovo hover:text-violet-700" href="/">Home</Link></li>
            <li><a className="font-Ovo hover:text-violet-700" href="/#about">About Me</a></li>
            <li><Link  className="font-Ovo hover:text-violet-700" href="/projects">Projects</Link ></li>
            <li><Link className="font-Ovo hover:text-violet-700" href="/learn">Learning of the Day</Link></li>
            <li><Link className="font-Ovo hover:text-violet-700" href="/paper">Paper Reviews</Link></li>
            <li><Link className="font-Ovo hover:text-violet-700" href="/FSA">Path to FSA</Link></li>
        </ul>

        <div className='flex items-center gap-4'>

          <a href ="https://chatwithdaniel.streamlit.app/" target="_blank" className='hidden lg:flex items-center gap-3 px-10
          py-2.5 border border-gray-500 rounded-full ml-4 font-Ovo hover:bg-violet-100 hover:-translate-y-1 duration-500'>
          Chat With Me <Image src={assets.arrow_icon} alt = "" className='w-3'/></a>

          <button className='block md:hidden ml-3' onClick={openMenu}>
            <Image src = {assets.menu_black} alt='' className='w-6'/> 
          </button>
        </div>
        {/*mobile menu*/}

        <ul ref = {sideMenuRef} className='flex md:hidden flex-col gap-4 py-20 px-10 fixed -right-64
        top-0 bottom-0 w-64 z-50 h-screen bg-rose-50 transition duration-500'>
            <div className='absolute right-6 top-6' onClick ={closeMenu}>
              <Image src={assets.close_black} alt='' className='w-5 cursor-pointer'/>
            </div>
            <li><Link className="font-Ovo" onClick ={closeMenu} href="/">Home</Link></li>
            <li><a className="font-Ovo" onClick ={closeMenu} href="/#about">About Me</a></li>
            <li><Link className="font-Ovo" onClick ={closeMenu} href="/projects">Projects</Link></li>
            <li><Link className="font-Ovo" onClick ={closeMenu} href="/learn">Learning of the day</Link></li>
            <li><Link className="font-Ovo" onClick ={closeMenu} href="/learn">Paper Reviews</Link></li>
            <li><Link className="font-Ovo" onClick ={closeMenu} href="/FSA">Path to FSA</Link></li>
        </ul>

      </nav>
    </>
  )
}

export default Navbar
