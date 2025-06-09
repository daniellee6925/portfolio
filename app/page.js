'use client'
import Navbar  from "./components/Navbar";
import Header  from "./components/Header";
import About  from "./components/About";
import Education from "./components/Education";
import Actuary from "./components/Actuary";
import Footer  from "./components/Footer";

export default function Home() {
  return (
    <>
    <Navbar />
    <Header />
    <About />
    <Actuary/>
    <Education />
    <Footer/>
    </>
  );
}
