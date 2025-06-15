import React, { useState, useEffect } from 'react';


export function Navbar() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  
  // Add scroll effect for sticky header
  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 20) {
        setScrolled(true);
      } else {
        setScrolled(false);
      }
    };
    
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);
  
  // Close mobile menu when clicking outside
  useEffect(() => {
    if (isMenuOpen) {
      document.body.style.overflow = 'hidden'; // Prevent scrolling when menu is open
      
      const handleClickOutside = (e) => {
        if (!e.target.closest('.nav-links') && !e.target.closest('.mobile-menu-button')) {
          setIsMenuOpen(false);
        }
      };
      
      document.addEventListener('click', handleClickOutside);
      return () => {
        document.removeEventListener('click', handleClickOutside);
        document.body.style.overflow = 'auto'; // Restore scrolling
      };
    }
  }, [isMenuOpen]);
  
  return (
    <nav className={`navbar ${scrolled ? 'scrolled' : ''}`}>
      <div className="logo">
        <svg 
          width="32" 
          height="32" 
          viewBox="0 0 24 24" 
          fill="none" 
          xmlns="http://www.w3.org/2000/svg"
          style={{ color: '#78d577' }}
        >
          <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" 
                stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
        <h1>Tajweed Checker</h1>
      </div>
      
      <button 
        className="mobile-menu-button" 
        onClick={() => setIsMenuOpen(!isMenuOpen)}
        aria-label={isMenuOpen ? "Close menu" : "Open menu"}
        aria-expanded={isMenuOpen}
      >
        {isMenuOpen ? (
          <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
            <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
          </svg>
        ) : (
          <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
            <path d="M3 18h18v-2H3v2zm0-5h18v-2H3v2zm0-7v2h18V6H3z"/>
          </svg>
        )}
      </button>
      
      <div className={`mobile-overlay ${isMenuOpen ? 'open' : ''}`}></div>
      
      <div className={`nav-links ${isMenuOpen ? 'open' : ''}`}>
        <a href="/" className="active">Home</a>
        <a href="/harf-selection">Practice</a>
        <a href="/harf-selection">Phonemes</a>
        <a href="/about">About</a>
        <a href="/contact" >Contact Us</a>
      </div>
    </nav>
  );
}