import React from 'react'
import logoImage from '../assets/luminix-logo.png'

export const Logo = ({ className = "h-8" }: { className?: string }) => {
  return (
    <img 
      src={logoImage}
      className={className}
      style={{ width: 'auto', height: '48px' }}
    />
  )
} 