"use client"

import { useState, useEffect, useRef } from 'react'
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Upload, ZoomIn, ZoomOut, Sun, Contrast, Wand2, Download } from 'lucide-react'
import { AIImageEnhancer } from '@/services/imageEnhancer'
import { Logo } from './Logo'

export default function ImageEnhancer() {
  const [image, setImage] = useState<string | null>(null)
  const [enhancedImage, setEnhancedImage] = useState<string | null>(null)
  const [brightness, setBrightness] = useState(100)
  const [contrast, setContrast] = useState(100)
  const [isEnhancing, setIsEnhancing] = useState(false)
  const [comparePosition, setComparePosition] = useState(50)
  const enhancerRef = useRef<AIImageEnhancer>();
  const imageContainerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // Initialize the enhancer
    const initEnhancer = async () => {
      enhancerRef.current = new AIImageEnhancer();
    };
    initEnhancer();
  }, []);

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    
    if (file && (file.type === "image/jpeg" || file.type === "image/png")) {
      const reader = new FileReader()
      reader.onload = (e) => {
        const result = e.target?.result
        if (typeof result === 'string') {
          setImage(result)
        }
      }
      reader.readAsDataURL(file)
    } else {
      alert("Please upload only JPEG or PNG images")
    }
  }

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
  }

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    const file = event.dataTransfer.files[0]
    
    if (file && (file.type === "image/jpeg" || file.type === "image/png")) {
      const reader = new FileReader()
      reader.onload = (e) => {
        const result = e.target?.result
        if (typeof result === 'string') {
          setImage(result)
        }
      }
      reader.readAsDataURL(file)
    } else {
      alert("Please upload only JPEG or PNG images")
    }
  }

  const handleAIEnhance = async () => {
    if (!image || !enhancerRef.current) return
    
    setIsEnhancing(true)
    
    try {
      const canvas = document.createElement('canvas')
      const ctx = canvas.getContext('2d')
      if (!ctx) {
        throw new Error('Could not get canvas context')
      }

      const img = new Image()
      img.crossOrigin = 'anonymous'
      
      await new Promise((resolve, reject) => {
        img.onload = resolve
        img.onerror = reject
        img.src = image
      })

      canvas.width = img.width
      canvas.height = img.height
      ctx.drawImage(img, 0, 0)

      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
      const enhancedImageData = await enhancerRef.current.enhance(imageData)
      
      ctx.putImageData(enhancedImageData, 0, 0)
      const enhancedDataUrl = canvas.toDataURL('image/png')
      setEnhancedImage(enhancedDataUrl)
    } catch (error) {
      console.error('Error during AI enhancement:', error)
    } finally {
      setIsEnhancing(false)
    }
  }

  const handleCompareMove = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!imageContainerRef.current) return;
    
    const rect = imageContainerRef.current.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const position = (x / rect.width) * 100;
    setComparePosition(Math.min(Math.max(position, 0), 100));
  };

  const handleDownload = (format: 'png' | 'jpeg') => {
    if (!enhancedImage) return
    
    const canvas = document.createElement('canvas')
    const img = new Image()
    
    img.onload = () => {
      canvas.width = img.width
      canvas.height = img.height
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      
      ctx.drawImage(img, 0, 0)
      
      // Get the data URL in the specified format
      const dataUrl = canvas.toDataURL(`image/${format}`, 0.9)
      
      const link = document.createElement('a')
      link.href = dataUrl
      link.download = `enhanced-image.${format}`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    }
    
    img.src = enhancedImage
  }

  return (
    <div className="min-h-screen bg-[#0B0F1A] text-gray-100 p-4">
      <div className="container mx-auto max-w-3xl">
        <div className="flex justify-between items-center mb-8">
          <Logo className="h-8" />
          <a 
            href="https://github.com/Rifl33/Luminix"
            target="_blank"
            rel="noopener noreferrer"
            className="text-[#3B82F6] hover:text-[#60A5FA] transition-colors"
          >
            View GitHub
          </a>
        </div>
        
        <div className="mb-8">
          <div 
            className="border-2 border-dashed border-[#1E293B] rounded-xl p-12 text-center cursor-pointer hover:border-[#3B82F6] hover:bg-[#111827] transition-all duration-300"
            onDragOver={handleDragOver}
            onDrop={handleDrop}
          >
            <input
              type="file"
              accept="image/jpeg, image/png"
              onChange={handleImageUpload}
              className="hidden"
              id="imageUpload"
            />
            <label htmlFor="imageUpload" className="cursor-pointer">
              <Upload className="mx-auto h-12 w-12 text-[#3B82F6]" />
              <p className="mt-4 text-base text-gray-400">Drop your image here or click to upload</p>
              <p className="mt-2 text-sm text-gray-500">Supports PNG and JPEG</p>
            </label>
          </div>
        </div>

        {image && enhancedImage && (
          <div className="space-y-8">
            <div 
              ref={imageContainerRef}
              className="relative h-[500px] overflow-hidden rounded-xl cursor-col-resize bg-[#111827]"
              onMouseMove={handleCompareMove}
            >
              <img 
                src={image} 
                alt="Original" 
                className="absolute top-0 left-0 w-full h-full object-contain"
              />
              
              <img 
                src={enhancedImage} 
                alt="Enhanced" 
                className="absolute top-0 left-0 w-full h-full object-contain"
                style={{
                  clipPath: `inset(0 ${100 - comparePosition}% 0 0)`
                }}
              />
              
              <div 
                className="absolute top-0 bottom-0 w-1 bg-white cursor-col-resize"
                style={{ left: `${comparePosition}%` }}
              >
                <div className="absolute top-1/2 -translate-y-1/2 left-1/2 -translate-x-1/2 w-8 h-8 bg-white rounded-full flex items-center justify-center shadow-lg">
                  <div className="w-6 h-px bg-gray-800 rotate-90" />
                  <div className="w-6 h-px bg-gray-800 absolute" />
                </div>
              </div>

              <div className="absolute top-4 left-4 bg-black/50 px-2 py-1 rounded">
                Original
              </div>
              <div className="absolute top-4 right-4 bg-black/50 px-2 py-1 rounded">
                Enhanced
              </div>
            </div>

            <div className="space-y-6">
              <div>
                <label htmlFor="brightness" className="block text-sm font-medium mb-2 text-gray-300">
                  Brightness
                </label>
                <div className="flex items-center gap-4">
                  <ZoomOut className="h-4 w-4 text-[#3B82F6]" />
                  <Slider
                    id="brightness"
                    min={0}
                    max={200}
                    step={1}
                    value={[brightness]}
                    onValueChange={(value: number[]) => setBrightness(value[0])}
                    className="flex-grow"
                  />
                  <ZoomIn className="h-4 w-4 text-[#3B82F6]" />
                </div>
              </div>
              <div>
                <label htmlFor="contrast" className="block text-sm font-medium mb-2 text-gray-300">
                  Contrast
                </label>
                <div className="flex items-center gap-4">
                  <Sun className="h-4 w-4 text-[#3B82F6]" />
                  <Slider
                    id="contrast"
                    min={0}
                    max={200}
                    step={1}
                    value={[contrast]}
                    onValueChange={(value) => setContrast(value[0])}
                    className="flex-grow"
                  />
                  <Contrast className="h-4 w-4 text-[#3B82F6]" />
                </div>
              </div>
            </div>
          </div>
        )}

        <div className="mt-8 flex justify-center space-x-4">
          {enhancedImage && (
            <div className="flex gap-2">
              <Button 
                onClick={() => handleDownload('png')} 
                variant="outline" 
                className="bg-[#1E293B] text-gray-100 hover:bg-[#2D3748] border-[#3B82F6]"
              >
                <Download className="w-4 h-4 mr-2 text-[#3B82F6]" />
                Download PNG
              </Button>
              <Button 
                onClick={() => handleDownload('jpeg')} 
                variant="outline" 
                className="bg-[#1E293B] text-gray-100 hover:bg-[#2D3748] border-[#3B82F6]"
              >
                <Download className="w-4 h-4 mr-2 text-[#3B82F6]" />
                Download JPEG
              </Button>
            </div>
          )}
          <Button 
            onClick={handleAIEnhance} 
            className="bg-[#3B82F6] text-white hover:bg-[#2563EB] transition-colors"
            disabled={isEnhancing || !image}
          >
            <Wand2 className="w-4 h-4 mr-2" />
            {isEnhancing ? 'Enhancing...' : 'AI Enhance'}
          </Button>
        </div>
      </div>
    </div>
  )
} 