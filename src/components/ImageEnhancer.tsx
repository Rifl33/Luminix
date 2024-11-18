"use client"

import { useState, useEffect, useRef } from 'react'
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Upload, ZoomIn, ZoomOut, Sun, Contrast, Wand2 } from 'lucide-react'
import { AIImageEnhancer } from '@/services/imageEnhancer'

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
    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => {
        const result = e.target?.result
        if (typeof result === 'string') {
          setImage(result)
        }
      }
      reader.readAsDataURL(file)
    }
  }

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
  }

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    const file = event.dataTransfer.files[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => {
        const result = e.target?.result
        if (typeof result === 'string') {
          setImage(result)
        }
      }
      reader.readAsDataURL(file)
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

  const handleDownload = () => {
    if (!enhancedImage) return
    
    const link = document.createElement('a')
    link.href = enhancedImage
    link.download = 'enhanced-image.png'
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 p-4">
      <div className="container mx-auto max-w-3xl">
        <h1 className="text-2xl font-bold mb-6 text-center">Image Enhancer</h1>
        
        <div className="mb-6">
          <div 
            className="border border-gray-700 rounded-lg p-8 text-center cursor-pointer hover:bg-gray-800 transition-colors"
            onDragOver={handleDragOver}
            onDrop={handleDrop}
          >
            <input
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              className="hidden"
              id="imageUpload"
            />
            <label htmlFor="imageUpload" className="cursor-pointer">
              <Upload className="mx-auto h-8 w-8 text-gray-400" />
              <p className="mt-2 text-sm text-gray-400">Upload image</p>
            </label>
          </div>
        </div>

        {image && enhancedImage && (
          <div className="space-y-6">
            <div 
              ref={imageContainerRef}
              className="relative h-[500px] overflow-hidden rounded-lg cursor-col-resize"
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

            <div className="mt-6 space-y-4">
              <div>
                <label htmlFor="brightness" className="block text-sm font-medium mb-1">
                  Brightness
                </label>
                <div className="flex items-center gap-4">
                  <ZoomOut className="h-4 w-4 text-gray-400" />
                  <Slider
                    id="brightness"
                    min={0}
                    max={200}
                    step={1}
                    value={[brightness]}
                    onValueChange={(value: number[]) => setBrightness(value[0])}
                    className="flex-grow"
                  />
                  <ZoomIn className="h-4 w-4 text-gray-400" />
                </div>
              </div>
              <div>
                <label htmlFor="contrast" className="block text-sm font-medium mb-1">
                  Contrast
                </label>
                <div className="flex items-center gap-4">
                  <Sun className="h-4 w-4 text-gray-400" />
                  <Slider
                    id="contrast"
                    min={0}
                    max={200}
                    step={1}
                    value={[contrast]}
                    onValueChange={(value) => setContrast(value[0])}
                    className="flex-grow"
                  />
                  <Contrast className="h-4 w-4 text-gray-400" />
                </div>
              </div>
            </div>
          </div>
        )}

        <div className="mt-6 flex justify-center space-x-4">
          {enhancedImage && (
            <Button 
              onClick={handleDownload} 
              variant="outline" 
              className="bg-gray-800 text-gray-100 hover:bg-gray-700"
            >
              Download Enhanced Image
            </Button>
          )}
          <Button 
            onClick={handleAIEnhance} 
            className="bg-blue-600 text-white hover:bg-blue-700"
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