import * as tf from '@tensorflow/tfjs';

export class AIImageEnhancer {
  private readonly COLOR_DEPTH = 255;

  constructor() {
    this.loadTensorflowBackend();
  }

  private async loadTensorflowBackend() {
    try {
      await tf.ready();
      await tf.setBackend('webgl');
      console.log('Using WebGL backend');
    } catch (error) {
      console.warn('WebGL not available, using CPU', error);
      await tf.setBackend('cpu');
    }
  }

  private async superResolution(tensor: tf.Tensor3D): Promise<tf.Tensor3D> {
    return tf.tidy(() => {
      const [height, width] = tensor.shape.slice(0, 2);
      let current = tensor;
      
      for (let i = 0; i < 2; i++) {
        const newHeight = Math.round(height * Math.pow(2, i + 1));
        const newWidth = Math.round(width * Math.pow(2, i + 1));
        
        current = tf.image.resizeBilinear(
          current.expandDims(0),
          [newHeight, newWidth],
          true
        ).squeeze([0]) as tf.Tensor3D;

        const details = this.extractMultiScaleDetails(current);
        current = tf.add(current, details.mul(0.7)).clipByValue(0, 1) as tf.Tensor3D;
        current = this.applyAdaptiveSharpening(current);
      }

      return current;
    });
  }

  private extractMultiScaleDetails(tensor: tf.Tensor3D): tf.Tensor3D {
    return tf.tidy(() => {
      const scales = [1, 2, 4];
      const weights = [0.5, 0.3, 0.2];
      
      const detailLevels = scales.map((scale, idx) => {
        // Extract details at different scales
        const blurred = tf.avgPool(
          tensor.expandDims(0),
          [scale * 2 + 1, scale * 2 + 1],
          [1, 1],
          'same'
        ).squeeze([0]) as tf.Tensor3D;
        
        const details = tensor.sub(blurred);
        
        // Apply frequency-dependent enhancement
        return details.mul(weights[idx] * (scale + 1));
      });

      // Combine all detail levels
      return tf.addN(detailLevels) as tf.Tensor3D;
    });
  }

  private applyAdaptiveSharpening(tensor: tf.Tensor3D): tf.Tensor3D {
    return tf.tidy(() => {
      // Create adaptive sharpening kernel
      const kernel = tf.tensor2d([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
      ]).div(8);

      const sharpenKernel = kernel
        .expandDims(2)
        .expandDims(3)
        .tile([1, 1, 3, 1]) as tf.Tensor4D;

      // Apply sharpening
      const sharpened = tf.conv2d(
        tensor.expandDims(0),
        sharpenKernel,
        1,
        'same'
      ).squeeze([0]) as tf.Tensor3D;

      // Detect edges for adaptive blending
      const edges = this.detectEdges(tensor);
      const edgeWeight = edges.mul(0.8); // Stronger sharpening near edges

      // Blend original and sharpened based on edge strength
      return tf.add(
        tensor.mul(tf.sub(1, edgeWeight)),
        sharpened.mul(edgeWeight)
      ).clipByValue(0, 1) as tf.Tensor3D;
    });
  }

  private async enhanceDetails(tensor: tf.Tensor3D): Promise<tf.Tensor3D> {
    return tf.tidy(() => {
      // Multi-frequency detail enhancement
      const frequencies = [
        { scale: 1, weight: 0.5, radius: 1 },
        { scale: 2, weight: 0.3, radius: 2 },
        { scale: 4, weight: 0.2, radius: 4 }
      ];

      const enhancedDetails = frequencies.map(({ scale, weight, radius }) => {
        // Extract and enhance details at each frequency
        const blurred = tf.avgPool(
          tensor.expandDims(0),
          [radius * 2 + 1, radius * 2 + 1],
          [1, 1],
          'same'
        ).squeeze([0]) as tf.Tensor3D;

        const details = tensor.sub(blurred);
        
        // Apply frequency-dependent enhancement
        return details.mul(weight * scale);
      });

      // Combine enhanced details
      const combinedDetails = tf.addN(enhancedDetails);

      // Apply local contrast enhancement
      const localContrast = this.enhanceLocalContrast(tensor);

      // Combine everything
      return tf.add(
        tensor,
        tf.add(combinedDetails, localContrast).mul(0.5)
      ).clipByValue(0, 1) as tf.Tensor3D;
    });
  }

  private enhanceLocalContrast(tensor: tf.Tensor3D): tf.Tensor3D {
    return tf.tidy(() => {
      const localMean = tf.avgPool(
        tensor.expandDims(0),
        [3, 3],
        [1, 1],
        'same'
      ).squeeze([0]) as tf.Tensor3D;

      const squaredDiff = tensor.sub(localMean).square() as tf.Tensor3D;
      const localVariance = tf.avgPool(
        squaredDiff.expandDims(0),
        [3, 3],
        [1, 1],
        'same'
      ).squeeze([0]) as tf.Tensor3D;

      return tensor.sub(localMean)
        .mul(tf.sqrt(localVariance.add(0.01)))
        .mul(1.5) as tf.Tensor3D;
    });
  }

  private async denoise(tensor: tf.Tensor3D): Promise<tf.Tensor3D> {
    return tf.tidy(() => {
      // Advanced bilateral filtering
      const sigma = 2.0;
      const kernelSize = 5;
      
      // Create spatial kernel
      const kernel = this.createGaussianKernel(kernelSize, sigma);
      
      // Apply filtering while preserving edges
      const filtered = tf.conv2d(
        tensor.expandDims(0),
        kernel,
        1,
        'same'
      ).squeeze([0]) as tf.Tensor3D;

      // Edge-aware blending
      const edges = this.detectEdges(tensor);
      return tf.where(
        edges,
        tensor,
        filtered
      ) as tf.Tensor3D;
    });
  }

  private async enhanceColors(tensor: tf.Tensor3D): Promise<tf.Tensor3D> {
    return tf.tidy(() => {
      const rgb = tensor.mul(this.COLOR_DEPTH);
      const [r, g, b] = tf.split(rgb, 3, 2) as [tf.Tensor3D, tf.Tensor3D, tf.Tensor3D];

      const colorParams = {
        vibrancy: 1.5,
        warmth: 1.2,
        saturation: 1.4,
        contrast: 1.25,
        brightness: 1.1,
        colorPop: 1.3
      };

      const maxRGB = tf.maximum(tf.maximum(r, g), b);
      const minRGB = tf.minimum(tf.minimum(r, g), b);
      const chroma = maxRGB.sub(minRGB);
      
      // Use luminance and saturationMask in calculations
      const luminance = r.mul(0.299).add(g.mul(0.587)).add(b.mul(0.114)) as tf.Tensor3D;
      const saturationMask = chroma.div(maxRGB.add(0.001));
      
      const midtones = maxRGB.add(minRGB).div(2);
      const luminanceWeight = luminance.mul(0.2);
      const saturationWeight = saturationMask.mul(0.8);
      
      // Apply weighted enhancements
      const rEnhanced = r.sub(midtones)
        .mul(colorParams.vibrancy)
        .add(midtones)
        .mul(colorParams.warmth)
        .mul(luminanceWeight.add(saturationWeight).add(1));

      const gEnhanced = g.sub(midtones)
        .mul(colorParams.vibrancy)
        .add(midtones)
        .mul(colorParams.warmth)
        .mul(luminanceWeight.add(saturationWeight).add(1));

      const bEnhanced = b.sub(midtones)
        .mul(colorParams.vibrancy)
        .add(midtones)
        .mul(0.95) // Slight cool tint
        .mul(luminanceWeight.add(saturationWeight).add(1));

      // Selective color enhancement based on color intensity
      const colorIntensity = chroma.div(maxRGB.add(0.001));
      const colorBoost = colorIntensity.mul(0.3).add(1);

      // Apply selective color enhancement
      const rFinal = rEnhanced.mul(colorBoost);
      const gFinal = gEnhanced.mul(colorBoost);
      const bFinal = bEnhanced.mul(colorBoost);

      // Combine and apply final adjustments
      const combined = tf.stack([rFinal, gFinal, bFinal], 2)
        .div(this.COLOR_DEPTH);

      // Apply contrast enhancement for more pop
      const contrastEnhanced = combined
        .sub(0.5)
        .mul(colorParams.contrast)
        .add(0.5);

      // Final color grading
      return contrastEnhanced
        .mul(colorParams.brightness)
        .clipByValue(0, 1) as tf.Tensor3D;
    });
  }

  private async optimizeLighting(tensor: tf.Tensor3D): Promise<tf.Tensor3D> {
    return tf.tidy(() => {
      const rgb = tensor.mul(this.COLOR_DEPTH);
      const [r, g, b] = tf.split(rgb, 3, 2) as [tf.Tensor3D, tf.Tensor3D, tf.Tensor3D];
      
      // Calculate luminance and zones
      const luminance = r.mul(0.299).add(g.mul(0.587)).add(b.mul(0.114));
      
      // Advanced tonal zones
      const deepShadows = luminance.less(50).asType('float32');
      const shadows = luminance.greater(50).mul(luminance.less(85)).asType('float32');
      const darkMidtones = luminance.greater(85).mul(luminance.less(128)).asType('float32');
      const lightMidtones = luminance.greater(128).mul(luminance.less(170)).asType('float32');
      const highlights = luminance.greater(170).mul(luminance.less(220)).asType('float32');
      const brightHighlights = luminance.greater(220).asType('float32');
      
      // Zone-specific adjustments
      const adjustments = {
        deepShadows: 1.4,    // Stronger shadow recovery
        shadows: 1.3,        // Shadow lift
        darkMidtones: 1.2,   // Dark midtone contrast
        lightMidtones: 1.1,  // Light midtone contrast
        highlights: 0.95,    // Highlight protection
        brightHighlights: 0.9 // Highlight recovery
      };

      // Apply zone-specific adjustments
      const adjusted = tf.addN([
        rgb.mul(deepShadows).mul(adjustments.deepShadows),
        rgb.mul(shadows).mul(adjustments.shadows),
        rgb.mul(darkMidtones).mul(adjustments.darkMidtones),
        rgb.mul(lightMidtones).mul(adjustments.lightMidtones),
        rgb.mul(highlights).mul(adjustments.highlights),
        rgb.mul(brightHighlights).mul(adjustments.brightHighlights)
      ]);

      return adjusted.div(this.COLOR_DEPTH).clipByValue(0, 1) as tf.Tensor3D;
    });
  }

  private async enhanceSkin(tensor: tf.Tensor3D): Promise<tf.Tensor3D> {
    return tf.tidy(() => {
      const rgb = tensor.mul(this.COLOR_DEPTH);
      const [r, g, b] = tf.split(rgb, 3, 2) as [tf.Tensor3D, tf.Tensor3D, tf.Tensor3D];

      // Detect skin tones
      const skinMask = this.detectSkinTones(r, g, b);

      // Enhance skin areas
      const skinEnhanced = this.applySkinEnhancements(r, g, b);
      
      // Blend with original
      return tf.where(
        skinMask,
        skinEnhanced,
        tensor
      ) as tf.Tensor3D;
    });
  }

  private detectSkinTones(r: tf.Tensor3D, g: tf.Tensor3D, b: tf.Tensor3D): tf.Tensor3D {
    return tf.tidy(() => {
      // YCbCr-based skin detection
      const y = r.mul(0.299).add(g.mul(0.587)).add(b.mul(0.114));
      const cb = b.sub(y).mul(0.564).add(128);
      const cr = r.sub(y).mul(0.713).add(128);

      // Skin tone ranges
      const skinMask = cb.greater(77).mul(cb.less(127))
        .mul(cr.greater(133)).mul(cr.less(173))
        .asType('float32') as tf.Tensor3D;

      return this.smoothMask(skinMask);
    });
  }

  private applySkinEnhancements(r: tf.Tensor3D, g: tf.Tensor3D, b: tf.Tensor3D): tf.Tensor3D {
    return tf.tidy(() => {
      // Skin enhancement parameters
      const params = {
        smoothing: 0.3,
        warmth: 1.1,
        brightness: 1.05,
        clarity: 1.1
      };

      // Enhanced skin tones
      const rEnhanced = r.mul(params.warmth).mul(params.brightness);
      const gEnhanced = g.mul(params.brightness);
      const bEnhanced = b.mul(0.95).mul(params.brightness); // Reduce blue

      return tf.stack([rEnhanced, gEnhanced, bEnhanced], 2)
        .div(this.COLOR_DEPTH)
        .clipByValue(0, 1) as tf.Tensor3D;
    });
  }

  private smoothMask(mask: tf.Tensor3D): tf.Tensor3D {
    return tf.tidy(() => {
      const kernel = tf.tensor2d(
        [[1, 2, 1],
         [2, 4, 2],
         [1, 2, 1]]
      ).div(16);

      const kernelTensor = kernel
        .expandDims(2)
        .expandDims(3)
        .tile([1, 1, 3, 1]) as tf.Tensor4D;

      return tf.conv2d(
        mask.expandDims(0),
        kernelTensor,
        1,
        'same'
      ).squeeze([0]) as tf.Tensor3D;
    });
  }

  private createGaussianKernel(size: number, sigma: number): tf.Tensor4D {
    return tf.tidy(() => {
      const kernel = tf.buffer([size, size]);
      const center = Math.floor(size / 2);
      
      for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
          const x = i - center;
          const y = j - center;
          const value = Math.exp(-(x * x + y * y) / (2 * sigma * sigma));
          kernel.set(value, i, j);
        }
      }
      
      const normalized = kernel.toTensor().div(kernel.toTensor().sum());
      return normalized.expandDims(2).expandDims(3).tile([1, 1, 3, 1]) as tf.Tensor4D;
    });
  }

  private detectEdges(tensor: tf.Tensor3D): tf.Tensor3D {
    return tf.tidy(() => {
      // Advanced edge detection with multiple kernels
      const sobelX = tf.tensor2d([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]);
      const sobelY = tf.tensor2d([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]);
      
      // Diagonal edge detection
      const sobelDiag1 = tf.tensor2d([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]);
      const sobelDiag2 = tf.tensor2d([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]);

      // Create kernels for each direction
      const kernels = [sobelX, sobelY, sobelDiag1, sobelDiag2].map(kernel => 
        kernel.expandDims(2).expandDims(3).tile([1, 1, 3, 1]) as tf.Tensor4D
      );

      // Apply each kernel
      const edgeResponses = kernels.map(kernel => 
        tf.conv2d(
          tensor.expandDims(0),
          kernel,
          1,
          'same'
        )
      );

      // Combine edge responses
      const magnitudes = edgeResponses.map(response => 
        response.square().squeeze([0])
      );

      const combinedMagnitude = tf.sqrt(tf.addN(magnitudes));

      // Apply non-maximum suppression for cleaner edges
      const nms = this.applyNonMaximumSuppression(combinedMagnitude as tf.Tensor3D);

      // Adaptive thresholding for better edge detection
      const threshold = this.calculateAdaptiveThreshold(nms);
      const edges = nms.greater(threshold).asType('float32');

      // Edge thinning and enhancement
      const thinned = this.thinEdges(edges as tf.Tensor3D);

      return thinned;
    });
  }

  private applyNonMaximumSuppression(magnitude: tf.Tensor3D): tf.Tensor3D {
    return tf.tidy(() => {
      const [height, width] = magnitude.shape.slice(0, 2);
      const suppressed = tf.buffer([height, width, 3]);

      const magnitudeData = magnitude.arraySync() as number[][][];

      for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
          for (let c = 0; c < 3; c++) {
            const current = magnitudeData[y][x][c];
            const neighbors = [
              magnitudeData[y-1][x][c],
              magnitudeData[y+1][x][c],
              magnitudeData[y][x-1][c],
              magnitudeData[y][x+1][c]
            ];

            if (current > Math.max(...neighbors)) {
              suppressed.set(current, y, x, c);
            } else {
              suppressed.set(0, y, x, c);
            }
          }
        }
      }

      return suppressed.toTensor() as tf.Tensor3D;
    });
  }

  private calculateAdaptiveThreshold(tensor: tf.Tensor3D): tf.Scalar {
    return tf.tidy(() => {
      // Calculate mean and standard deviation
      const mean = tf.mean(tensor);
      const std = tf.sqrt(
        tf.mean(tf.square(tf.sub(tensor, mean)))
      );

      // Adaptive threshold based on local statistics
      return tf.add(mean, std.mul(2)) as tf.Scalar;
    });
  }

  private thinEdges(edges: tf.Tensor3D): tf.Tensor3D {
    return tf.tidy(() => {
      const [height, width] = edges.shape.slice(0, 2);
      const thinned = tf.buffer([height, width, 3]);

      const edgeData = edges.arraySync() as number[][][];

      // Edge thinning using Zhang-Suen algorithm
      for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
          for (let c = 0; c < 3; c++) {
            if (edgeData[y][x][c] > 0) {
              const neighbors = [
                edgeData[y-1][x-1][c], edgeData[y-1][x][c], edgeData[y-1][x+1][c],
                edgeData[y][x-1][c],                        edgeData[y][x+1][c],
                edgeData[y+1][x-1][c], edgeData[y+1][x][c], edgeData[y+1][x+1][c]
              ];

              const transitions = neighbors.reduce((count, val, i) => {
                const next = neighbors[(i + 1) % 8];
                return count + (val === 0 && next === 1 ? 1 : 0);
              }, 0);

              const neighborCount = neighbors.filter(n => n > 0).length;

              if (transitions === 1 && neighborCount >= 2 && neighborCount <= 6) {
                thinned.set(1, y, x, c);
              }
            }
          }
        }
      }

      return thinned.toTensor() as tf.Tensor3D;
    });
  }

  private async applyAdvancedEnhancements(tensor: tf.Tensor3D): Promise<tf.Tensor3D> {
    return tf.tidy(() => {
      // Advanced frequency separation
      const lowFreq = tf.avgPool(
        tensor.expandDims(0),
        [5, 5],
        [1, 1],
        'same'
      ).squeeze([0]) as tf.Tensor3D;

      const highFreq = tensor.sub(lowFreq);

      // Enhance high frequency details
      const enhancedHighFreq = highFreq.mul(1.4); // Boost fine details

      // Clarity enhancement using local contrast
      const localContrast = this.enhanceLocalContrast(tensor);
      
      // Combine frequencies with enhanced contrast
      const recombined = tf.add(
        lowFreq,
        enhancedHighFreq.mul(1.2)
      ).add(localContrast.mul(0.3)) as tf.Tensor3D;

      // Apply advanced tone mapping
      return this.applyAdvancedToneMapping(recombined);
    });
  }

  private applyAdvancedToneMapping(tensor: tf.Tensor3D): tf.Tensor3D {
    return tf.tidy(() => {
      // Split into luminance and color
      const [r, g, b] = tf.split(tensor, 3, 2) as [tf.Tensor3D, tf.Tensor3D, tf.Tensor3D];
      const luminance = r.mul(0.299).add(g.mul(0.587)).add(b.mul(0.114)) as tf.Tensor3D;

      // Apply S-curve for better contrast
      const enhanced = luminance.sub(0.5)
        .mul(Math.PI)
        .tanh()
        .mul(0.5)
        .add(0.5) as tf.Tensor3D;

      // Calculate color ratios
      const colorRatios = tf.stack([
        r.div(luminance.add(0.001)),
        g.div(luminance.add(0.001)),
        b.div(luminance.add(0.001))
      ], 2) as tf.Tensor3D;

      // Reapply color with enhanced luminance
      return colorRatios.mul(enhanced.expandDims(2)) as tf.Tensor3D;
    });
  }

  private async applySmartSharpening(tensor: tf.Tensor3D): Promise<tf.Tensor3D> {
    return tf.tidy(() => {
      // Multi-scale sharpening
      const scales = [1, 2, 4];
      const weights = [0.5, 0.3, 0.2];

      const sharpened = scales.map((scale, idx) => {
        const kernel = this.createUnsharpMaskKernel(scale);
        const blurred = tf.conv2d(
          tensor.expandDims(0),
          kernel,
          1,
          'same'
        ).squeeze([0]) as tf.Tensor3D;

        const mask = tensor.sub(blurred);
        return tensor.add(mask.mul(weights[idx]));
      });

      // Combine sharpened versions
      return tf.addN(sharpened).div(scales.length) as tf.Tensor3D;
    });
  }

  private createUnsharpMaskKernel(scale: number): tf.Tensor4D {
    return tf.tidy(() => {
      const size = scale * 2 + 1;
      const sigma = scale * 0.5;
      const center = Math.floor(size / 2);
      const kernel2D = tf.buffer([size, size]);

      let sum = 0;
      for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
          const x = i - center;
          const y = j - center;
          const value = Math.exp(-(x * x + y * y) / (2 * sigma * sigma));
          kernel2D.set(value, i, j);
          sum += value;
        }
      }

      // Normalize and create unsharp mask
      const normalized = kernel2D.toTensor().div(-sum);
      normalized.bufferSync().set(1 + sum, center, center); // Center pixel

      return normalized
        .expandDims(2)
        .expandDims(3)
        .tile([1, 1, 3, 1]) as tf.Tensor4D;
    });
  }

  public async enhance(imageData: ImageData): Promise<ImageData> {
    try {
      const tensor = tf.tidy(() => 
        tf.browser.fromPixels(imageData)
          .toFloat()
          .div(this.COLOR_DEPTH)
      ) as tf.Tensor3D;

      // Enhanced processing pipeline with new algorithms
      const denoised = await this.denoise(tensor);
      const upscaled = await this.superResolution(denoised);
      const advancedEnhanced = await this.applyAdvancedEnhancements(upscaled);
      const smartSharpened = await this.applySmartSharpening(advancedEnhanced);
      const detailsEnhanced = await this.enhanceDetails(smartSharpened);
      const colorEnhanced = await this.enhanceColors(detailsEnhanced);
      const lightingOptimized = await this.optimizeLighting(colorEnhanced);
      const skinEnhanced = await this.enhanceSkin(lightingOptimized);

      // Final adjustments with advanced tone mapping
      const enhanced = tf.tidy(() => {
        const finalParams = {
          sharpness: 1.3,
          contrast: 1.25,
          brightness: 1.1,
          saturation: 1.15,
          clarity: 1.2
        };

        const withParams = skinEnhanced
          .mul(finalParams.contrast)
          .mul(finalParams.brightness)
          .mul(finalParams.sharpness)
          .mul(finalParams.clarity) as tf.Tensor3D;

        return this.applyAdvancedToneMapping(withParams)
          .clipByValue(0, 1) as tf.Tensor3D;
      });

      // Convert to ImageData
      const outputCanvas = document.createElement('canvas');
      const ctx = outputCanvas.getContext('2d')!;
      
      const [height, width] = enhanced.shape.slice(0, 2);
      outputCanvas.width = width;
      outputCanvas.height = height;
      
      await tf.browser.toPixels(enhanced, outputCanvas);
      
      const enhancedImageData = ctx.getImageData(
        0, 
        0,
        outputCanvas.width,
        outputCanvas.height
      );

      // Cleanup
      tf.dispose([
        tensor, denoised, upscaled, advancedEnhanced,
        smartSharpened, detailsEnhanced, colorEnhanced,
        lightingOptimized, skinEnhanced, enhanced
      ]);

      return enhancedImageData;
    } catch (error) {
      console.error('Error enhancing image:', error);
      return imageData;
    }
  }
} 