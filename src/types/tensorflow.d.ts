declare module '@tensorflow/tfjs' {
  export interface Tensor {
    shape: number[];
    dtype: string;
    size: number;
    rank: number;
    mul(b: Tensor | number): Tensor;
    div(b: Tensor | number): Tensor;
    sub(b: Tensor | number): Tensor;
    add(b: Tensor | number): Tensor;
    greater(b: Tensor | number): Tensor;
    less(b: Tensor | number): Tensor;
    clipByValue(min: number, max: number): Tensor;
    expandDims(axis?: number): Tensor;
    squeeze(axis?: number[]): Tensor;
    tile(reps: number[]): Tensor;
    abs(): Tensor;
    asType(dtype: DataType): Tensor;
    toFloat(): Tensor;
    logicalAnd(b: Tensor): Tensor;
    square(): Tensor;
    maximum(b: Tensor | number): Tensor;
    minimum(b: Tensor | number): Tensor;
    sum(axis?: number | number[], keepDims?: boolean): Tensor;
    tanh(): Tensor;
    arraySync(): number[] | number[][] | number[][][] | number[][][][];
    bufferSync(): Buffer;
  }

  export interface Tensor2D extends Tensor {
    shape: [number, number];
    expandDims(axis?: number): Tensor3D;
  }

  export interface Tensor3D extends Tensor {
    shape: [number, number, number];
    expandDims(axis?: number): Tensor4D;
    squeeze(axis?: number[]): Tensor2D;
  }

  export interface Tensor4D extends Tensor {
    shape: [number, number, number, number];
    squeeze(axis?: number[]): Tensor3D;
  }

  export type DataType = 'float32' | 'int32' | 'bool' | 'complex64' | 'string';

  export type TensorLike = number | number[] | number[][] | number[][][] | number[][][][] | Uint8Array | Float32Array;

  export interface NamedTensorMap {
    [name: string]: Tensor;
  }

  export interface ModelPredictConfig {
    batchSize?: number;
    verbose?: boolean;
  }

  export function tensor(values: TensorLike, shape?: number[], dtype?: DataType): Tensor;
  export function tensor2d(values: TensorLike, shape?: [number, number], dtype?: DataType): Tensor2D;
  export function tensor3d(values: TensorLike, shape?: [number, number, number], dtype?: DataType): Tensor3D;
  export function tensor4d(values: TensorLike, shape?: [number, number, number, number], dtype?: DataType): Tensor4D;

  export function tidy<T>(nameOrFn: string | (() => T), fn?: () => T): T;
  export function dispose(container: TensorContainer): void;
  export function memory(): {
    numTensors: number;
    numDataBuffers: number;
    numBytes: number;
    unreliable?: boolean;
  };
  export function ready(): Promise<void>;
  export function setBackend(backendName: string): Promise<boolean>;

  export const browser: {
    fromPixels(pixels: ImageData | HTMLImageElement | HTMLCanvasElement | HTMLVideoElement, numChannels?: number): Tensor3D;
    toPixels(tensor: Tensor3D | Tensor2D, canvas?: HTMLCanvasElement): Promise<Uint8ClampedArray>;
  };

  export const image: {
    resizeBilinear(images: Tensor3D | Tensor4D, size: [number, number], alignCorners?: boolean): Tensor3D | Tensor4D;
  };

  export function split(x: Tensor | TensorLike, numOrSizeSplits: number[] | number, axis?: number): Tensor[];
  export function stack(tensors: Array<Tensor | TensorLike>, axis?: number): Tensor;
  export function add(a: Tensor | TensorLike, b: Tensor | TensorLike): Tensor;
  export function sub(a: Tensor | TensorLike, b: Tensor | TensorLike): Tensor;
  export function mul(a: Tensor | TensorLike, b: Tensor | TensorLike): Tensor;
  export function div(a: Tensor | TensorLike, b: Tensor | TensorLike): Tensor;
  export function addN(tensors: Array<Tensor | TensorLike>): Tensor;
  export function maximum(a: Tensor | TensorLike, b: Tensor | TensorLike): Tensor;
  export function minimum(a: Tensor | TensorLike, b: Tensor | TensorLike): Tensor;
  export function conv2d(
    x: Tensor3D | Tensor4D,
    filter: Tensor4D,
    strides: number | [number, number],
    pad: 'valid' | 'same'
  ): Tensor3D | Tensor4D;
  export function avgPool(
    x: Tensor3D | Tensor4D,
    filterSize: [number, number] | number,
    strides: [number, number] | number,
    pad: 'valid' | 'same'
  ): Tensor3D | Tensor4D;

  export function buffer(shape: number[], dtype?: DataType, values?: TensorLike): Buffer;
  export interface Buffer {
    set(...args: Array<number>): void;
    toTensor(): Tensor;
  }

  export type TensorContainer = Tensor | number | boolean | string | null | TensorContainerObject | TensorContainerArray;
  export interface TensorContainerObject {
    [key: string]: TensorContainer;
  }
  export interface TensorContainerArray extends Array<TensorContainer> {}

  export function where(condition: Tensor, a: Tensor | TensorLike, b: Tensor | TensorLike): Tensor;
  export function sqrt(x: Tensor | TensorLike): Tensor;
  export function mean(x: Tensor | TensorLike, axis?: number | number[], keepDims?: boolean): Tensor;
  export function square(x: Tensor | TensorLike): Tensor;
  export function tanh(x: Tensor | TensorLike): Tensor;

  export type Scalar = Tensor & {
    shape: [];
    dataSync(): number;
  };
} 