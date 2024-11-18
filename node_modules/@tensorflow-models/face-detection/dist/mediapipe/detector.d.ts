import { FaceDetector } from '../face_detector';
import { Face, FaceDetectorInput } from '../types';
import { MediaPipeFaceDetectorMediaPipeEstimationConfig, MediaPipeFaceDetectorMediaPipeModelConfig } from './types';
/**
 * MediaPipe detector class.
 */
export declare class MediaPipeFaceDetectorMediaPipe implements FaceDetector {
    private readonly faceDetectorSolution;
    private width;
    private height;
    private faces;
    private selfieMode;
    constructor(config: MediaPipeFaceDetectorMediaPipeModelConfig);
    private normalizedToAbsolute;
    /**
     * Estimates faces for an image or video frame.
     *
     * It returns a single face or multiple faces based on the maxFaces
     * parameter passed to the constructor of the class.
     *
     * @param input
     * ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement The input
     * image to feed through the network.
     *
     * @param config Optional.
     *       flipHorizontal: Optional. Default to false. When image data comes
     *       from camera, the result has to flip horizontally.
     *
     *       staticImageMode: Optional. Defaults to false. Currently unused in
     * this implementation. Image input types are assumed to be static images, and
     * video inputs are assumed to be non static images.
     *
     * @return An array of `Face`s.
     */
    estimateFaces(input: FaceDetectorInput, estimationConfig?: MediaPipeFaceDetectorMediaPipeEstimationConfig): Promise<Face[]>;
    dispose(): void;
    reset(): void;
    initialize(): Promise<void>;
}
/**
 * Loads the MediaPipe solution.
 *
 * @param modelConfig An object that contains parameters for
 * the MediaPipeFaceDetector loading process. Please find more details of each
 * parameters in the documentation of the
 * `MediaPipeFaceDetectorMediaPipeModelConfig` interface.
 */
export declare function load(modelConfig: MediaPipeFaceDetectorMediaPipeModelConfig): Promise<FaceDetector>;
