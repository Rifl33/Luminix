import { FaceLandmarksDetector } from '../face_landmarks_detector';
import { MediaPipeFaceMeshMediaPipeModelConfig } from './types';
/**
 * Loads the MediaPipe solution.
 *
 * @param modelConfig An object that contains parameters for
 * the MediaPipeFaceMesh loading process. Please find more details of each
 * parameters in the documentation of the
 * `MediaPipeFaceMeshMediaPipeModelConfig` interface.
 */
export declare function load(modelConfig: MediaPipeFaceMeshMediaPipeModelConfig): Promise<FaceLandmarksDetector>;
