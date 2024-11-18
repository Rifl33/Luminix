/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import * as tfconv from '@tensorflow/tfjs-converter';
import { FaceDetector } from '../face_detector';
import { Detection } from '../shared/calculators/interfaces/shape_interfaces';
import { Face, FaceDetectorInput } from '../types';
import { MediaPipeFaceDetectorTfjsEstimationConfig, MediaPipeFaceDetectorTfjsModelConfig } from './types';
export declare class MediaPipeFaceDetectorTfjs implements FaceDetector {
    private readonly detectorModel;
    private readonly maxFaces;
    private readonly imageToTensorConfig;
    private readonly tensorsToDetectionConfig;
    private readonly anchors;
    private readonly anchorTensor;
    constructor(detectorModelType: 'short' | 'full', detectorModel: tfconv.GraphModel, maxFaces: number);
    dispose(): void;
    reset(): void;
    detectFaces(image: FaceDetectorInput, flipHorizontal?: boolean): Promise<Detection[]>;
    estimateFaces(image: FaceDetectorInput, estimationConfig?: MediaPipeFaceDetectorTfjsEstimationConfig): Promise<Face[]>;
}
/**
 * Loads the MediaPipeFaceDetector model.
 *
 * @param modelConfig ModelConfig object that contains parameters for
 * the MediaPipeFaceDetector loading process. Please find more details of each
 * parameters in the documentation of the `MediaPipeFaceDetectorTfjsModelConfig`
 * interface.
 */
export declare function load(modelConfig: MediaPipeFaceDetectorTfjsModelConfig): Promise<MediaPipeFaceDetectorTfjs>;
