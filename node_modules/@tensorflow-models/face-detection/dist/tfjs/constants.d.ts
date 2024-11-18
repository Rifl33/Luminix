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
import { AnchorConfig, ImageToTensorConfig, TensorsToDetectionsConfig } from '../shared/calculators/interfaces/config_interfaces';
import { MediaPipeFaceDetectorTfjsEstimationConfig, MediaPipeFaceDetectorTfjsModelConfig } from './types';
export declare const DEFAULT_DETECTOR_MODEL_URL_FULL_SPARSE = "https://tfhub.dev/mediapipe/tfjs-model/face_detection/full/1";
export declare const DEFAULT_DETECTOR_MODEL_URL_SHORT = "https://tfhub.dev/mediapipe/tfjs-model/face_detection/short/1";
export declare const SHORT_RANGE_DETECTOR_ANCHOR_CONFIG: AnchorConfig;
export declare const FULL_RANGE_DETECTOR_ANCHOR_CONFIG: AnchorConfig;
export declare const DEFAULT_FACE_DETECTOR_MODEL_CONFIG: MediaPipeFaceDetectorTfjsModelConfig;
export declare const DEFAULT_FACE_DETECTOR_ESTIMATION_CONFIG: MediaPipeFaceDetectorTfjsEstimationConfig;
export declare const SHORT_RANGE_TENSORS_TO_DETECTION_CONFIG: TensorsToDetectionsConfig;
export declare const FULL_RANGE_TENSORS_TO_DETECTION_CONFIG: TensorsToDetectionsConfig;
export declare const DETECTOR_NON_MAX_SUPPRESSION_CONFIG: {
    overlapType: "intersection-over-union";
    minSuppressionThreshold: number;
};
export declare const SHORT_RANGE_IMAGE_TO_TENSOR_CONFIG: ImageToTensorConfig;
export declare const FULL_RANGE_IMAGE_TO_TENSOR_CONFIG: ImageToTensorConfig;
