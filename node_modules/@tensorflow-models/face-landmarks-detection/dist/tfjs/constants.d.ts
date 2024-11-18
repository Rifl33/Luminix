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
import { ImageToTensorConfig, LandmarksRefinementConfig, RectTransformationConfig, TensorsToLandmarksConfig } from '../shared/calculators/interfaces/config_interfaces';
import { MediaPipeFaceMeshTfjsEstimationConfig, MediaPipeFaceMeshTfjsModelConfig } from './types';
export declare const DEFAULT_LANDMARK_MODEL_URL = "https://tfhub.dev/mediapipe/tfjs-model/face_landmarks_detection/face_mesh/1";
export declare const DEFAULT_LANDMARK_MODEL_URL_WITH_ATTENTION = "https://tfhub.dev/mediapipe/tfjs-model/face_landmarks_detection/attention_mesh/1";
export declare const DEFAULT_FACE_MESH_MODEL_CONFIG: MediaPipeFaceMeshTfjsModelConfig;
export declare const DEFAULT_FACE_MESH_ESTIMATION_CONFIG: MediaPipeFaceMeshTfjsEstimationConfig;
export declare const RECT_TRANSFORMATION_CONFIG: RectTransformationConfig;
export declare const LANDMARK_IMAGE_TO_TENSOR_CONFIG: ImageToTensorConfig;
export declare const FACE_PRESENCE_SCORE = 0.5;
export declare const MIN_SIMILARITY_THRESHOLD = 0.5;
export declare const TENSORS_TO_LANDMARKS_MESH_CONFIG: TensorsToLandmarksConfig;
export declare const TENSORS_TO_LANDMARKS_LIPS_CONFIG: TensorsToLandmarksConfig;
export declare const TENSORS_TO_LANDMARKS_EYE_CONFIG: TensorsToLandmarksConfig;
export declare const TENSORS_TO_LANDMARKS_IRIS_CONFIG: TensorsToLandmarksConfig;
export declare const LANDMARKS_REFINEMENT_MESH_CONFIG: LandmarksRefinementConfig;
export declare const LANDMARKS_REFINEMENT_LIPS_CONFIG: LandmarksRefinementConfig;
export declare const LANDMARKS_REFINEMENT_LEFT_EYE_CONFIG: LandmarksRefinementConfig;
export declare const LANDMARKS_REFINEMENT_RIGHT_EYE_CONFIG: LandmarksRefinementConfig;
export declare const LANDMARKS_REFINEMENT_LEFT_IRIS_CONFIG: LandmarksRefinementConfig;
export declare const LANDMARKS_REFINEMENT_RIGHT_IRIS_CONFIG: LandmarksRefinementConfig;
