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
export declare const MEDIAPIPE_FACE_MESH_NUM_KEYPOINTS = 468;
export declare const MEDIAPIPE_FACE_MESH_NUM_KEYPOINTS_WITH_IRISES = 478;
declare type PairArray = Array<[number, number]>;
export declare const MEDIAPIPE_FACE_MESH_KEYPOINTS_BY_CONTOUR: {
    lips: number[];
    leftEye: number[];
    leftEyebrow: number[];
    leftIris: number[];
    rightEye: number[];
    rightEyebrow: number[];
    rightIris: number[];
    faceOval: number[];
};
export declare const MEDIAPIPE_FACE_MESH_CONNECTED_KEYPOINTS_PAIRS: PairArray;
export declare const MEDIAPIPE_FACE_MESH_KEYPOINTS: Map<number, string>;
export {};
