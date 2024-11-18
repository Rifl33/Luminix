"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.FULL_RANGE_IMAGE_TO_TENSOR_CONFIG = exports.SHORT_RANGE_IMAGE_TO_TENSOR_CONFIG = exports.DETECTOR_NON_MAX_SUPPRESSION_CONFIG = exports.FULL_RANGE_TENSORS_TO_DETECTION_CONFIG = exports.SHORT_RANGE_TENSORS_TO_DETECTION_CONFIG = exports.DEFAULT_FACE_DETECTOR_ESTIMATION_CONFIG = exports.DEFAULT_FACE_DETECTOR_MODEL_CONFIG = exports.FULL_RANGE_DETECTOR_ANCHOR_CONFIG = exports.SHORT_RANGE_DETECTOR_ANCHOR_CONFIG = exports.DEFAULT_DETECTOR_MODEL_URL_SHORT = exports.DEFAULT_DETECTOR_MODEL_URL_FULL_SPARSE = void 0;
// Non-sparse full model is not currently used but is available if needed for
// future use cases.
exports.DEFAULT_DETECTOR_MODEL_URL_FULL_SPARSE = 'https://tfhub.dev/mediapipe/tfjs-model/face_detection/full/1';
exports.DEFAULT_DETECTOR_MODEL_URL_SHORT = 'https://tfhub.dev/mediapipe/tfjs-model/face_detection/short/1';
exports.SHORT_RANGE_DETECTOR_ANCHOR_CONFIG = {
    reduceBoxesInLowestLayer: false,
    interpolatedScaleAspectRatio: 1.0,
    featureMapHeight: [],
    featureMapWidth: [],
    numLayers: 4,
    minScale: 0.1484375,
    maxScale: 0.75,
    inputSizeHeight: 128,
    inputSizeWidth: 128,
    anchorOffsetX: 0.5,
    anchorOffsetY: 0.5,
    strides: [8, 16, 16, 16],
    aspectRatios: [1.0],
    fixedAnchorSize: true
};
exports.FULL_RANGE_DETECTOR_ANCHOR_CONFIG = {
    reduceBoxesInLowestLayer: false,
    interpolatedScaleAspectRatio: 0.0,
    featureMapHeight: [],
    featureMapWidth: [],
    numLayers: 1,
    minScale: 0.1484375,
    maxScale: 0.75,
    inputSizeHeight: 192,
    inputSizeWidth: 192,
    anchorOffsetX: 0.5,
    anchorOffsetY: 0.5,
    strides: [4],
    aspectRatios: [1.0],
    fixedAnchorSize: true
};
exports.DEFAULT_FACE_DETECTOR_MODEL_CONFIG = {
    runtime: 'tfjs',
    modelType: 'short',
    maxFaces: 1,
    detectorModelUrl: exports.DEFAULT_DETECTOR_MODEL_URL_SHORT,
};
exports.DEFAULT_FACE_DETECTOR_ESTIMATION_CONFIG = {
    flipHorizontal: false,
};
exports.SHORT_RANGE_TENSORS_TO_DETECTION_CONFIG = {
    applyExponentialOnBoxSize: false,
    flipVertically: false,
    ignoreClasses: [],
    numClasses: 1,
    numBoxes: 896,
    numCoords: 16,
    boxCoordOffset: 0,
    keypointCoordOffset: 4,
    numKeypoints: 6,
    numValuesPerKeypoint: 2,
    sigmoidScore: true,
    scoreClippingThresh: 100.0,
    reverseOutputOrder: true,
    xScale: 128.0,
    yScale: 128.0,
    hScale: 128.0,
    wScale: 128.0,
    minScoreThresh: 0.5
};
exports.FULL_RANGE_TENSORS_TO_DETECTION_CONFIG = {
    applyExponentialOnBoxSize: false,
    flipVertically: false,
    ignoreClasses: [],
    numClasses: 1,
    numBoxes: 2304,
    numCoords: 16,
    boxCoordOffset: 0,
    keypointCoordOffset: 4,
    numKeypoints: 6,
    numValuesPerKeypoint: 2,
    sigmoidScore: true,
    scoreClippingThresh: 100.0,
    reverseOutputOrder: true,
    xScale: 192.0,
    yScale: 192.0,
    hScale: 192.0,
    wScale: 192.0,
    minScoreThresh: 0.6
};
exports.DETECTOR_NON_MAX_SUPPRESSION_CONFIG = {
    overlapType: 'intersection-over-union',
    minSuppressionThreshold: 0.3
};
exports.SHORT_RANGE_IMAGE_TO_TENSOR_CONFIG = {
    outputTensorSize: { width: 128, height: 128 },
    keepAspectRatio: true,
    outputTensorFloatRange: [-1, 1],
    borderMode: 'zero'
};
exports.FULL_RANGE_IMAGE_TO_TENSOR_CONFIG = {
    outputTensorSize: { width: 192, height: 192 },
    keepAspectRatio: true,
    outputTensorFloatRange: [-1, 1],
    borderMode: 'zero'
};
//# sourceMappingURL=constants.js.map