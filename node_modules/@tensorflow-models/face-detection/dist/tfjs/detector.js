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
var __assign = (this && this.__assign) || function () {
    __assign = Object.assign || function(t) {
        for (var s, i = 1, n = arguments.length; i < n; i++) {
            s = arguments[i];
            for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p))
                t[p] = s[p];
        }
        return t;
    };
    return __assign.apply(this, arguments);
};
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.load = exports.MediaPipeFaceDetectorTfjs = void 0;
var tfconv = require("@tensorflow/tfjs-converter");
var tf = require("@tensorflow/tfjs-core");
var constants_1 = require("../constants");
var convert_image_to_tensor_1 = require("../shared/calculators/convert_image_to_tensor");
var create_ssd_anchors_1 = require("../shared/calculators/create_ssd_anchors");
var detection_projection_1 = require("../shared/calculators/detection_projection");
var detector_result_1 = require("../shared/calculators/detector_result");
var image_utils_1 = require("../shared/calculators/image_utils");
var non_max_suppression_1 = require("../shared/calculators/non_max_suppression");
var tensors_to_detections_1 = require("../shared/calculators/tensors_to_detections");
var constants = require("./constants");
var detector_utils_1 = require("./detector_utils");
var MediaPipeFaceDetectorTfjs = /** @class */ (function () {
    function MediaPipeFaceDetectorTfjs(detectorModelType, detectorModel, maxFaces) {
        this.detectorModel = detectorModel;
        this.maxFaces = maxFaces;
        if (detectorModelType === 'full') {
            this.imageToTensorConfig = constants.FULL_RANGE_IMAGE_TO_TENSOR_CONFIG;
            this.tensorsToDetectionConfig =
                constants.FULL_RANGE_TENSORS_TO_DETECTION_CONFIG;
            this.anchors =
                (0, create_ssd_anchors_1.createSsdAnchors)(constants.FULL_RANGE_DETECTOR_ANCHOR_CONFIG);
        }
        else {
            this.imageToTensorConfig = constants.SHORT_RANGE_IMAGE_TO_TENSOR_CONFIG;
            this.tensorsToDetectionConfig =
                constants.SHORT_RANGE_TENSORS_TO_DETECTION_CONFIG;
            this.anchors =
                (0, create_ssd_anchors_1.createSsdAnchors)(constants.SHORT_RANGE_DETECTOR_ANCHOR_CONFIG);
        }
        var anchorW = tf.tensor1d(this.anchors.map(function (a) { return a.width; }));
        var anchorH = tf.tensor1d(this.anchors.map(function (a) { return a.height; }));
        var anchorX = tf.tensor1d(this.anchors.map(function (a) { return a.xCenter; }));
        var anchorY = tf.tensor1d(this.anchors.map(function (a) { return a.yCenter; }));
        this.anchorTensor = { x: anchorX, y: anchorY, w: anchorW, h: anchorH };
    }
    MediaPipeFaceDetectorTfjs.prototype.dispose = function () {
        this.detectorModel.dispose();
        tf.dispose([
            this.anchorTensor.x, this.anchorTensor.y, this.anchorTensor.w,
            this.anchorTensor.h
        ]);
    };
    MediaPipeFaceDetectorTfjs.prototype.reset = function () { };
    // Detects faces.
    // Subgraph: FaceDetectionShort/FullRangeCpu.
    // ref:
    // https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_detection/face_detection_short_range_cpu.pbtxt
    // https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_detection/face_detection_full_range_cpu.pbtxt
    MediaPipeFaceDetectorTfjs.prototype.detectFaces = function (image, flipHorizontal) {
        if (flipHorizontal === void 0) { flipHorizontal = false; }
        return __awaiter(this, void 0, void 0, function () {
            var image3d, _a, inputTensors, transformMatrix, detectionResult, _b, boxes, logits, unfilteredDetections, filteredDetections, detections;
            return __generator(this, function (_c) {
                switch (_c.label) {
                    case 0:
                        if (image == null) {
                            this.reset();
                            return [2 /*return*/, []];
                        }
                        image3d = tf.tidy(function () {
                            var imageTensor = tf.cast((0, image_utils_1.toImageTensor)(image), 'float32');
                            if (flipHorizontal) {
                                var batchAxis = 0;
                                imageTensor = tf.squeeze(tf.image.flipLeftRight(
                                // tslint:disable-next-line: no-unnecessary-type-assertion
                                tf.expandDims(imageTensor, batchAxis)), [batchAxis]);
                            }
                            return imageTensor;
                        });
                        _a = (0, convert_image_to_tensor_1.convertImageToTensor)(image3d, this.imageToTensorConfig), inputTensors = _a.imageTensor, transformMatrix = _a.transformationMatrix;
                        detectionResult = this.detectorModel.execute(inputTensors, 'Identity:0');
                        _b = (0, detector_result_1.detectorResult)(detectionResult), boxes = _b.boxes, logits = _b.logits;
                        return [4 /*yield*/, (0, tensors_to_detections_1.tensorsToDetections)([logits, boxes], this.anchorTensor, this.tensorsToDetectionConfig)];
                    case 1:
                        unfilteredDetections = _c.sent();
                        if (unfilteredDetections.length === 0) {
                            tf.dispose([image3d, inputTensors, detectionResult, logits, boxes]);
                            return [2 /*return*/, unfilteredDetections];
                        }
                        return [4 /*yield*/, (0, non_max_suppression_1.nonMaxSuppression)(unfilteredDetections, this.maxFaces, constants.DETECTOR_NON_MAX_SUPPRESSION_CONFIG.minSuppressionThreshold, constants.DETECTOR_NON_MAX_SUPPRESSION_CONFIG.overlapType)];
                    case 2:
                        filteredDetections = _c.sent();
                        detections = 
                        // FaceDetectionShortRangeModelCpu:
                        // DetectionProjectionCalculator
                        (0, detection_projection_1.detectionProjection)(filteredDetections, transformMatrix);
                        tf.dispose([image3d, inputTensors, detectionResult, logits, boxes]);
                        return [2 /*return*/, detections];
                }
            });
        });
    };
    MediaPipeFaceDetectorTfjs.prototype.estimateFaces = function (image, estimationConfig) {
        return __awaiter(this, void 0, void 0, function () {
            var imageSize, flipHorizontal;
            return __generator(this, function (_a) {
                imageSize = (0, image_utils_1.getImageSize)(image);
                flipHorizontal = estimationConfig ? estimationConfig.flipHorizontal : false;
                return [2 /*return*/, this.detectFaces(image, flipHorizontal)
                        .then(function (detections) { return detections.map(function (detection) {
                        var keypoints = detection.locationData.relativeKeypoints.map(function (keypoint, i) { return (__assign(__assign({}, keypoint), { x: keypoint.x * imageSize.width, y: keypoint.y * imageSize.height, name: constants_1.MEDIAPIPE_FACE_DETECTOR_KEYPOINTS[i] })); });
                        var box = detection.locationData.relativeBoundingBox;
                        for (var _i = 0, _a = ['width', 'xMax', 'xMin']; _i < _a.length; _i++) {
                            var key = _a[_i];
                            box[key] *= imageSize.width;
                        }
                        for (var _b = 0, _c = ['height', 'yMax', 'yMin']; _b < _c.length; _b++) {
                            var key = _c[_b];
                            box[key] *= imageSize.height;
                        }
                        return { keypoints: keypoints, box: box };
                    }); })];
            });
        });
    };
    return MediaPipeFaceDetectorTfjs;
}());
exports.MediaPipeFaceDetectorTfjs = MediaPipeFaceDetectorTfjs;
/**
 * Loads the MediaPipeFaceDetector model.
 *
 * @param modelConfig ModelConfig object that contains parameters for
 * the MediaPipeFaceDetector loading process. Please find more details of each
 * parameters in the documentation of the `MediaPipeFaceDetectorTfjsModelConfig`
 * interface.
 */
function load(modelConfig) {
    return __awaiter(this, void 0, void 0, function () {
        var config, detectorFromTFHub, detectorModel;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    config = (0, detector_utils_1.validateModelConfig)(modelConfig);
                    detectorFromTFHub = typeof config.detectorModelUrl === 'string' &&
                        (config.detectorModelUrl.indexOf('https://tfhub.dev') > -1);
                    return [4 /*yield*/, tfconv.loadGraphModel(config.detectorModelUrl, { fromTFHub: detectorFromTFHub })];
                case 1:
                    detectorModel = _a.sent();
                    return [2 /*return*/, new MediaPipeFaceDetectorTfjs(config.modelType, detectorModel, config.maxFaces)];
            }
        });
    });
}
exports.load = load;
//# sourceMappingURL=detector.js.map