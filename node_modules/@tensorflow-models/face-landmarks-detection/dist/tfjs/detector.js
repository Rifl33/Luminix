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
exports.load = void 0;
var face_detection_1 = require("@tensorflow-models/face-detection");
var tfconv = require("@tensorflow/tfjs-converter");
var tf = require("@tensorflow/tfjs-core");
var constants_1 = require("../constants");
var association_norm_rect_1 = require("../shared/calculators/association_norm_rect");
var calculate_landmark_projection_1 = require("../shared/calculators/calculate_landmark_projection");
var convert_image_to_tensor_1 = require("../shared/calculators/convert_image_to_tensor");
var detection_to_rect_1 = require("../shared/calculators/detection_to_rect");
var image_utils_1 = require("../shared/calculators/image_utils");
var landmarks_refinement_1 = require("../shared/calculators/landmarks_refinement");
var landmarks_to_detection_1 = require("../shared/calculators/landmarks_to_detection");
var normalized_keypoints_to_keypoints_1 = require("../shared/calculators/normalized_keypoints_to_keypoints");
var tensors_to_landmarks_1 = require("../shared/calculators/tensors_to_landmarks");
var transform_rect_1 = require("../shared/calculators/transform_rect");
var constants = require("./constants");
var detector_utils_1 = require("./detector_utils");
/**
 * MediaPipFaceMesh class.
 */
var MediaPipeFaceMeshTfjsLandmarksDetector = /** @class */ (function () {
    function MediaPipeFaceMeshTfjsLandmarksDetector(detector, landmarkModel, maxFaces, withAttention) {
        this.detector = detector;
        this.landmarkModel = landmarkModel;
        this.maxFaces = maxFaces;
        this.withAttention = withAttention;
        // Store global states.
        this.prevFaceRectsFromLandmarks = null;
    }
    /**
     * Estimates faces for an image or video frame.
     *
     * It returns a single face or multiple faces based on the maxFaces
     * parameter from the `config`.
     *
     * @param image
     * ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement The input
     * image to feed through the network.
     *
     * @param estimationConfig Optional. See
     *     `MediaPipeFaceMeshTfjsEstimationConfig` documentation for detail.
     *
     * @return An array of `Face`s.
     */
    // TF.js implementation of the mediapipe face landmark pipeline.
    // ref graph:
    // https://github.com/google/mediapipe/blob/master/mediapipe/mediapipe/modules/face_landmark/face_landmark_front_cpu.pbtxt
    MediaPipeFaceMeshTfjsLandmarksDetector.prototype.estimateFaces = function (image, estimationConfig) {
        return __awaiter(this, void 0, void 0, function () {
            var config, imageSize, image3d, prevFaceRectsFromLandmarks, faceRectsFromDetections, allFaceDetections, faceDetections, faceRects, faceLandmarks, faces, i, landmarks, keypoints, detection;
            var _this = this;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        config = (0, detector_utils_1.validateEstimationConfig)(estimationConfig);
                        if (image == null) {
                            this.reset();
                            return [2 /*return*/, []];
                        }
                        imageSize = (0, image_utils_1.getImageSize)(image);
                        image3d = tf.tidy(function () {
                            var imageTensor = tf.cast((0, image_utils_1.toImageTensor)(image), 'float32');
                            if (config.flipHorizontal) {
                                var batchAxis = 0;
                                imageTensor = tf.squeeze(tf.image.flipLeftRight(
                                // tslint:disable-next-line: no-unnecessary-type-assertion
                                tf.expandDims(imageTensor, batchAxis)), [batchAxis]);
                            }
                            return imageTensor;
                        });
                        prevFaceRectsFromLandmarks = this.prevFaceRectsFromLandmarks;
                        if (!(config.staticImageMode || prevFaceRectsFromLandmarks == null ||
                            prevFaceRectsFromLandmarks.length < this.maxFaces)) return [3 /*break*/, 2];
                        return [4 /*yield*/, this.detector.detectFaces(image3d, false)];
                    case 1:
                        allFaceDetections = _a.sent();
                        if (allFaceDetections.length === 0) {
                            this.reset();
                            image3d.dispose();
                            return [2 /*return*/, []];
                        }
                        faceDetections = allFaceDetections;
                        // FaceLandmarkFrontCpu: FaceDetectionFrontDetectionToRoi
                        // Calculates region of interest based on face detections, so that can be
                        // used to detect landmarks.
                        faceRectsFromDetections = faceDetections.map(function (detection) {
                            return _this.faceDetectionFrontDetectionToRoi(detection, imageSize);
                        });
                        return [3 /*break*/, 3];
                    case 2:
                        faceRectsFromDetections = [];
                        _a.label = 3;
                    case 3:
                        faceRects = (0, association_norm_rect_1.calculateAssociationNormRect)([faceRectsFromDetections, prevFaceRectsFromLandmarks || []], constants.MIN_SIMILARITY_THRESHOLD);
                        return [4 /*yield*/, Promise.all(faceRects.map(function (faceRect) { return _this.faceLandmark(faceRect, image3d); }))];
                    case 4:
                        faceLandmarks = _a.sent();
                        faces = [];
                        this.prevFaceRectsFromLandmarks = [];
                        for (i = 0; i < faceLandmarks.length; ++i) {
                            landmarks = faceLandmarks[i];
                            if (landmarks == null) {
                                continue;
                            }
                            this.prevFaceRectsFromLandmarks.push(this.faceLandmarksToRoi(landmarks, imageSize));
                            keypoints = (0, normalized_keypoints_to_keypoints_1.normalizedKeypointsToKeypoints)(landmarks, imageSize);
                            // Add keypoint name.
                            if (keypoints != null) {
                                keypoints.forEach(function (keypoint, i) {
                                    var name = constants_1.MEDIAPIPE_FACE_MESH_KEYPOINTS.get(i);
                                    if (name != null) {
                                        keypoint.name = name;
                                    }
                                });
                            }
                            detection = (0, landmarks_to_detection_1.landmarksToDetection)(keypoints);
                            faces.push({ keypoints: keypoints, box: detection.locationData.relativeBoundingBox });
                        }
                        image3d.dispose();
                        return [2 /*return*/, faces];
                }
            });
        });
    };
    MediaPipeFaceMeshTfjsLandmarksDetector.prototype.dispose = function () {
        this.detector.dispose();
        this.landmarkModel.dispose();
    };
    MediaPipeFaceMeshTfjsLandmarksDetector.prototype.reset = function () {
        this.detector.reset();
        this.prevFaceRectsFromLandmarks = null;
    };
    // calculates face ROI from face detection.
    // Subgraph: FaceDetectionFrontDetectionToRoi.
    // ref:
    // https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt
    MediaPipeFaceMeshTfjsLandmarksDetector.prototype.faceDetectionFrontDetectionToRoi = function (detection, imageSize) {
        // Converts results of face detection into a rectangle (normalized by
        // image size) that encloses the face and is rotated such that the line
        // connecting left eye and right eye is aligned with the X-axis of the
        // rectangle.
        // FaceDetectionFrontDetectionToRoi: DetectionsToRectsCalculator.
        var rawRoi = (0, detection_to_rect_1.calculateDetectionsToRects)(detection, 'boundingbox', 'normRect', imageSize, {
            rotationVectorStartKeypointIndex: 0,
            rotationVectorEndKeypointIndex: 1,
            rotationVectorTargetAngleDegree: 0
        });
        // Expands and shifts the rectangle that contains the face so that it's
        // likely to cover the entire face.
        // FaceDetectionFrontDetectionToRoi: RectTransformationCalculation.
        var roi = (0, transform_rect_1.transformNormalizedRect)(rawRoi, imageSize, constants.RECT_TRANSFORMATION_CONFIG);
        return roi;
    };
    // Predict face landmarks.
    // subgraph: FaceLandmarkCpu
    // ref:
    // https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_landmark/face_landmark_cpu.pbtxt
    MediaPipeFaceMeshTfjsLandmarksDetector.prototype.faceLandmark = function (roi, image) {
        return __awaiter(this, void 0, void 0, function () {
            var inputTensors, outputs, outputTensors, faceFlagTensor, landmarkTensors, facePresenceScore, landmarks, _a, faceLandmarks;
            return __generator(this, function (_b) {
                switch (_b.label) {
                    case 0:
                        inputTensors = (0, convert_image_to_tensor_1.convertImageToTensor)(image, constants.LANDMARK_IMAGE_TO_TENSOR_CONFIG, roi).imageTensor;
                        outputs = ['output_faceflag'].concat(this.withAttention ?
                            [
                                'output_mesh_identity', 'output_lips', 'Identity_6:0',
                                'Identity_1:0', 'Identity_2:0', 'Identity_5:0'
                            ] :
                            ['output_mesh']);
                        outputTensors = this.landmarkModel.execute(inputTensors, outputs);
                        faceFlagTensor = outputTensors[0], landmarkTensors = outputTensors.slice(1);
                        return [4 /*yield*/, faceFlagTensor.data()];
                    case 1:
                        facePresenceScore = (_b.sent())[0];
                        // Applies a threshold to the confidence score to determine whether a face
                        // is present.
                        if (facePresenceScore < constants.FACE_PRESENCE_SCORE) {
                            tf.dispose(outputTensors);
                            tf.dispose(inputTensors);
                            return [2 /*return*/, null];
                        }
                        if (!this.withAttention) return [3 /*break*/, 3];
                        return [4 /*yield*/, this.tensorsToFaceLandmarksWithAttention(landmarkTensors)];
                    case 2:
                        _a = _b.sent();
                        return [3 /*break*/, 5];
                    case 3: return [4 /*yield*/, this.tensorsToFaceLandmarks(landmarkTensors)];
                    case 4:
                        _a = _b.sent();
                        _b.label = 5;
                    case 5:
                        landmarks = _a;
                        faceLandmarks = (0, calculate_landmark_projection_1.calculateLandmarkProjection)(landmarks, roi);
                        tf.dispose(outputTensors);
                        tf.dispose(inputTensors);
                        return [2 /*return*/, faceLandmarks];
                }
            });
        });
    };
    // Transform single tensor into 468 facial landmarks.
    // subgraph: TensorsToFaceLandmarks
    // ref:
    // https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_landmark/tensors_to_face_landmarks.pbtxt
    MediaPipeFaceMeshTfjsLandmarksDetector.prototype.tensorsToFaceLandmarks = function (landmarkTensors) {
        return __awaiter(this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                return [2 /*return*/, (0, tensors_to_landmarks_1.tensorsToLandmarks)(landmarkTensors[0], constants.TENSORS_TO_LANDMARKS_MESH_CONFIG)];
            });
        });
    };
    // Transform model output tensors into 478 facial landmarks with refined
    // lips, eyes and irises.
    // subgraph: TensorsToFaceLandmarks
    // ref:
    // https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_landmark/tensors_to_face_landmarks_with_attention.pbtxt
    MediaPipeFaceMeshTfjsLandmarksDetector.prototype.tensorsToFaceLandmarksWithAttention = function (landmarkTensors) {
        return __awaiter(this, void 0, void 0, function () {
            var meshLandmarks, lipsLandmarks, leftEyeLandmarks, rightEyeLandmarks, leftIrisLandmarks, rightIrisLandmarks;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, (0, tensors_to_landmarks_1.tensorsToLandmarks)(landmarkTensors[0], constants.TENSORS_TO_LANDMARKS_MESH_CONFIG)];
                    case 1:
                        meshLandmarks = _a.sent();
                        return [4 /*yield*/, (0, tensors_to_landmarks_1.tensorsToLandmarks)(landmarkTensors[1], constants.TENSORS_TO_LANDMARKS_LIPS_CONFIG)];
                    case 2:
                        lipsLandmarks = _a.sent();
                        return [4 /*yield*/, (0, tensors_to_landmarks_1.tensorsToLandmarks)(landmarkTensors[3], constants.TENSORS_TO_LANDMARKS_EYE_CONFIG)];
                    case 3:
                        leftEyeLandmarks = _a.sent();
                        return [4 /*yield*/, (0, tensors_to_landmarks_1.tensorsToLandmarks)(landmarkTensors[5], constants.TENSORS_TO_LANDMARKS_EYE_CONFIG)];
                    case 4:
                        rightEyeLandmarks = _a.sent();
                        return [4 /*yield*/, (0, tensors_to_landmarks_1.tensorsToLandmarks)(landmarkTensors[4], constants.TENSORS_TO_LANDMARKS_IRIS_CONFIG)];
                    case 5:
                        leftIrisLandmarks = _a.sent();
                        return [4 /*yield*/, (0, tensors_to_landmarks_1.tensorsToLandmarks)(landmarkTensors[2], constants.TENSORS_TO_LANDMARKS_IRIS_CONFIG)];
                    case 6:
                        rightIrisLandmarks = _a.sent();
                        return [2 /*return*/, (0, landmarks_refinement_1.landmarksRefinement)([
                                meshLandmarks, lipsLandmarks, leftEyeLandmarks, rightEyeLandmarks,
                                leftIrisLandmarks, rightIrisLandmarks
                            ], [
                                constants.LANDMARKS_REFINEMENT_MESH_CONFIG,
                                constants.LANDMARKS_REFINEMENT_LIPS_CONFIG,
                                constants.LANDMARKS_REFINEMENT_LEFT_EYE_CONFIG,
                                constants.LANDMARKS_REFINEMENT_RIGHT_EYE_CONFIG,
                                constants.LANDMARKS_REFINEMENT_LEFT_IRIS_CONFIG,
                                constants.LANDMARKS_REFINEMENT_RIGHT_IRIS_CONFIG
                            ])];
                }
            });
        });
    };
    // Calculate face region of interest (ROI) from detections.
    // subgraph: FaceLandmarkLandmarksToRoi
    // ref:
    // https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_landmark/face_landmark_landmarks_to_roi.pbtxt
    MediaPipeFaceMeshTfjsLandmarksDetector.prototype.faceLandmarksToRoi = function (landmarks, imageSize) {
        // Converts face landmarks to a detection that tightly encloses all
        // landmarks.
        // FaceLandmarkLandmarksToRoi: LandmarksToDetectionCalculator.
        var faceDetection = (0, landmarks_to_detection_1.landmarksToDetection)(landmarks);
        // Converts the face detection into a rectangle (normalized by image size)
        // that encloses the face and is rotated such that the line connecting
        // left side of the left eye and right side of the right eye is aligned
        // with the X-axis of the rectangle.
        // FaceLandmarkLandmarksToRoi: DetectionsToRectsCalculator
        var faceRectFromLandmarks = (0, detection_to_rect_1.calculateDetectionsToRects)(faceDetection, 'boundingbox', 'normRect', imageSize, {
            rotationVectorStartKeypointIndex: 33,
            rotationVectorEndKeypointIndex: 263,
            rotationVectorTargetAngleDegree: 0
        });
        // Expands the face rectangle so that in the next video image it's likely
        // to still contain the face even with some motion.
        // FaceLandmarkLandmarksToRoi: RectTransformationCalculator.
        // TODO: `squareLong` in the config should be set to false in MediaPipe code
        // but is not due to a bug in their processing. Once fixed on their end,
        // split RECT_TRANSFORMATION_CONFIG into separate detector and landmark
        // configs, with landmark's config's `squareLong` set to false.
        var roi = (0, transform_rect_1.transformNormalizedRect)(faceRectFromLandmarks, imageSize, constants.RECT_TRANSFORMATION_CONFIG);
        return roi;
    };
    return MediaPipeFaceMeshTfjsLandmarksDetector;
}());
/**
 * Loads the MediaPipeFaceMesh model.
 *
 * @param modelConfig ModelConfig object that contains parameters for
 * the MediaPipeFaceMesh loading process. Please find more details of each
 * parameters in the documentation of the `MediaPipeFaceMeshTfjsModelConfig`
 * interface.
 */
function load(modelConfig) {
    return __awaiter(this, void 0, void 0, function () {
        var config, landmarkFromTFHub, landmarkModel, detector;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    config = (0, detector_utils_1.validateModelConfig)(modelConfig);
                    landmarkFromTFHub = typeof config.landmarkModelUrl === 'string' &&
                        (config.landmarkModelUrl.indexOf('https://tfhub.dev') > -1);
                    return [4 /*yield*/, tfconv.loadGraphModel(config.landmarkModelUrl, { fromTFHub: landmarkFromTFHub })];
                case 1:
                    landmarkModel = _a.sent();
                    return [4 /*yield*/, (0, face_detection_1.createDetector)(face_detection_1.SupportedModels.MediaPipeFaceDetector, {
                            modelType: 'short',
                            maxFaces: config.maxFaces,
                            detectorModelUrl: config.detectorModelUrl,
                            runtime: config.runtime
                        })];
                case 2:
                    detector = _a.sent();
                    return [2 /*return*/, new MediaPipeFaceMeshTfjsLandmarksDetector(detector, landmarkModel, config.maxFaces, config.refineLandmarks)];
            }
        });
    });
}
exports.load = load;
//# sourceMappingURL=detector.js.map