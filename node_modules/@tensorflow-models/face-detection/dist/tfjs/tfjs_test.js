"use strict";
/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
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
var tf = require("@tensorflow/tfjs-core");
// tslint:disable-next-line: no-imports-from-dist
var jasmine_util_1 = require("@tensorflow/tfjs-core/dist/jasmine_util");
// tslint:disable-next-line: no-imports-from-dist
var test_util_1 = require("@tensorflow/tfjs-core/dist/test_util");
var faceDetection = require("../index");
var mediapipe_test_1 = require("../mediapipe/mediapipe_test");
var test_util_2 = require("../shared/test_util");
var TFJS_MODEL_CONFIG = {
    runtime: 'tfjs',
    maxFaces: 1
};
// Measured in pixels.
var EPSILON_IMAGE = 10;
(0, jasmine_util_1.describeWithFlags)('TFJS FaceDetector ', jasmine_util_1.ALL_ENVS, function () {
    var timeout;
    beforeAll(function () {
        timeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
        jasmine.DEFAULT_TIMEOUT_INTERVAL = 120000; // 2mins
    });
    afterAll(function () {
        jasmine.DEFAULT_TIMEOUT_INTERVAL = timeout;
    });
    function expectTFJSFaceDetector(modelType) {
        return __awaiter(this, void 0, void 0, function () {
            var startTensors, detector, input, beforeTensors;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        startTensors = tf.memory().numTensors;
                        return [4 /*yield*/, faceDetection.createDetector(faceDetection.SupportedModels.MediaPipeFaceDetector, __assign(__assign({}, TFJS_MODEL_CONFIG), { modelType: modelType }))];
                    case 1:
                        detector = _a.sent();
                        input = tf.zeros([128, 128, 3]);
                        beforeTensors = tf.memory().numTensors;
                        return [4 /*yield*/, detector.estimateFaces(input)];
                    case 2:
                        _a.sent();
                        expect(tf.memory().numTensors).toEqual(beforeTensors);
                        detector.dispose();
                        input.dispose();
                        expect(tf.memory().numTensors).toEqual(startTensors);
                        return [2 /*return*/];
                }
            });
        });
    }
    it('short range detectFaces does not leak memory.', function () { return __awaiter(void 0, void 0, void 0, function () {
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0: return [4 /*yield*/, expectTFJSFaceDetector('short')];
                case 1:
                    _a.sent();
                    return [2 /*return*/];
            }
        });
    }); });
    it('full range detectFaces does not leak memory.', function () { return __awaiter(void 0, void 0, void 0, function () {
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0: return [4 /*yield*/, expectTFJSFaceDetector('full')];
                case 1:
                    _a.sent();
                    return [2 /*return*/];
            }
        });
    }); });
});
(0, jasmine_util_1.describeWithFlags)('TFJS FaceDetector static image ', jasmine_util_1.BROWSER_ENVS, function () {
    var image;
    var timeout;
    beforeAll(function () { return __awaiter(void 0, void 0, void 0, function () {
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    timeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
                    jasmine.DEFAULT_TIMEOUT_INTERVAL = 120000; // 2mins
                    return [4 /*yield*/, (0, test_util_2.loadImage)('portrait.jpg', 820, 1024)];
                case 1:
                    image = _a.sent();
                    return [2 /*return*/];
            }
        });
    }); });
    afterAll(function () {
        jasmine.DEFAULT_TIMEOUT_INTERVAL = timeout;
    });
    function expectTFJSFaceDetector(modelType) {
        return __awaiter(this, void 0, void 0, function () {
            var model, detector;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        model = faceDetection.SupportedModels.MediaPipeFaceDetector;
                        return [4 /*yield*/, faceDetection.createDetector(model, __assign(__assign({}, TFJS_MODEL_CONFIG), { modelType: modelType }))];
                    case 1:
                        detector = _a.sent();
                        return [2 /*return*/, (0, mediapipe_test_1.expectFaceDetector)(detector, image, modelType)];
                }
            });
        });
    }
    it('short range.', function () { return __awaiter(void 0, void 0, void 0, function () {
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0: return [4 /*yield*/, expectTFJSFaceDetector('short')];
                case 1:
                    _a.sent();
                    return [2 /*return*/];
            }
        });
    }); });
    it('full range.', function () { return __awaiter(void 0, void 0, void 0, function () {
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0: return [4 /*yield*/, expectTFJSFaceDetector('full')];
                case 1:
                    _a.sent();
                    return [2 /*return*/];
            }
        });
    }); });
    it('TFJS and Mediapipe backends match.', function () { return __awaiter(void 0, void 0, void 0, function () {
        var model, tfjsResults, mediapipeResults, tfjsKeypoints, mediapipeKeypoints, i, _i, _a, key;
        return __generator(this, function (_b) {
            switch (_b.label) {
                case 0:
                    model = faceDetection.SupportedModels.MediaPipeFaceDetector;
                    return [4 /*yield*/, faceDetection.createDetector(model, __assign({}, TFJS_MODEL_CONFIG))
                            .then(function (detector) { return detector.estimateFaces(image); })];
                case 1:
                    tfjsResults = _b.sent();
                    return [4 /*yield*/, faceDetection.createDetector(model, __assign({}, mediapipe_test_1.MEDIAPIPE_MODEL_CONFIG))
                            .then(function (detector) { return __awaiter(void 0, void 0, void 0, function () {
                            return __generator(this, function (_a) {
                                switch (_a.label) {
                                    case 0: return [4 /*yield*/, detector.estimateFaces(image)];
                                    case 1:
                                        _a.sent();
                                        // Initialize model.
                                        return [2 /*return*/, detector.estimateFaces(image)];
                                }
                            });
                        }); })];
                case 2:
                    mediapipeResults = _b.sent();
                    expect(tfjsResults.length).toBe(mediapipeResults.length);
                    tfjsKeypoints = tfjsResults
                        .map(function (face) { return face.keypoints.map(function (keypoint) { return [keypoint.x, keypoint.y,
                        keypoint.name]; }); })
                        .flat();
                    mediapipeKeypoints = mediapipeResults
                        .map(function (face) { return face.keypoints.map(function (keypoint) { return [keypoint.x, keypoint.y,
                        keypoint.name]; }); })
                        .flat();
                    (0, test_util_1.expectArraysClose)(tfjsKeypoints.map(function (keypoint) { return [keypoint[0], keypoint[1]]; }), mediapipeKeypoints.map(function (keypoint) { return [keypoint[0], keypoint[1]]; }), EPSILON_IMAGE);
                    (0, test_util_1.expectArraysEqual)(tfjsKeypoints.map(function (keypoint) { return keypoint[2]; }), mediapipeKeypoints.map(function (keypoint) { return keypoint[2]; }));
                    for (i = 0; i < tfjsResults.length; i++) {
                        for (_i = 0, _a = ['height', 'width', 'xMax', 'xMin', 'yMax', 'yMin']; _i < _a.length; _i++) {
                            key = _a[_i];
                            (0, test_util_1.expectNumbersClose)(tfjsResults[i].box[key], mediapipeResults[i].box[key], EPSILON_IMAGE);
                        }
                    }
                    return [2 /*return*/];
            }
        });
    }); });
});
//# sourceMappingURL=tfjs_test.js.map