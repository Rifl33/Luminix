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
exports.expectFaceMesh = exports.MEDIAPIPE_MODEL_CONFIG = void 0;
// tslint:disable-next-line: no-imports-from-dist
var jasmine_util_1 = require("@tensorflow/tfjs-core/dist/jasmine_util");
// tslint:disable-next-line: no-imports-from-dist
var test_util_1 = require("@tensorflow/tfjs-core/dist/test_util");
var constants_1 = require("../constants");
var faceLandmarksDetection = require("../index");
var test_util_2 = require("../shared/test_util");
exports.MEDIAPIPE_MODEL_CONFIG = {
    runtime: 'mediapipe',
    solutionPath: 'base/node_modules/@mediapipe/face_mesh'
};
// ref:
// https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_test.py
// Measured in pixels.
var EPSILON_IMAGE = 5;
var EYE_INDICES_TO_LANDMARKS = [
    [33, [345, 178]],
    [7, [348, 179]],
    [163, [352, 178]],
    [144, [357, 179]],
    [145, [365, 179]],
    [153, [371, 179]],
    [154, [378, 178]],
    [155, [381, 177]],
    [133, [383, 177]],
    [246, [347, 175]],
    [161, [350, 174]],
    [160, [355, 172]],
    [159, [362, 170]],
    [158, [368, 171]],
    [157, [375, 172]],
    [173, [380, 175]],
    [263, [467, 176]],
    [249, [464, 177]],
    [390, [460, 177]],
    [373, [455, 178]],
    [374, [448, 179]],
    [380, [441, 179]],
    [381, [435, 178]],
    [382, [432, 177]],
    [362, [430, 177]],
    [466, [465, 175]],
    [388, [462, 173]],
    [387, [457, 171]],
    [386, [450, 170]],
    [385, [444, 171]],
    [384, [437, 172]],
    [398, [432, 175]] //
];
var IRIS_INDICES_TO_LANDMARKS = [
    [468, [362, 175]],
    [469, [371, 175]],
    [470, [362, 167]],
    [471, [354, 175]],
    [472, [363, 182]],
    [473, [449, 174]],
    [474, [458, 174]],
    [475, [449, 167]],
    [476, [440, 174]],
    [477, [449, 181]] //
];
var EXPECTED_BOX = {
    xMin: 305,
    xMax: 504,
    yMin: 103,
    yMax: 347,
    width: 199,
    height: 244
};
function expectFaceMesh(detector, image, staticImageMode, refineLandmarks, numFrames, epsilon) {
    return __awaiter(this, void 0, void 0, function () {
        var i, result, box, keypoints, _i, EYE_INDICES_TO_LANDMARKS_1, _a, eyeIdx, gtLds, _b, IRIS_INDICES_TO_LANDMARKS_1, _c, irisIdx, gtLds;
        return __generator(this, function (_d) {
            switch (_d.label) {
                case 0:
                    i = 0;
                    _d.label = 1;
                case 1:
                    if (!(i < numFrames)) return [3 /*break*/, 4];
                    return [4 /*yield*/, detector.estimateFaces(image, { staticImageMode: staticImageMode })];
                case 2:
                    result = _d.sent();
                    expect(result.length).toBe(1);
                    box = result[0].box;
                    (0, test_util_1.expectNumbersClose)(box.xMin, EXPECTED_BOX.xMin, EPSILON_IMAGE);
                    (0, test_util_1.expectNumbersClose)(box.xMax, EXPECTED_BOX.xMax, EPSILON_IMAGE);
                    (0, test_util_1.expectNumbersClose)(box.yMin, EXPECTED_BOX.yMin, EPSILON_IMAGE);
                    (0, test_util_1.expectNumbersClose)(box.yMax, EXPECTED_BOX.yMax, EPSILON_IMAGE);
                    (0, test_util_1.expectNumbersClose)(box.width, EXPECTED_BOX.width, EPSILON_IMAGE);
                    (0, test_util_1.expectNumbersClose)(box.height, EXPECTED_BOX.height, EPSILON_IMAGE);
                    keypoints = result[0].keypoints.map(function (keypoint) { return [keypoint.x, keypoint.y]; });
                    expect(keypoints.length)
                        .toBe(refineLandmarks ? constants_1.MEDIAPIPE_FACE_MESH_NUM_KEYPOINTS_WITH_IRISES :
                        constants_1.MEDIAPIPE_FACE_MESH_NUM_KEYPOINTS);
                    for (_i = 0, EYE_INDICES_TO_LANDMARKS_1 = EYE_INDICES_TO_LANDMARKS; _i < EYE_INDICES_TO_LANDMARKS_1.length; _i++) {
                        _a = EYE_INDICES_TO_LANDMARKS_1[_i], eyeIdx = _a[0], gtLds = _a[1];
                        (0, test_util_1.expectArraysClose)(keypoints[eyeIdx], gtLds, epsilon);
                    }
                    if (refineLandmarks) {
                        for (_b = 0, IRIS_INDICES_TO_LANDMARKS_1 = IRIS_INDICES_TO_LANDMARKS; _b < IRIS_INDICES_TO_LANDMARKS_1.length; _b++) {
                            _c = IRIS_INDICES_TO_LANDMARKS_1[_b], irisIdx = _c[0], gtLds = _c[1];
                            (0, test_util_1.expectArraysClose)(keypoints[irisIdx], gtLds, epsilon);
                        }
                    }
                    _d.label = 3;
                case 3:
                    ++i;
                    return [3 /*break*/, 1];
                case 4: return [2 /*return*/];
            }
        });
    });
}
exports.expectFaceMesh = expectFaceMesh;
(0, jasmine_util_1.describeWithFlags)('MediaPipe FaceMesh ', jasmine_util_1.BROWSER_ENVS, function () {
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
    function expectMediaPipeFaceMesh(image, staticImageMode, refineLandmarks, numFrames) {
        return __awaiter(this, void 0, void 0, function () {
            var model, detector;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
                        return [4 /*yield*/, faceLandmarksDetection.createDetector(model, __assign(__assign({}, exports.MEDIAPIPE_MODEL_CONFIG), { refineLandmarks: refineLandmarks }))];
                    case 1:
                        detector = _a.sent();
                        return [4 /*yield*/, expectFaceMesh(detector, image, staticImageMode, refineLandmarks, numFrames, EPSILON_IMAGE)];
                    case 2:
                        _a.sent();
                        return [2 /*return*/];
                }
            });
        });
    }
    it('static image mode no attention.', function () { return __awaiter(void 0, void 0, void 0, function () {
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0: return [4 /*yield*/, expectMediaPipeFaceMesh(image, true, false, 5)];
                case 1:
                    _a.sent();
                    return [2 /*return*/];
            }
        });
    }); });
    it('static image mode with attention.', function () { return __awaiter(void 0, void 0, void 0, function () {
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0: return [4 /*yield*/, expectMediaPipeFaceMesh(image, true, true, 5)];
                case 1:
                    _a.sent();
                    return [2 /*return*/];
            }
        });
    }); });
    it('streaming mode no attention.', function () { return __awaiter(void 0, void 0, void 0, function () {
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0: return [4 /*yield*/, expectMediaPipeFaceMesh(image, false, false, 10)];
                case 1:
                    _a.sent();
                    return [2 /*return*/];
            }
        });
    }); });
    it('streaming mode with attention.', function () { return __awaiter(void 0, void 0, void 0, function () {
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0: return [4 /*yield*/, expectMediaPipeFaceMesh(image, false, true, 10)];
                case 1:
                    _a.sent();
                    return [2 /*return*/];
            }
        });
    }); });
});
//# sourceMappingURL=mediapipe_test.js.map