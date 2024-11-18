"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.getAdjacentPairs = exports.getKeypointIndexByContour = void 0;
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
var constants = require("./constants");
var types_1 = require("./types");
function getKeypointIndexByContour(model) {
    switch (model) {
        case types_1.SupportedModels.MediaPipeFaceMesh:
            return constants.MEDIAPIPE_FACE_MESH_KEYPOINTS_BY_CONTOUR;
        default:
            throw new Error("Model " + model + " is not supported.");
    }
}
exports.getKeypointIndexByContour = getKeypointIndexByContour;
function getAdjacentPairs(model) {
    switch (model) {
        case types_1.SupportedModels.MediaPipeFaceMesh:
            return constants.MEDIAPIPE_FACE_MESH_CONNECTED_KEYPOINTS_PAIRS;
        default:
            throw new Error("Model " + model + " is not supported.");
    }
}
exports.getAdjacentPairs = getAdjacentPairs;
//# sourceMappingURL=util.js.map