/**
 * @license
 * Copyright 2023 Google LLC.
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
!function(t,e){"object"==typeof exports&&"undefined"!=typeof module?e(exports,require("@tensorflow/tfjs-converter"),require("@tensorflow/tfjs-core")):"function"==typeof define&&define.amd?define(["exports","@tensorflow/tfjs-converter","@tensorflow/tfjs-core"],e):e((t="undefined"!=typeof globalThis?globalThis:t||self)["body-pix"]={},t.tf,t.tf)}(this,(function(t,e,n){"use strict";function r(t){var e=Object.create(null);return t&&Object.keys(t).forEach((function(n){if("default"!==n){var r=Object.getOwnPropertyDescriptor(t,n);Object.defineProperty(e,n,r.get?r:{enumerable:!0,get:function(){return t[n]}})}})),e.default=t,Object.freeze(e)}var o=r(e),i=r(n),a=function(t,e){return a=Object.setPrototypeOf||{__proto__:[]}instanceof Array&&function(t,e){t.__proto__=e}||function(t,e){for(var n in e)e.hasOwnProperty(n)&&(t[n]=e[n])},a(t,e)};function s(t,e){function n(){this.constructor=t}a(t,e),t.prototype=null===e?Object.create(e):(n.prototype=e.prototype,new n)}var u=function(){return u=Object.assign||function(t){for(var e,n=1,r=arguments.length;n<r;n++)for(var o in e=arguments[n])Object.prototype.hasOwnProperty.call(e,o)&&(t[o]=e[o]);return t},u.apply(this,arguments)};function f(t,e,n,r){return new(n||(n=Promise))((function(o,i){function a(t){try{u(r.next(t))}catch(t){i(t)}}function s(t){try{u(r.throw(t))}catch(t){i(t)}}function u(t){var e;t.done?o(t.value):(e=t.value,e instanceof n?e:new n((function(t){t(e)}))).then(a,s)}u((r=r.apply(t,e||[])).next())}))}function c(t,e){var n,r,o,i,a={label:0,sent:function(){if(1&o[0])throw o[1];return o[1]},trys:[],ops:[]};return i={next:s(0),throw:s(1),return:s(2)},"function"==typeof Symbol&&(i[Symbol.iterator]=function(){return this}),i;function s(i){return function(s){return function(i){if(n)throw new TypeError("Generator is already executing.");for(;a;)try{if(n=1,r&&(o=2&i[0]?r.return:i[0]?r.throw||((o=r.return)&&o.call(r),0):r.next)&&!(o=o.call(r,i[1])).done)return o;switch(r=0,o&&(i=[2&i[0],o.value]),i[0]){case 0:case 1:o=i;break;case 4:return a.label++,{value:i[1],done:!1};case 5:a.label++,r=i[1],i=[0];continue;case 7:i=a.ops.pop(),a.trys.pop();continue;default:if(!(o=a.trys,(o=o.length>0&&o[o.length-1])||6!==i[0]&&2!==i[0])){a=0;continue}if(3===i[0]&&(!o||i[1]>o[0]&&i[1]<o[3])){a.label=i[1];break}if(6===i[0]&&a.label<o[1]){a.label=o[1],o=i;break}if(o&&a.label<o[2]){a.label=o[2],a.ops.push(i);break}o[2]&&a.ops.pop(),a.trys.pop();continue}i=e.call(t,a)}catch(t){i=[6,t],r=0}finally{n=o=0}if(5&i[0])throw i[1];return{value:i[0]?i[1]:void 0,done:!0}}([i,s])}}}
/**
     * @license
     * Copyright 2019 Google LLC. All Rights Reserved.
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
     */function d(t){var e=t.shape[2],n=i.argMax(t,2),r=i.reshape(n,[-1]);return i.oneHot(r,e)}function l(t,e){return i.tidy((function(){return i.cast(i.greater(t,i.scalar(e)),"int32")}))}function h(t,e){var n=e.shape,r=n[0],o=n[1],a=n[2];return i.tidy((function(){var n,s,u=d(e),f=i.expandDims(i.range(0,a,1,"int32"),1),c=i.cast(i.matMul(u,f),"int32"),l=i.reshape(c,[r,o]),h=i.add(l,i.scalar(1,"int32"));return i.sub((n=h,s=t,i.mul(n,s)),i.scalar(1,"int32"))}))}
/**
     * @license
     * Copyright 2019 Google Inc. All Rights Reserved.
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
var p=function(){function t(t,e){this.model=t,this.outputStride=e;var n=this.model.inputs[0].shape;i.util.assert(-1===n[1]&&-1===n[2],(function(){return"Input shape [".concat(n[1],", ").concat(n[2],"] ")+"must both be equal to or -1"}))}return t.prototype.predict=function(t){var e=this;return i.tidy((function(){var n=e.preprocessInput(i.cast(t,"float32")),r=i.expandDims(n,0),o=e.model.predict(r).map((function(t){return i.squeeze(t,[0])})),a=e.nameOutputResults(o);return{heatmapScores:i.sigmoid(a.heatmap),offsets:a.offsets,displacementFwd:a.displacementFwd,displacementBwd:a.displacementBwd,segmentation:a.segmentation,partHeatmaps:a.partHeatmaps,longOffsets:a.longOffsets,partOffsets:a.partOffsets}}))},t.prototype.dispose=function(){this.model.dispose()},t}(),m=function(t){function e(){return null!==t&&t.apply(this,arguments)||this}return s(e,t),e.prototype.preprocessInput=function(t){return i.tidy((function(){return i.sub(i.div(t,127.5),1)}))},e.prototype.nameOutputResults=function(t){return{offsets:t[0],segmentation:t[1],partHeatmaps:t[2],longOffsets:t[3],heatmap:t[4],displacementFwd:t[5],displacementBwd:t[6],partOffsets:t[7]}},e}(p),g=["nose","leftEye","rightEye","leftEar","rightEar","leftShoulder","rightShoulder","leftElbow","rightElbow","leftWrist","rightWrist","leftHip","rightHip","leftKnee","rightKnee","leftAnkle","rightAnkle"],v=g.length,w=g.reduce((function(t,e,n){return t[e]=n,t}),{});
/**
     * @license
     * Copyright 2019 Google Inc. All Rights Reserved.
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
/**
     * @license
     * Copyright 2019 Google LLC. All Rights Reserved.
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
function y(t,e,n){var r=t[0],o=t[1],i=e[0],a=e[1],s=n.top,u=n.bottom;return[a/(n.left+n.right+o),i/(s+u+r)]}function b(t,e,n,r){return{y:r.get(t,e,n),x:r.get(t,e,n+v)}}function x(t,e,n){var r=b(t.heatmapY,t.heatmapX,t.id,n),o=r.y,i=r.x;return{x:t.heatmapX*e+i,y:t.heatmapY*e+o}}function S(t,e,n){return t<e?e:t>n?n:t}function k(t,e){return{x:t.x+e.x,y:t.y+e.y}}
/**
     * @license
     * Copyright 2019 Google Inc. All Rights Reserved.
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
     */function M(t,e,n){void 0===n&&(n=.3);for(var r=0,o=0,i=0;i<t.length;i++)e.keypoints[i].score>n&&(o+=1,r+=Math.pow(t[i].x-e.keypoints[i].position.x,2)+Math.pow(t[i].y-e.keypoints[i].position.y,2));return 0===o?r=1/0:r/=o,r}function O(t,e,n,r,o,i,a){for(var s=a[0],u=a[1],f=n(t),c=f.y*r+f.x,d=o[v*(2*c)+e],l=o[v*(2*c+1)+e],h=t.y+d,p=t.x+l,m=0;m<i;m++){h=Math.min(h,s-1);var g=n({x:p=Math.min(p,u-1),y:h}),w=g.y*r+g.x;h+=d=o[v*(2*w)+e],p+=l=o[v*(2*w+1)+e]}return{x:p,y:h}}function E(t,e,n,r,o,i,a,s,u,f){for(var c=o[0],d=o[1],l=i[0],h=i[1],p=s[0],m=s[1],g=[],v=function(t){return function(t,e,n,r){var o=e[0],i=e[1],a=n[0],s=n[1],u=Math.round(((o+t.y+1)*s-1)/r);return{x:Math.round(((i+t.x+1)*a-1)/r),y:u}}(t,[c,d],[l,h],u)},w=0;w<r;w++){var y=O(t,w,v,a,e,f,[p,m]);g.push(y)}for(var b=-1,x=1/0,S=0;S<n.length;S++){var k=M(g,n[S]);k<x&&(b=S,x=k)}return b}function _(t,e){var n=t[0],r=t[1];return[Math.round((r-1)/e+1),Math.round((n-1)/e+1)]}
/**
     * @license
     * Copyright 2019 Google Inc. All Rights Reserved.
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
function P(t,e,n,r,o,a,s,u,f,c,d){for(var l=s[0],h=s[1],p=t.shape,m=p[0],g=p[1],w=e.shape.slice(0,2),b=w[0],x=w[1],S=i.reshape(e,[b,x,2,v]),k=new Float32Array(d*v*3).fill(0),M=0;M<n.length;M++)for(var O=M*v*3,E=n[M],_=0;_<v;_++){var P=E.keypoints[_],A=O+3*_;k[A]=P.score,k[A+1]=P.position.y,k[A+2]=P.position.x}var I=y([r,o],[l,h],u),R=I[0],T=I[1],H=i.tensor(k,[d,v,3]),B=u.top,F=u.left,D={variableNames:["segmentation","longOffsets","poses"],outputShape:[m,g],userCode:"\n    int convertToPositionInOutput(int pos, int pad, float scale, int stride) {\n      return round(((float(pos + pad) + 1.0) * scale - 1.0) / float(stride));\n    }\n\n    float convertToPositionInOutputFloat(\n        int pos, int pad, float scale, int stride) {\n      return ((float(pos + pad) + 1.0) * scale - 1.0) / float(stride);\n    }\n\n    float dist(float x1, float y1, float x2, float y2) {\n      return pow(x1 - x2, 2.0) + pow(y1 - y2, 2.0);\n    }\n\n    float sampleLongOffsets(float h, float w, int d, int k) {\n      float fh = fract(h);\n      float fw = fract(w);\n      int clH = int(ceil(h));\n      int clW = int(ceil(w));\n      int flH = int(floor(h));\n      int flW = int(floor(w));\n      float o11 = getLongOffsets(flH, flW, d, k);\n      float o12 = getLongOffsets(flH, clW, d, k);\n      float o21 = getLongOffsets(clH, flW, d, k);\n      float o22 = getLongOffsets(clH, clW, d, k);\n      float o1 = mix(o11, o12, fw);\n      float o2 = mix(o21, o22, fw);\n      return mix(o1, o2, fh);\n    }\n\n    int findNearestPose(int h, int w) {\n      float prob = getSegmentation(h, w);\n      if (prob < 1.0) {\n        return -1;\n      }\n\n      // Done(Tyler): convert from output space h/w to strided space.\n      float stridedH = convertToPositionInOutputFloat(\n        h, ".concat(B,", ").concat(T,", ").concat(a,");\n      float stridedW = convertToPositionInOutputFloat(\n        w, ").concat(F,", ").concat(R,", ").concat(a,");\n\n      float minDist = 1000000.0;\n      int iMin = -1;\n      for (int i = 0; i < ").concat(d,"; i++) {\n        float curDistSum = 0.0;\n        int numKpt = 0;\n        for (int k = 0; k < ").concat(v,"; k++) {\n          float dy = sampleLongOffsets(stridedH, stridedW, 0, k);\n          float dx = sampleLongOffsets(stridedH, stridedW, 1, k);\n\n          float y = float(h) + dy;\n          float x = float(w) + dx;\n\n          for (int s = 0; s < ").concat(f,"; s++) {\n            int yRounded = round(min(y, float(").concat(r-1,")));\n            int xRounded = round(min(x, float(").concat(o-1,")));\n\n            float yStrided = convertToPositionInOutputFloat(\n              yRounded, ").concat(B,", ").concat(T,", ").concat(a,");\n            float xStrided = convertToPositionInOutputFloat(\n              xRounded, ").concat(F,", ").concat(R,", ").concat(a,");\n\n            float dy = sampleLongOffsets(yStrided, xStrided, 0, k);\n            float dx = sampleLongOffsets(yStrided, xStrided, 1, k);\n\n            y = y + dy;\n            x = x + dx;\n          }\n\n          float poseScore = getPoses(i, k, 0);\n          float poseY = getPoses(i, k, 1);\n          float poseX = getPoses(i, k, 2);\n          if (poseScore > ").concat(c,") {\n            numKpt = numKpt + 1;\n            curDistSum = curDistSum + dist(x, y, poseX, poseY);\n          }\n        }\n        if (numKpt > 0 && curDistSum / float(numKpt) < minDist) {\n          minDist = curDistSum / float(numKpt);\n          iMin = i;\n        }\n      }\n      return iMin;\n    }\n\n    void main() {\n        ivec2 coords = getOutputCoords();\n        int nearestPose = findNearestPose(coords[0], coords[1]);\n        setOutput(float(nearestPose));\n      }\n  ")};return i.backend().compileAndRun(D,[t,S,H])}
/**
     * @license
     * Copyright 2019 Google Inc. All Rights Reserved.
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
     */function A(){return"webgl"===n.getBackend()}function I(t,e,n,r,o,a,s,u,d,l,h,p){var m=s[0],g=s[1];return void 0===d&&(d=.2),void 0===l&&(l=8),void 0===h&&(h=.3),void 0===p&&(p=10),f(this,void 0,void 0,(function(){var s,f,v,w,b;return c(this,(function(c){switch(c.label){case 0:return s=n.filter((function(t){return t.score>=d})),A()?(v=i.tidy((function(){var n=P(t,e,s,r,o,a,[m,g],u,l,h,p),f=i.engine().makeTensorFromDataId(n.dataId,n.shape,n.dtype);return s.map((function(t,e){return function(t,e){return i.tidy((function(){return i.cast(i.equal(t,i.scalar(e)),"int32")}))}(f,e)}))})),[4,Promise.all(v.map((function(t){return t.data()})))]):[3,2];case 1:return f=c.sent(),v.forEach((function(t){return t.dispose()})),[3,5];case 2:return[4,t.data()];case 3:return w=c.sent(),[4,e.data()];case 4:b=c.sent(),f=function(t,e,n,r,o,i,a,s,u,f){var c=a[0],d=a[1];void 0===f&&(f=5);for(var l=n.map((function(t){return new Uint8Array(r*o).fill(0)})),h=s.top,p=s.left,m=y([r,o],[c,d],s),g=m[0],v=m[1],w=_([c,d],i)[0],b=0;b<r;b+=1)for(var x=0;x<o;x+=1){var S=b*o+x;if(1===t[S]){var k=E({x:x,y:b},e,n,f,[h,p],[g,v],w,[r,o],i,u);k>=0&&(l[k][S]=1)}}return l}(w,b,s,r,o,a,[m,g],u,l),c.label=5;case 5:return[2,f.map((function(t,e){return{data:t,pose:s[e],width:o,height:r}}))]}}))}))}function R(t,e,n,r,o,a,s,u,d,l,h,p,m){var g=u[0],v=u[1];return void 0===l&&(l=.2),void 0===h&&(h=8),void 0===p&&(p=.3),void 0===m&&(m=10),f(this,void 0,void 0,(function(){var u,f,w,b,x,S;return c(this,(function(c){switch(c.label){case 0:return u=r.filter((function(t){return t.score>=l})),A()?(w=i.tidy((function(){var r=P(t,e,u,o,a,s,[g,v],d,h,p,m),f=i.engine().makeTensorFromDataId(r.dataId,r.shape,r.dtype);return u.map((function(t,e){return function(t,e,n){return i.tidy((function(){return i.sub(i.mul(i.cast(i.equal(t,i.scalar(n)),"int32"),i.add(e,1)),1)}))}(f,n,e)}))})),[4,Promise.all(w.map((function(t){return t.data()})))]):[3,2];case 1:return f=c.sent(),w.forEach((function(t){return t.dispose()})),[3,6];case 2:return[4,t.data()];case 3:return b=c.sent(),[4,e.data()];case 4:return x=c.sent(),[4,n.data()];case 5:S=c.sent(),f=function(t,e,n,r,o,i,a,s,u,f,c){var d=s[0],l=s[1];void 0===c&&(c=5);for(var h=r.map((function(t){return new Int32Array(o*i).fill(-1)})),p=u.top,m=u.left,g=y([o,i],[d,l],u),v=g[0],w=g[1],b=_([d,l],a)[0],x=0;x<o;x+=1)for(var S=0;S<i;S+=1){var k=x*i+S;if(1===t[k]){var M=E({x:S,y:x},e,r,c,[p,m],[v,w],b,[o,i],a,f);M>=0&&(h[M][k]=n[k])}}return h}(b,x,S,u,o,a,s,[g,v],d,h),c.label=6;case 6:return[2,f.map((function(t,e){return{pose:u[e],data:t,height:o,width:a}}))]}}))}))}
/**
     * @license
     * Copyright 2019 Google Inc. All Rights Reserved.
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
     */function T(t){return Math.floor(t/2)}[["leftHip","leftShoulder"],["leftElbow","leftShoulder"],["leftElbow","leftWrist"],["leftHip","leftKnee"],["leftKnee","leftAnkle"],["rightHip","rightShoulder"],["rightElbow","rightShoulder"],["rightElbow","rightWrist"],["rightHip","rightKnee"],["rightKnee","rightAnkle"],["leftShoulder","rightShoulder"],["leftHip","rightHip"]].map((function(t){var e=t[0],n=t[1];return[w[e],w[n]]}));var H=function(){function t(t,e){this.priorityQueue=new Array(t),this.numberOfElements=-1,this.getElementValue=e}return t.prototype.enqueue=function(t){this.priorityQueue[++this.numberOfElements]=t,this.swim(this.numberOfElements)},t.prototype.dequeue=function(){var t=this.priorityQueue[0];return this.exchange(0,this.numberOfElements--),this.sink(0),this.priorityQueue[this.numberOfElements+1]=null,t},t.prototype.empty=function(){return-1===this.numberOfElements},t.prototype.size=function(){return this.numberOfElements+1},t.prototype.all=function(){return this.priorityQueue.slice(0,this.numberOfElements+1)},t.prototype.max=function(){return this.priorityQueue[0]},t.prototype.swim=function(t){for(;t>0&&this.less(T(t),t);)this.exchange(t,T(t)),t=T(t)},t.prototype.sink=function(t){for(;2*t<=this.numberOfElements;){var e=2*t;if(e<this.numberOfElements&&this.less(e,e+1)&&e++,!this.less(t,e))break;this.exchange(t,e),t=e}},t.prototype.getValueAt=function(t){return this.getElementValue(this.priorityQueue[t])},t.prototype.less=function(t,e){return this.getValueAt(t)<this.getValueAt(e)},t.prototype.exchange=function(t,e){var n=this.priorityQueue[t];this.priorityQueue[t]=this.priorityQueue[e],this.priorityQueue[e]=n},t}();
/**
     * @license
     * Copyright 2019 Google LLC. All Rights Reserved.
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
     */function B(t,e,n,r,o,i){for(var a=i.shape,s=a[0],u=a[1],f=!0,c=Math.max(n-o,0),d=Math.min(n+o+1,s),l=c;l<d;++l){for(var h=Math.max(r-o,0),p=Math.min(r+o+1,u),m=h;m<p;++m)if(i.get(l,m,t)>e){f=!1;break}if(!f)break}return f}
/**
     * @license
     * Copyright 2019 Google LLC. All Rights Reserved.
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
var F=[["nose","leftEye"],["leftEye","leftEar"],["nose","rightEye"],["rightEye","rightEar"],["nose","leftShoulder"],["leftShoulder","leftElbow"],["leftElbow","leftWrist"],["leftShoulder","leftHip"],["leftHip","leftKnee"],["leftKnee","leftAnkle"],["nose","rightShoulder"],["rightShoulder","rightElbow"],["rightElbow","rightWrist"],["rightShoulder","rightHip"],["rightHip","rightKnee"],["rightKnee","rightAnkle"]].map((function(t){var e=t[0],n=t[1];return[w[e],w[n]]})),D=F.map((function(t){return t[1]})),C=F.map((function(t){return t[0]}));function L(t,e,n,r){return{y:S(Math.round(t.y/e),0,n-1),x:S(Math.round(t.x/e),0,r-1)}}function q(t,e,n,r,o,i,a,s){void 0===s&&(s=2);for(var u=r.shape,f=u[0],c=u[1],d=function(t,e,n){var r=n.shape[2]/2;return{y:n.get(e.y,e.x,t),x:n.get(e.y,e.x,r+t)}}(t,L(e.position,i,f,c),a),l=k(e.position,d),h=0;h<s;h++){var p=L(l,i,f,c),m=b(p.y,p.x,n,o);l=k({x:p.x*i,y:p.y*i},{x:m.x,y:m.y})}var v=L(l,i,f,c),w=r.get(v.y,v.x,n);return{position:l,part:g[n],score:w}}function j(t,e,n,r,o,i){var a=e.shape[2],s=D.length,u=new Array(a),f=t.part,c=t.score,d=x(f,r,n);u[f.id]={score:c,part:g[f.id],position:d};for(var l=s-1;l>=0;--l){var h=D[l],p=C[l];u[h]&&!u[p]&&(u[p]=q(l,u[h],p,e,n,r,i))}for(l=0;l<s;++l){h=C[l],p=D[l];u[h]&&!u[p]&&(u[p]=q(l,u[h],p,e,n,r,o))}return u}
/**
     * @license
     * Copyright 2019 Google Inc. All Rights Reserved.
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
     */function z(t,e,n,r){var o=n.x,i=n.y;return t.some((function(t){var n,a,s,u,f,c,d=t.keypoints[r].position;return n=i,a=o,s=d.y,u=d.x,(f=s-n)*f+(c=u-a)*c<=e}))}function W(t,e,n){var r=n.reduce((function(n,r,o){var i=r.position,a=r.score;return z(t,e,i,o)||(n+=a),n}),0);return r/n.length}var K=1;function N(t,e,n,r,o,i,a,s){void 0===a&&(a=.5),void 0===s&&(s=20);for(var u=[],f=function(t,e,n){for(var r=n.shape,o=r[0],i=r[1],a=r[2],s=new H(o*i*a,(function(t){return t.score})),u=0;u<o;++u)for(var f=0;f<i;++f)for(var c=0;c<a;++c){var d=n.get(u,f,c);d<t||B(c,d,u,f,e,n)&&s.enqueue({score:d,part:{heatmapY:u,heatmapX:f,id:c}})}return s}(a,K,t),c=s*s;u.length<i&&!f.empty();){var d=f.dequeue();if(!z(u,c,x(d.part,o,e),d.part.id)){var l=j(d,t,e,o,n,r),h=W(u,c,l);u.push({keypoints:l,score:h})}}return u}
/**
     * @license
     * Copyright 2019 Google Inc. All Rights Reserved.
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
     */var V,Q=[-123.15,-115.9,-103.06],U=function(t){function e(){return null!==t&&t.apply(this,arguments)||this}return s(e,t),e.prototype.preprocessInput=function(t){return i.add(t,Q)},e.prototype.nameOutputResults=function(t){var e=t[0],n=t[1],r=t[2],o=t[3],i=t[4],a=t[5];return{offsets:i,segmentation:t[6],partHeatmaps:a,longOffsets:o,heatmap:r,displacementFwd:n,displacementBwd:e,partOffsets:t[7]}},e}(p),X="https://storage.googleapis.com/tfjs-models/savedmodel/bodypix/resnet50/",Y="https://storage.googleapis.com/tfjs-models/savedmodel/bodypix/mobilenet/";function G(t){if("undefined"!=typeof HTMLCanvasElement&&t instanceof HTMLCanvasElement||"undefined"!=typeof OffscreenCanvas&&t instanceof OffscreenCanvas||"undefined"!=typeof HTMLImageElement&&t instanceof HTMLImageElement)return function(t){if("offsetHeight"in t&&0!==t.offsetHeight&&"offsetWidth"in t&&0!==t.offsetWidth)return[t.offsetHeight,t.offsetWidth];if(null!=t.height&&null!=t.width)return[t.height,t.width];throw new Error("HTMLImageElement must have height and width attributes set.")}(t);if("undefined"!=typeof ImageData&&t instanceof ImageData)return[t.height,t.width];if("undefined"!=typeof HTMLVideoElement&&t instanceof HTMLVideoElement)return function(t){return t.hasAttribute("height")&&t.hasAttribute("width")?[t.height,t.width]:[t.videoHeight,t.videoWidth]}(t);if(t instanceof i.Tensor)return[t.shape[0],t.shape[1]];throw new Error("error: Unknown input type: ".concat(t,"."))}function J(t,e){return function(t,e){return(t-1)%e==0}(t,e)?t:Math.floor(t/e)*e+1}var Z={low:"low",medium:"medium",high:"high",full:"full"},$=((V={})[Z.low]=.25,V[Z.medium]=.5,V[Z.high]=.75,V[Z.full]=1,V),tt=.1,et=2;function nt(t,e,n){var r=n[0],o=n[1],a=function(t){if("string"==typeof t){var e=$[t];return i.util.assert("number"==typeof e,(function(){return"string value of inputResolution must be one of ".concat(Object.values(Z).join(",")," but was ").concat(t,".")})),e}return i.util.assert("number"==typeof t&&t<=et&&t>=tt,(function(){return"inputResolution must be a string or number between ".concat(tt," and ").concat(et,", but ")+"was ".concat(t)})),t}(t);return[J(r*a,e),J(o*a,e)]}function rt(t,e,n,r,o){var a=e[0],s=e[1],u=n[0],f=n[1],c=r[0],d=c[0],l=c[1],h=r[1],p=h[0],m=h[1];return void 0===o&&(o=!1),i.tidy((function(){var e=i.image.resizeBilinear(t,[u,f],!0);return o&&(e=i.sigmoid(e)),function(t,e,n){var r=e[0],o=e[1],a=n[0],s=a[0],u=a[1],f=n[1],c=f[0],d=f[1];return i.tidy((function(){var e=i.expandDims(t);return i.squeeze(i.image.cropAndResize(e,[[s/(r+s+u-1),c/(o+c+d-1),(s+r-1)/(r+s+u-1),(c+o-1)/(o+c+d-1)]],[0],[r,o]),[0])}))}(e,[a,s],[[d,l],[p,m]])}))}function ot(t,e){var n=e[0],r=e[1],o=G(t),a=o[0],s=o[1],u=r/n,f=[0,0,0,0],c=f[0],d=f[1],l=f[2],h=f[3];s/a<u?(c=0,d=0,l=Math.round(.5*(u*a-s)),h=Math.round(.5*(u*a-s))):(c=Math.round(.5*(1/u*s-a)),d=Math.round(.5*(1/u*s-a)),l=0,h=0);var p=i.tidy((function(){var e=function(t){return t instanceof i.Tensor?t:i.browser.fromPixels(t)}(t);return e=i.pad3d(e,[[c,d],[l,h],[0,0]]),i.image.resizeBilinear(e,[n,r])}));return{resized:p,padding:{top:c,left:l,right:h,bottom:d}}}function it(t){return f(this,void 0,void 0,(function(){return c(this,(function(e){return[2,Promise.all(t.map((function(t){return t.buffer()})))]}))}))}function at(t,e){return{score:t.score,keypoints:t.keypoints.map((function(t){var n=t.score,r=t.part,o=t.position;return{score:n,part:r,position:{x:e-1-o.x,y:o.y}}}))}}function st(t,e,n,r,o){var i=e[0],a=e[1],s=n[0],u=n[1],f=function(t,e,n,r,o){return void 0===r&&(r=0),void 0===o&&(o=0),1===n&&1===e&&0===r&&0===o?t:t.map((function(t){return function(t,e,n,r,o){return void 0===r&&(r=0),void 0===o&&(o=0),{score:t.score,keypoints:t.keypoints.map((function(t){var i=t.score,a=t.part,s=t.position;return{score:i,part:a,position:{x:s.x*n+o,y:s.y*e+r}}}))}}(t,e,n,r,o)}))}(t,(i+r.top+r.bottom)/s,(a+r.left+r.right)/u,-r.top,-r.left);return o?function(t,e){return e<=0?t:t.map((function(t){return at(t,e)}))}(f,a):f}
/**
     * @license
     * Copyright 2019 Google Inc. All Rights Reserved.
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
     */var ut=!0,ft=!1,ct={architecture:"MobileNetV1",outputStride:16,quantBytes:4,multiplier:.75},dt=["MobileNetV1","ResNet50"],lt={MobileNetV1:[8,16,32],ResNet50:[32,16]},ht={MobileNetV1:[.5,.75,1],ResNet50:[1]},pt=[1,2,4];var mt={flipHorizontal:!1,internalResolution:"medium",segmentationThreshold:.7,maxDetections:10,scoreThreshold:.4,nmsRadius:20},gt={flipHorizontal:!1,internalResolution:"medium",segmentationThreshold:.7,maxDetections:10,scoreThreshold:.4,nmsRadius:20,minKeypointScore:.3,refineSteps:10};function vt(t){var e=t.segmentationThreshold,n=t.maxDetections,r=t.scoreThreshold,o=t.nmsRadius;if(e<0||e>1)throw new Error("segmentationThreshold ".concat(e,". ")+"Should be in range [0.0, 1.0]");if(n<=0)throw new Error("Invalid maxDetections ".concat(n,". ")+"Should be > 0");if(r<0||r>1)throw new Error("Invalid scoreThreshold ".concat(r,". ")+"Should be in range [0.0, 1.0]");if(o<=0)throw new Error("Invalid nmsRadius ".concat(o,"."))}function wt(t){var e=t.segmentationThreshold,n=t.maxDetections,r=t.scoreThreshold,o=t.nmsRadius,i=t.minKeypointScore,a=t.refineSteps;if(e<0||e>1)throw new Error("segmentationThreshold ".concat(e,". ")+"Should be in range [0.0, 1.0]");if(n<=0)throw new Error("Invalid maxDetections ".concat(n,". ")+"Should be > 0");if(r<0||r>1)throw new Error("Invalid scoreThreshold ".concat(r,". ")+"Should be in range [0.0, 1.0]");if(o<=0)throw new Error("Invalid nmsRadius ".concat(o,"."));if(i<0||i>1)throw new Error("Invalid minKeypointScore ".concat(i,".")+"Should be in range [0.0, 1.0]");if(a<=0||a>20)throw new Error("Invalid refineSteps ".concat(a,".")+"Should be in range [1, 20]")}var yt=function(){function t(t){this.baseModel=t}return t.prototype.predictForPersonSegmentation=function(t){var e=this.baseModel.predict(t);return{segmentLogits:e.segmentation,heatmapScores:e.heatmapScores,offsets:e.offsets,displacementFwd:e.displacementFwd,displacementBwd:e.displacementBwd}},t.prototype.predictForPersonSegmentationAndPart=function(t){var e=this.baseModel.predict(t);return{segmentLogits:e.segmentation,partHeatmapLogits:e.partHeatmaps,heatmapScores:e.heatmapScores,offsets:e.offsets,displacementFwd:e.displacementFwd,displacementBwd:e.displacementBwd}},t.prototype.predictForMultiPersonInstanceSegmentationAndPart=function(t){var e=this.baseModel.predict(t);return{segmentLogits:e.segmentation,longOffsets:e.longOffsets,heatmapScores:e.heatmapScores,offsets:e.offsets,displacementFwd:e.displacementFwd,displacementBwd:e.displacementBwd,partHeatmaps:e.partHeatmaps}},t.prototype.segmentPersonActivation=function(t,e,n){var r=this;void 0===n&&(n=.5);var o=G(t),a=o[0],s=o[1],u=nt(e,this.baseModel.outputStride,[a,s]),f=ot(t,u),c=f.resized,d=f.padding,h=i.tidy((function(){var t=r.predictForPersonSegmentation(c),e=t.segmentLogits,o=t.heatmapScores,u=t.offsets,f=t.displacementFwd,h=t.displacementBwd,p=c.shape,m=p[0],g=p[1],v=rt(e,[a,s],[m,g],[[d.top,d.bottom],[d.left,d.right]],ut);return{segmentation:l(i.squeeze(v),n),heatmapScores:o,offsets:u,displacementFwd:f,displacementBwd:h}})),p=h.segmentation,m=h.heatmapScores,g=h.offsets,v=h.displacementFwd,w=h.displacementBwd;return c.dispose(),{segmentation:p,heatmapScores:m,offsets:g,displacementFwd:v,displacementBwd:w,padding:d,internalResolutionHeightAndWidth:u}},t.prototype.segmentPerson=function(t,e){return void 0===e&&(e=mt),f(this,void 0,void 0,(function(){var n,r,o,i,a,s,f,d,l,h,p,m,g,v,w,y,b,x;return c(this,(function(c){switch(c.label){case 0:return vt(e=u(u({},mt),e)),n=this.segmentPersonActivation(t,e.internalResolution,e.segmentationThreshold),r=n.segmentation,o=n.heatmapScores,i=n.offsets,a=n.displacementFwd,s=n.displacementBwd,f=n.padding,d=n.internalResolutionHeightAndWidth,l=r.shape,h=l[0],p=l[1],[4,r.data()];case 1:return m=c.sent(),r.dispose(),[4,it([o,i,a,s])];case 2:return g=c.sent(),v=g[0],w=g[1],y=g[2],b=g[3],x=st(x=N(v,w,y,b,this.baseModel.outputStride,e.maxDetections,e.scoreThreshold,e.nmsRadius),[h,p],d,f,ft),o.dispose(),i.dispose(),a.dispose(),s.dispose(),[2,{height:h,width:p,data:m,allPoses:x}]}}))}))},t.prototype.segmentMultiPerson=function(t,e){return void 0===e&&(e=gt),f(this,void 0,void 0,(function(){var n,r,o,a,s,f,d,h,p,m,g,v,w,y,b,x,S,k,M,O,E,_=this;return c(this,(function(c){switch(c.label){case 0:return wt(e=u(u({},gt),e)),n=G(t),r=n[0],o=n[1],a=nt(e.internalResolution,this.baseModel.outputStride,[r,o]),s=ot(t,a),f=s.resized,d=s.padding,h=i.tidy((function(){var t,n=_.predictForMultiPersonInstanceSegmentationAndPart(f),s=n.segmentLogits,u=n.longOffsets,c=n.heatmapScores,h=n.offsets,p=n.displacementFwd,m=n.displacementBwd,g=rt(s,[r,o],a,[[d.top,d.bottom],[d.left,d.right]],ut);return t=u,{segmentation:l(i.squeeze(g),e.segmentationThreshold),longOffsets:t,heatmapScoresRaw:c,offsetsRaw:h,displacementFwdRaw:p,displacementBwdRaw:m}})),p=h.segmentation,m=h.longOffsets,g=h.heatmapScoresRaw,v=h.offsetsRaw,w=h.displacementFwdRaw,y=h.displacementBwdRaw,[4,it([g,v,w,y])];case 1:return b=c.sent(),x=b[0],S=b[1],k=b[2],M=b[3],O=st(O=N(x,S,k,M,this.baseModel.outputStride,e.maxDetections,e.scoreThreshold,e.nmsRadius),[r,o],a,d,ft),[4,I(p,m,O,r,o,this.baseModel.outputStride,a,d,e.scoreThreshold,e.refineSteps,e.minKeypointScore,e.maxDetections)];case 2:return E=c.sent(),f.dispose(),p.dispose(),m.dispose(),g.dispose(),v.dispose(),w.dispose(),y.dispose(),[2,E]}}))}))},t.prototype.segmentPersonPartsActivation=function(t,e,n){var r=this;void 0===n&&(n=.5);var o=G(t),a=o[0],s=o[1],u=nt(e,this.baseModel.outputStride,[a,s]),f=ot(t,u),c=f.resized,d=f.padding,p=i.tidy((function(){var t=r.predictForPersonSegmentationAndPart(c),e=t.segmentLogits,o=t.partHeatmapLogits,u=t.heatmapScores,f=t.offsets,p=t.displacementFwd,m=t.displacementBwd,g=c.shape,v=g[0],w=g[1],y=rt(e,[a,s],[v,w],[[d.top,d.bottom],[d.left,d.right]],ut),b=rt(o,[a,s],[v,w],[[d.top,d.bottom],[d.left,d.right]],ut);return{partSegmentation:h(l(i.squeeze(y),n),b),heatmapScores:u,offsets:f,displacementFwd:p,displacementBwd:m}})),m=p.partSegmentation,g=p.heatmapScores,v=p.offsets,w=p.displacementFwd,y=p.displacementBwd;return c.dispose(),{partSegmentation:m,heatmapScores:g,offsets:v,displacementFwd:w,displacementBwd:y,padding:d,internalResolutionHeightAndWidth:u}},t.prototype.segmentPersonParts=function(t,e){return void 0===e&&(e=mt),f(this,void 0,void 0,(function(){var n,r,o,i,a,s,f,d,l,h,p,m,g,v,w,y,b,x;return c(this,(function(c){switch(c.label){case 0:return vt(e=u(u({},mt),e)),n=this.segmentPersonPartsActivation(t,e.internalResolution,e.segmentationThreshold),r=n.partSegmentation,o=n.heatmapScores,i=n.offsets,a=n.displacementFwd,s=n.displacementBwd,f=n.padding,d=n.internalResolutionHeightAndWidth,l=r.shape,h=l[0],p=l[1],[4,r.data()];case 1:return m=c.sent(),r.dispose(),[4,it([o,i,a,s])];case 2:return g=c.sent(),v=g[0],w=g[1],y=g[2],b=g[3],x=st(x=N(v,w,y,b,this.baseModel.outputStride,e.maxDetections,e.scoreThreshold,e.nmsRadius),[h,p],d,f,ft),o.dispose(),i.dispose(),a.dispose(),s.dispose(),[2,{height:h,width:p,data:m,allPoses:x}]}}))}))},t.prototype.segmentMultiPersonParts=function(t,e){return void 0===e&&(e=gt),f(this,void 0,void 0,(function(){var n,r,o,a,s,f,h,p,m,g,v,w,y,b,x,S,k,M,O,E,_,P,A=this;return c(this,(function(c){switch(c.label){case 0:return wt(e=u(u({},gt),e)),n=G(t),r=n[0],o=n[1],a=nt(e.internalResolution,this.baseModel.outputStride,[r,o]),s=ot(t,a),f=s.resized,h=s.padding,p=i.tidy((function(){var t=A.predictForMultiPersonInstanceSegmentationAndPart(f),n=t.segmentLogits,s=t.longOffsets,u=t.heatmapScores,c=t.offsets,p=t.displacementFwd,m=t.displacementBwd,g=t.partHeatmaps,v=rt(n,[r,o],a,[[h.top,h.bottom],[h.left,h.right]],ut),w=rt(g,[r,o],a,[[h.top,h.bottom],[h.left,h.right]],ut),y=s,b=l(i.squeeze(v),e.segmentationThreshold),x=function(t){var e=t.shape,n=e[0],r=e[1],o=e[2];return i.tidy((function(){var e=d(t),a=i.expandDims(i.range(0,o,1,"int32"),1),s=i.cast(i.matMul(e,a),"int32");return i.reshape(s,[n,r])}))}(w);return{segmentation:b,longOffsets:y,heatmapScoresRaw:u,offsetsRaw:c,displacementFwdRaw:p,displacementBwdRaw:m,partSegmentation:x}})),m=p.segmentation,g=p.longOffsets,v=p.heatmapScoresRaw,w=p.offsetsRaw,y=p.displacementFwdRaw,b=p.displacementBwdRaw,x=p.partSegmentation,[4,it([v,w,y,b])];case 1:return S=c.sent(),k=S[0],M=S[1],O=S[2],E=S[3],_=st(_=N(k,M,O,E,this.baseModel.outputStride,e.maxDetections,e.scoreThreshold,e.nmsRadius),[r,o],a,h,ft),[4,R(m,g,x,_,r,o,this.baseModel.outputStride,a,h,e.scoreThreshold,e.refineSteps,e.minKeypointScore,e.maxDetections)];case 2:return P=c.sent(),f.dispose(),m.dispose(),g.dispose(),v.dispose(),w.dispose(),y.dispose(),b.dispose(),x.dispose(),[2,P]}}))}))},t.prototype.dispose=function(){this.baseModel.dispose()},t}();function bt(t){return f(this,void 0,void 0,(function(){var e,n,r,a,s,u;return c(this,(function(f){switch(f.label){case 0:if(e=t.outputStride,n=t.quantBytes,r=t.multiplier,null==i)throw new Error("Cannot find TensorFlow.js. If you are using a <script> tag, please also include @tensorflow/tfjs on the page before using this\n        model.");return a=function(t,e,n){var r={1:"100",.75:"075",.5:"050"},o="model-stride".concat(t,".json");return 4===n?Y+"float/".concat(r[e],"/")+o:Y+"quant".concat(n,"/").concat(r[e],"/")+o}
/**
     * @license
     * Copyright 2020 Google Inc. All Rights Reserved.
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
     *
     * =============================================================================
     */(e,r,n),[4,o.loadGraphModel(t.modelUrl||a)];case 1:return s=f.sent(),u=new m(s,e),[2,new yt(u)]}}))}))}function xt(t){return f(this,void 0,void 0,(function(){var e,n,r,a,s;return c(this,(function(u){switch(u.label){case 0:if(e=t.outputStride,n=t.quantBytes,null==i)throw new Error("Cannot find TensorFlow.js. If you are using a <script> tag, please also include @tensorflow/tfjs on the page before using this\n        model.");return r=function(t,e){var n="model-stride".concat(t,".json");return 4===e?X+"float/"+n:X+"quant".concat(e,"/")+n}(e,n),[4,o.loadGraphModel(t.modelUrl||r)];case 1:return a=u.sent(),s=new U(a,e),[2,new yt(s)]}}))}))}
/**
     * @license
     * Copyright 2019 Google LLC. All Rights Reserved.
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
var St={};function kt(t,e,n,r){var o=t.width,i=t.height,a=e.width,s=e.height;if(o!==a||i!==s)throw new Error("error: dimensions must match. ".concat(n," has dimensions ").concat(o,"x").concat(i,", ").concat(r," has dimensions ").concat(a,"x").concat(s))}function Mt(t){var e=t.getContext("2d");e.scale(-1,1),e.translate(-t.width,0)}function Ot(t,e,n){t.globalCompositeOperation=n,t.drawImage(e,0,0)}function Et(t){return St[t]||(St[t]=function(){if("undefined"!=typeof document)return document.createElement("canvas");if("undefined"!=typeof OffscreenCanvas)return new OffscreenCanvas(0,0);throw new Error("Cannot create a canvas in this context")}()),St[t]}function _t(t,e,n){var r=t.height,o=t.width,i=n.getContext("2d");n.width=o,n.height=r,i.clearRect(0,0,o,r),i.save(),/^((?!chrome|android).)*safari/i.test(navigator.userAgent)?
/**
     * @license
     * Copyright 2019 Google LLC. All Rights Reserved.
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
function(t,e,n){for(var r=t.getContext("2d"),o=0,i=1/(2*Math.PI*5*5),a=n<3?1:2,s=-n;s<=n;s+=a)for(var u=-n;u<=n;u+=a)o+=i*Math.exp(-(u*u+s*s)/50);for(s=-n;s<=n;s+=a)for(u=-n;u<=n;u+=a)r.globalAlpha=i*Math.exp(-(u*u+s*s)/50)/o*n,r.drawImage(e,u,s);r.globalAlpha=1}(n,t,e):(i.filter="blur(".concat(e,"px)"),i.drawImage(t,0,0,o,r)),i.restore()}function Pt(t,e,n){var r=Et(n);return 0===e?function(t,e){var n=t.width,r=t.height;e.width=n,e.height=r;var o=e.getContext("2d");o.drawImage(t,0,0,n,r)}(t,r):_t(t,e,r),r}function At(t,e){var n=Et(e);return function(t,e){e.width=t.width,e.height=t.height,e.getContext("2d").putImageData(t,0,0)}(t,n),n}function It(t,e,n,r,o){if(void 0===e&&(e={r:0,g:0,b:0,a:0}),void 0===n&&(n={r:0,g:0,b:0,a:255}),void 0===r&&(r=!1),void 0===o&&(o=[1]),Array.isArray(t)&&0===t.length)return null;var i,a=(i=Array.isArray(t)?t:[t])[0],s=a.width,u=a.height,f=new Uint8ClampedArray(s*u*4);function c(t,e,n,r,o,i){void 0===i&&(i={r:0,g:255,b:255,a:255});for(var a=-o;a<=o;a++)for(var s=-o;s<=o;s++)if(0!==a&&0!==s){var u=(e+a)*r+(n+s);t[4*u+0]=i.r,t[4*u+1]=i.g,t[4*u+2]=i.b,t[4*u+3]=i.a}}function d(t,e,n,r,o,i){void 0===o&&(o=[1]),void 0===i&&(i=1);for(var a=0,s=-i;s<=i;s++)for(var u=function(i){if(0!==s&&0!==i){var u=(e+s)*r+(n+i);o.some((function(e){return e===t[u]}))||(a+=1)}},f=-i;f<=i;f++)u(f);return a>0}for(var l=0;l<u;l+=1)for(var h=function(t){var a=l*s+t;f[4*a+0]=n.r,f[4*a+1]=n.g,f[4*a+2]=n.b,f[4*a+3]=n.a;for(var h=function(n){if(o.some((function(t){return t===i[n].data[a]}))){f[4*a]=e.r,f[4*a+1]=e.g,f[4*a+2]=e.b,f[4*a+3]=e.a;var h=d(i[n].data,l,t,s,o);r&&l-1>=0&&l+1<u&&t-1>=0&&t+1<s&&h&&c(f,l,t,s,1)}},p=0;p<i.length;p++)h(p)},p=0;p<s;p+=1)h(p);return new ImageData(f,s,u)}var Rt=[[110,64,170],[143,61,178],[178,60,178],[210,62,167],[238,67,149],[255,78,125],[255,94,99],[255,115,75],[255,140,56],[239,167,47],[217,194,49],[194,219,64],[175,240,91],[135,245,87],[96,247,96],[64,243,115],[40,234,141],[28,219,169],[26,199,194],[33,176,213],[47,150,224],[65,125,224],[84,101,214],[99,81,195]];var Tt={blurred:"blurred",blurredMask:"blurred-mask",mask:"mask",lowresPartMask:"lowres-part-mask"};t.BodyPix=yt,t.PART_CHANNELS=["left_face","right_face","left_upper_arm_front","left_upper_arm_back","right_upper_arm_front","right_upper_arm_back","left_lower_arm_front","left_lower_arm_back","right_lower_arm_front","right_lower_arm_back","left_hand","right_hand","torso_front","torso_back","left_upper_leg_front","left_upper_leg_back","right_upper_leg_front","right_upper_leg_back","left_lower_leg_front","left_lower_leg_back","right_lower_leg_front","right_lower_leg_back","left_feet","right_feet"],t.blurBodyPart=function(t,e,n,r,o,i,a){void 0===r&&(r=[0,1]),void 0===o&&(o=3),void 0===i&&(i=3),void 0===a&&(a=!1);var s=Pt(e,o,Tt.blurred);t.width=s.width,t.height=s.height;var u=t.getContext("2d");if(Array.isArray(n)&&0===n.length)u.drawImage(s,0,0);else{var f=function(t,e,n){var r=At(It(t,{r:0,g:0,b:0,a:0},{r:0,g:0,b:0,a:255},!0,e),Tt.mask);return 0===n?r:Pt(r,n,Tt.blurredMask)}(n,r,i);u.save(),a&&Mt(t);var c=G(e),d=c[0],l=c[1];u.drawImage(e,0,0,l,d),Ot(u,f,"destination-in"),Ot(u,s,"destination-over"),u.restore()}}
/**
     * @license
     * Copyright 2020 Google Inc. All Rights Reserved.
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
     *
     * =============================================================================
     */,t.drawBokehEffect=function(t,e,n,r,o,i){void 0===r&&(r=3),void 0===o&&(o=3),void 0===i&&(i=!1);var a=Pt(e,r,Tt.blurred);t.width=a.width,t.height=a.height;var s=t.getContext("2d");if(Array.isArray(n)&&0===n.length)s.drawImage(a,0,0);else{var u=function(t,e){var n=At(It(t,{r:0,g:0,b:0,a:255},{r:0,g:0,b:0,a:0}),Tt.mask);return 0===e?n:Pt(n,e,Tt.blurredMask)}(n,o);s.save(),i&&Mt(t);var f=G(e),c=f[0],d=f[1];s.drawImage(e,0,0,d,c),Ot(s,u,"destination-in"),Ot(s,a,"destination-over"),s.restore()}},t.drawMask=function(t,e,n,r,o,i){void 0===r&&(r=.7),void 0===o&&(o=0),void 0===i&&(i=!1);var a=G(e),s=a[0],u=a[1];t.width=u,t.height=s;var f=t.getContext("2d");if(f.save(),i&&Mt(t),f.drawImage(e,0,0),f.globalAlpha=r,n){kt({width:u,height:s},n,"image","mask");var c=Pt(At(n,Tt.mask),o,Tt.blurredMask);f.drawImage(c,0,0,u,s)}f.restore()},t.drawPixelatedMask=function(t,e,n,r,o,i,a){void 0===r&&(r=.7),void 0===o&&(o=0),void 0===i&&(i=!1),void 0===a&&(a=10);var s=G(e),u=s[0];kt({width:s[1],height:u},n,"image","mask");var f=Pt(At(n,Tt.mask),o,Tt.blurredMask);t.width=f.width,t.height=f.height;var c=t.getContext("2d");c.save(),i&&Mt(t);var d=Et(Tt.lowresPartMask),l=d.getContext("2d");d.width=f.width*(1/a),d.height=f.height*(1/a),l.drawImage(f,0,0,f.width,f.height,0,0,d.width,d.height),c.imageSmoothingEnabled=!1,c.drawImage(d,0,0,d.width,d.height,0,0,t.width,t.height);for(var h=0;h<d.width;h++)c.beginPath(),c.strokeStyle="#ffffff",c.moveTo(a*h,0),c.lineTo(a*h,t.height),c.stroke();for(h=0;h<d.height;h++)c.beginPath(),c.strokeStyle="#ffffff",c.moveTo(0,a*h),c.lineTo(t.width,a*h),c.stroke();c.globalAlpha=1-r,c.drawImage(e,0,0,f.width,f.height),c.restore()},t.flipPoseHorizontal=at,t.load=function(t){return void 0===t&&(t=ct),f(this,void 0,void 0,(function(){return c(this,(function(e){return"ResNet50"===(t=function(t){if(null==(t=t||ct).architecture&&(t.architecture="MobileNetV1"),dt.indexOf(t.architecture)<0)throw new Error("Invalid architecture ".concat(t.architecture,". ")+"Should be one of ".concat(dt));if(null==t.outputStride&&(t.outputStride=16),lt[t.architecture].indexOf(t.outputStride)<0)throw new Error("Invalid outputStride ".concat(t.outputStride,". ")+"Should be one of ".concat(lt[t.architecture]," ")+"for architecture ".concat(t.architecture,"."));if(null==t.multiplier&&(t.multiplier=1),ht[t.architecture].indexOf(t.multiplier)<0)throw new Error("Invalid multiplier ".concat(t.multiplier,". ")+"Should be one of ".concat(ht[t.architecture]," ")+"for architecture ".concat(t.architecture,"."));if(null==t.quantBytes&&(t.quantBytes=4),pt.indexOf(t.quantBytes)<0)throw new Error("Invalid quantBytes ".concat(t.quantBytes,". ")+"Should be one of ".concat(pt," ")+"for architecture ".concat(t.architecture,"."));return t}(t)).architecture?[2,xt(t)]:"MobileNetV1"===t.architecture?[2,bt(t)]:[2,null]}))}))},t.resizeAndPadTo=function(t,e,n){var r=e[0],o=e[1];void 0===n&&(n=!1);var a,s,u,f,c,d,l=t.shape,h=l[0],p=l[1]/h;if(p>o/r){a=o;var m=r-(s=Math.ceil(a/p));u=0,f=0,c=Math.floor(m/2),d=r-(s+c)}else{s=r;var g=o-(a=Math.ceil(r*p));u=Math.floor(g/2),f=o-(a+u),c=0,d=0}return{resizedAndPadded:i.tidy((function(){var e;return e=n?i.image.resizeBilinear(i.reverse(t,1),[s,a]):i.image.resizeBilinear(t,[s,a]),i.pad3d(e,[[c,d],[u,f],[0,0]])})),paddedBy:[[c,d],[u,f]]}},t.scaleAndCropToInputTensorShape=rt,t.toColoredPartMask=function(t,e){if(void 0===e&&(e=Rt),Array.isArray(t)&&0===t.length)return null;for(var n,r=(n=Array.isArray(t)?t:[t])[0],o=r.width,i=r.height,a=new Uint8ClampedArray(o*i*4),s=0;s<i*o;++s){var u=4*s;a[u+0]=255,a[u+1]=255,a[u+2]=255,a[u+3]=255;for(var f=0;f<n.length;f++){var c=n[f].data[s];if(-1!==c){var d=e[c];if(!d)throw new Error("No color could be found for part id ".concat(c));a[u+0]=d[0],a[u+1]=d[1],a[u+2]=d[2],a[u+3]=255}}}return new ImageData(a,o,i)},t.toMask=It,t.version="2.2.1"}));
//# sourceMappingURL=body-pix.min.umd.js.map
