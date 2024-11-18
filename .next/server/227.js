"use strict";exports.id=227,exports.ids=[227],exports.modules={6577:t=>{var e={};function r(t,e){!e.unsigned&&--t;let r=e.unsigned?0:-Math.pow(2,t),s=Math.pow(2,t)-1,i=e.moduloBitLength?Math.pow(2,e.moduloBitLength):Math.pow(2,t),h=e.moduloBitLength?Math.pow(2,e.moduloBitLength-1):Math.pow(2,t-1);return function(t,n){n||(n={});let o=+t;if(n.enforceRange){if(!Number.isFinite(o))throw TypeError("Argument is not a finite number");if((o=(o<0?-1:1)*Math.floor(Math.abs(o)))<r||o>s)throw TypeError("Argument is not in byte range");return o}if(!isNaN(o)&&n.clamp){var u;return(o=(u=o)%1==.5&&(1&u)==0?Math.floor(u):Math.round(u))<r&&(o=r),o>s&&(o=s),o}if(!Number.isFinite(o)||0===o)return 0;if(o=(o<0?-1:1)*Math.floor(Math.abs(o))%i,!e.unsigned&&o>=h)return o-i;if(e.unsigned){if(o<0)o+=i;else if(-0===o)return 0}return o}}t.exports=e,e.void=function(){},e.boolean=function(t){return!!t},e.byte=r(8,{unsigned:!1}),e.octet=r(8,{unsigned:!0}),e.short=r(16,{unsigned:!1}),e["unsigned short"]=r(16,{unsigned:!0}),e.long=r(32,{unsigned:!1}),e["unsigned long"]=r(32,{unsigned:!0}),e["long long"]=r(32,{unsigned:!1,moduloBitLength:64}),e["unsigned long long"]=r(32,{unsigned:!0,moduloBitLength:64}),e.double=function(t){let e=+t;if(!Number.isFinite(e))throw TypeError("Argument is not a finite floating-point value");return e},e["unrestricted double"]=function(t){let e=+t;if(isNaN(e))throw TypeError("Argument is NaN");return e},e.float=e.double,e["unrestricted float"]=e["unrestricted double"],e.DOMString=function(t,e){return(e||(e={}),e.treatNullAsEmptyString&&null===t)?"":String(t)},e.ByteString=function(t,e){let r;let s=String(t);for(let t=0;void 0!==(r=s.codePointAt(t));++t)if(r>255)throw TypeError("Argument is not a valid bytestring");return s},e.USVString=function(t){let e=String(t),r=e.length,s=[];for(let t=0;t<r;++t){let i=e.charCodeAt(t);if(i<55296||i>57343)s.push(String.fromCodePoint(i));else if(56320<=i&&i<=57343)s.push(String.fromCodePoint(65533));else if(t===r-1)s.push(String.fromCodePoint(65533));else{let r=e.charCodeAt(t+1);if(56320<=r&&r<=57343){let e=1023&i,h=1023&r;s.push(String.fromCodePoint(65536+1024*e+h)),++t}else s.push(String.fromCodePoint(65533))}}return s.join("")},e.Date=function(t,e){if(!(t instanceof Date))throw TypeError("Argument is not a Date object");if(!isNaN(t))return t},e.RegExp=function(t,e){return t instanceof RegExp||(t=new RegExp(t)),t}},3672:(t,e,r)=>{let s=r(9283);e.implementation=class{constructor(t){let e=t[0],r=t[1],i=null;if(void 0!==r&&"failure"===(i=s.basicURLParse(r)))throw TypeError("Invalid base URL");let h=s.basicURLParse(e,{baseURL:i});if("failure"===h)throw TypeError("Invalid URL");this._url=h}get href(){return s.serializeURL(this._url)}set href(t){let e=s.basicURLParse(t);if("failure"===e)throw TypeError("Invalid URL");this._url=e}get origin(){return s.serializeURLOrigin(this._url)}get protocol(){return this._url.scheme+":"}set protocol(t){s.basicURLParse(t+":",{url:this._url,stateOverride:"scheme start"})}get username(){return this._url.username}set username(t){s.cannotHaveAUsernamePasswordPort(this._url)||s.setTheUsername(this._url,t)}get password(){return this._url.password}set password(t){s.cannotHaveAUsernamePasswordPort(this._url)||s.setThePassword(this._url,t)}get host(){let t=this._url;return null===t.host?"":null===t.port?s.serializeHost(t.host):s.serializeHost(t.host)+":"+s.serializeInteger(t.port)}set host(t){this._url.cannotBeABaseURL||s.basicURLParse(t,{url:this._url,stateOverride:"host"})}get hostname(){return null===this._url.host?"":s.serializeHost(this._url.host)}set hostname(t){this._url.cannotBeABaseURL||s.basicURLParse(t,{url:this._url,stateOverride:"hostname"})}get port(){return null===this._url.port?"":s.serializeInteger(this._url.port)}set port(t){s.cannotHaveAUsernamePasswordPort(this._url)||(""===t?this._url.port=null:s.basicURLParse(t,{url:this._url,stateOverride:"port"}))}get pathname(){return this._url.cannotBeABaseURL?this._url.path[0]:0===this._url.path.length?"":"/"+this._url.path.join("/")}set pathname(t){this._url.cannotBeABaseURL||(this._url.path=[],s.basicURLParse(t,{url:this._url,stateOverride:"path start"}))}get search(){return null===this._url.query||""===this._url.query?"":"?"+this._url.query}set search(t){let e=this._url;if(""===t){e.query=null;return}let r="?"===t[0]?t.substring(1):t;e.query="",s.basicURLParse(r,{url:e,stateOverride:"query"})}get hash(){return null===this._url.fragment||""===this._url.fragment?"":"#"+this._url.fragment}set hash(t){if(""===t){this._url.fragment=null;return}let e="#"===t[0]?t.substring(1):t;this._url.fragment="",s.basicURLParse(e,{url:this._url,stateOverride:"fragment"})}toJSON(){return this.href}}},1125:(t,e,r)=>{let s=r(6577),i=r(4069),h=r(3672),n=i.implSymbol;function o(e){if(!this||this[n]||!(this instanceof o))throw TypeError("Failed to construct 'URL': Please use the 'new' operator, this DOM object constructor cannot be called as a function.");if(arguments.length<1)throw TypeError("Failed to construct 'URL': 1 argument required, but only "+arguments.length+" present.");let r=[];for(let t=0;t<arguments.length&&t<2;++t)r[t]=arguments[t];r[0]=s.USVString(r[0]),void 0!==r[1]&&(r[1]=s.USVString(r[1])),t.exports.setup(this,r)}o.prototype.toJSON=function(){if(!this||!t.exports.is(this))throw TypeError("Illegal invocation");let e=[];for(let t=0;t<arguments.length&&t<0;++t)e[t]=arguments[t];return this[n].toJSON.apply(this[n],e)},Object.defineProperty(o.prototype,"href",{get(){return this[n].href},set(t){t=s.USVString(t),this[n].href=t},enumerable:!0,configurable:!0}),o.prototype.toString=function(){if(!this||!t.exports.is(this))throw TypeError("Illegal invocation");return this.href},Object.defineProperty(o.prototype,"origin",{get(){return this[n].origin},enumerable:!0,configurable:!0}),Object.defineProperty(o.prototype,"protocol",{get(){return this[n].protocol},set(t){t=s.USVString(t),this[n].protocol=t},enumerable:!0,configurable:!0}),Object.defineProperty(o.prototype,"username",{get(){return this[n].username},set(t){t=s.USVString(t),this[n].username=t},enumerable:!0,configurable:!0}),Object.defineProperty(o.prototype,"password",{get(){return this[n].password},set(t){t=s.USVString(t),this[n].password=t},enumerable:!0,configurable:!0}),Object.defineProperty(o.prototype,"host",{get(){return this[n].host},set(t){t=s.USVString(t),this[n].host=t},enumerable:!0,configurable:!0}),Object.defineProperty(o.prototype,"hostname",{get(){return this[n].hostname},set(t){t=s.USVString(t),this[n].hostname=t},enumerable:!0,configurable:!0}),Object.defineProperty(o.prototype,"port",{get(){return this[n].port},set(t){t=s.USVString(t),this[n].port=t},enumerable:!0,configurable:!0}),Object.defineProperty(o.prototype,"pathname",{get(){return this[n].pathname},set(t){t=s.USVString(t),this[n].pathname=t},enumerable:!0,configurable:!0}),Object.defineProperty(o.prototype,"search",{get(){return this[n].search},set(t){t=s.USVString(t),this[n].search=t},enumerable:!0,configurable:!0}),Object.defineProperty(o.prototype,"hash",{get(){return this[n].hash},set(t){t=s.USVString(t),this[n].hash=t},enumerable:!0,configurable:!0}),t.exports={is:t=>!!t&&t[n]instanceof h.implementation,create(t,e){let r=Object.create(o.prototype);return this.setup(r,t,e),r},setup(t,e,r){r||(r={}),r.wrapper=t,t[n]=new h.implementation(e,r),t[n][i.wrapperSymbol]=t},interface:o,expose:{Window:{URL:o},Worker:{URL:o}}}},6227:(t,e,r)=>{e.URL=r(1125).interface,r(9283).serializeURL,r(9283).serializeURLOrigin,r(9283).basicURLParse,r(9283).setTheUsername,r(9283).setThePassword,r(9283).serializeHost,r(9283).serializeInteger,r(9283).parseURL},9283:(t,e,r)=>{let s=r(5477),i=r(4249),h={ftp:21,file:null,gopher:70,http:80,https:443,ws:80,wss:443},n=Symbol("failure");function o(t){return s.ucs2.decode(t).length}function u(t,e){let r=t[e];return isNaN(r)?void 0:String.fromCodePoint(r)}function a(t){return t>=48&&t<=57}function l(t){return t>=65&&t<=90||t>=97&&t<=122}function p(t){return a(t)||t>=65&&t<=70||t>=97&&t<=102}function f(t){return"."===t||"%2e"===t.toLowerCase()}function c(t){return 2===t.length&&l(t.codePointAt(0))&&(":"===t[1]||"|"===t[1])}function g(t){return void 0!==h[t]}function b(t){return g(t.scheme)}function m(t){let e=t.toString(16).toUpperCase();return 1===e.length&&(e="0"+e),"%"+e}function d(t){return t<=31||t>126}let y=new Set([32,34,35,60,62,63,96,123,125]);function w(t){return d(t)||y.has(t)}let S=new Set([47,58,59,61,64,91,92,93,94,124]);function v(t){return w(t)||S.has(t)}function U(t,e){let r=String.fromCodePoint(t);return e(t)?function(t){let e=new Buffer(t),r="";for(let t=0;t<e.length;++t)r+=m(e[t]);return r}(r):r}function O(t,e){if("["===t[0])return"]"!==t[t.length-1]?n:function(t){let e=[0,0,0,0,0,0,0,0],r=0,i=null,h=0;if(58===(t=s.ucs2.decode(t))[h]){if(58!==t[h+1])return n;h+=2,i=++r}for(;h<t.length;){if(8===r)return n;if(58===t[h]){if(null!==i)return n;++h,i=++r;continue}let s=0,o=0;for(;o<4&&p(t[h]);)s=16*s+parseInt(u(t,h),16),++h,++o;if(46===t[h]){if(0===o||(h-=o,r>6))return n;let s=0;for(;void 0!==t[h];){let i=null;if(s>0){if(46!==t[h]||!(s<4))return n;++h}if(!a(t[h]))return n;for(;a(t[h]);){let e=parseInt(u(t,h));if(null===i)i=e;else{if(0===i)return n;i=10*i+e}if(i>255)return n;++h}e[r]=256*e[r]+i,(2==++s||4===s)&&++r}if(4!==s)return n;break}if(58===t[h]){if(void 0===t[++h])return n}else if(void 0!==t[h])return n;e[r]=s,++r}if(null!==i){let t=r-i;for(r=7;0!==r&&t>0;){let s=e[i+t-1];e[i+t-1]=e[r],e[r]=s,--r,--t}}else if(null===i&&8!==r)return n;return e}(t.substring(1,t.length-1));if(!e)return function(t){if(-1!==t.search(/\u0000|\u0009|\u000A|\u000D|\u0020|#|\/|:|\?|@|\[|\\|\]/))return n;let e="",r=s.ucs2.decode(t);for(let t=0;t<r.length;++t)e+=U(r[t],d);return e}(t);let r=function(t){let e=new Buffer(t),r=[];for(let t=0;t<e.length;++t)37!==e[t]?r.push(e[t]):37===e[t]&&p(e[t+1])&&p(e[t+2])?(r.push(parseInt(e.slice(t+1,t+3).toString(),16)),t+=2):r.push(e[t]);return new Buffer(r).toString()}(t),h=i.toASCII(r,!1,i.PROCESSING_OPTIONS.NONTRANSITIONAL,!1);if(null===h||-1!==h.search(/\u0000|\u0009|\u000A|\u000D|\u0020|#|%|\/|:|\?|@|\[|\\|\]/))return n;let o=function(t){let e=t.split(".");if(""===e[e.length-1]&&e.length>1&&e.pop(),e.length>4)return t;let r=[];for(let s of e){if(""===s)return t;let e=function(t){let e=10;return(t.length>=2&&"0"===t.charAt(0)&&"x"===t.charAt(1).toLowerCase()?(t=t.substring(2),e=16):t.length>=2&&"0"===t.charAt(0)&&(t=t.substring(1),e=8),""===t)?0:(10===e?/[^0-9]/:16===e?/[^0-9A-Fa-f]/:/[^0-7]/).test(t)?n:parseInt(t,e)}(s);if(e===n)return t;r.push(e)}for(let t=0;t<r.length-1;++t)if(r[t]>255)return n;if(r[r.length-1]>=Math.pow(256,5-r.length))return n;let s=r.pop(),i=0;for(let t of r)s+=t*Math.pow(256,3-i),++i;return s}(h);return"number"==typeof o||o===n?o:h}function L(t){return"number"==typeof t?function(t){let e="",r=t;for(let t=1;t<=4;++t)e=String(r%256)+e,4!==t&&(e="."+e),r=Math.floor(r/256);return e}(t):t instanceof Array?"["+function(t){let e="",r=function(t){let e=null,r=1,s=null,i=0;for(let h=0;h<t.length;++h)0!==t[h]?(i>r&&(e=s,r=i),s=null,i=0):(null===s&&(s=h),++i);return i>r&&(e=s,r=i),{idx:e,len:r}}(t).idx,s=!1;for(let i=0;i<=7;++i)if(!s||0!==t[i]){if(s&&(s=!1),r===i){e+=0===i?"::":":",s=!0;continue}e+=t[i].toString(16),7!==i&&(e+=":")}return e}(t)+"]":t}function P(t){var e;let r=t.path;!(0===r.length||"file"===t.scheme&&1===r.length&&(e=r[0],/^[A-Za-z]:$/.test(e)))&&r.pop()}function R(t){return""!==t.username||""!==t.password}function E(t,e,r,i,h){if(this.pointer=0,this.input=t,this.base=e||null,this.encodingOverride=r||"utf-8",this.stateOverride=h,this.url=i,this.failure=!1,this.parseError=!1,!this.url){this.url={scheme:"",username:"",password:"",host:null,port:null,path:[],query:null,fragment:null,cannotBeABaseURL:!1};let t=this.input.replace(/^[\u0000-\u001F\u0020]+|[\u0000-\u001F\u0020]+$/g,"");t!==this.input&&(this.parseError=!0),this.input=t}let o=this.input.replace(/\u0009|\u000A|\u000D/g,"");for(o!==this.input&&(this.parseError=!0),this.input=o,this.state=h||"scheme start",this.buffer="",this.atFlag=!1,this.arrFlag=!1,this.passwordTokenSeenFlag=!1,this.input=s.ucs2.decode(this.input);this.pointer<=this.input.length;++this.pointer){let t=this.input[this.pointer],e=isNaN(t)?void 0:String.fromCodePoint(t),r=this["parse "+this.state](t,e);if(r){if(r===n){this.failure=!0;break}}else break}}E.prototype["parse scheme start"]=function(t,e){if(l(t))this.buffer+=e.toLowerCase(),this.state="scheme";else{if(this.stateOverride)return this.parseError=!0,n;this.state="no scheme",--this.pointer}return!0},E.prototype["parse scheme"]=function(t,e){if(l(t)||a(t)||43===t||45===t||46===t)this.buffer+=e.toLowerCase();else if(58===t){if(this.stateOverride&&(b(this.url)&&!g(this.buffer)||!b(this.url)&&g(this.buffer)||(R(this.url)||null!==this.url.port)&&"file"===this.buffer||"file"===this.url.scheme&&(""===this.url.host||null===this.url.host))||(this.url.scheme=this.buffer,this.buffer="",this.stateOverride))return!1;"file"===this.url.scheme?((47!==this.input[this.pointer+1]||47!==this.input[this.pointer+2])&&(this.parseError=!0),this.state="file"):b(this.url)&&null!==this.base&&this.base.scheme===this.url.scheme?this.state="special relative or authority":b(this.url)?this.state="special authority slashes":47===this.input[this.pointer+1]?(this.state="path or authority",++this.pointer):(this.url.cannotBeABaseURL=!0,this.url.path.push(""),this.state="cannot-be-a-base-URL path")}else{if(this.stateOverride)return this.parseError=!0,n;this.buffer="",this.state="no scheme",this.pointer=-1}return!0},E.prototype["parse no scheme"]=function(t){return null===this.base||this.base.cannotBeABaseURL&&35!==t?n:(this.base.cannotBeABaseURL&&35===t?(this.url.scheme=this.base.scheme,this.url.path=this.base.path.slice(),this.url.query=this.base.query,this.url.fragment="",this.url.cannotBeABaseURL=!0,this.state="fragment"):("file"===this.base.scheme?this.state="file":this.state="relative",--this.pointer),!0)},E.prototype["parse special relative or authority"]=function(t){return 47===t&&47===this.input[this.pointer+1]?(this.state="special authority ignore slashes",++this.pointer):(this.parseError=!0,this.state="relative",--this.pointer),!0},E.prototype["parse path or authority"]=function(t){return 47===t?this.state="authority":(this.state="path",--this.pointer),!0},E.prototype["parse relative"]=function(t){return this.url.scheme=this.base.scheme,isNaN(t)?(this.url.username=this.base.username,this.url.password=this.base.password,this.url.host=this.base.host,this.url.port=this.base.port,this.url.path=this.base.path.slice(),this.url.query=this.base.query):47===t?this.state="relative slash":63===t?(this.url.username=this.base.username,this.url.password=this.base.password,this.url.host=this.base.host,this.url.port=this.base.port,this.url.path=this.base.path.slice(),this.url.query="",this.state="query"):35===t?(this.url.username=this.base.username,this.url.password=this.base.password,this.url.host=this.base.host,this.url.port=this.base.port,this.url.path=this.base.path.slice(),this.url.query=this.base.query,this.url.fragment="",this.state="fragment"):b(this.url)&&92===t?(this.parseError=!0,this.state="relative slash"):(this.url.username=this.base.username,this.url.password=this.base.password,this.url.host=this.base.host,this.url.port=this.base.port,this.url.path=this.base.path.slice(0,this.base.path.length-1),this.state="path",--this.pointer),!0},E.prototype["parse relative slash"]=function(t){return b(this.url)&&(47===t||92===t)?(92===t&&(this.parseError=!0),this.state="special authority ignore slashes"):47===t?this.state="authority":(this.url.username=this.base.username,this.url.password=this.base.password,this.url.host=this.base.host,this.url.port=this.base.port,this.state="path",--this.pointer),!0},E.prototype["parse special authority slashes"]=function(t){return 47===t&&47===this.input[this.pointer+1]?(this.state="special authority ignore slashes",++this.pointer):(this.parseError=!0,this.state="special authority ignore slashes",--this.pointer),!0},E.prototype["parse special authority ignore slashes"]=function(t){return 47!==t&&92!==t?(this.state="authority",--this.pointer):this.parseError=!0,!0},E.prototype["parse authority"]=function(t,e){if(64===t){this.parseError=!0,this.atFlag&&(this.buffer="%40"+this.buffer),this.atFlag=!0;let t=o(this.buffer);for(let e=0;e<t;++e){let t=this.buffer.codePointAt(e);if(58===t&&!this.passwordTokenSeenFlag){this.passwordTokenSeenFlag=!0;continue}let r=U(t,v);this.passwordTokenSeenFlag?this.url.password+=r:this.url.username+=r}this.buffer=""}else if(isNaN(t)||47===t||63===t||35===t||b(this.url)&&92===t){if(this.atFlag&&""===this.buffer)return this.parseError=!0,n;this.pointer-=o(this.buffer)+1,this.buffer="",this.state="host"}else this.buffer+=e;return!0},E.prototype["parse hostname"]=E.prototype["parse host"]=function(t,e){if(this.stateOverride&&"file"===this.url.scheme)--this.pointer,this.state="file host";else if(58!==t||this.arrFlag){if(isNaN(t)||47===t||63===t||35===t||b(this.url)&&92===t){if(--this.pointer,b(this.url)&&""===this.buffer)return this.parseError=!0,n;if(this.stateOverride&&""===this.buffer&&(R(this.url)||null!==this.url.port))return this.parseError=!0,!1;let t=O(this.buffer,b(this.url));if(t===n)return n;if(this.url.host=t,this.buffer="",this.state="path start",this.stateOverride)return!1}else 91===t?this.arrFlag=!0:93===t&&(this.arrFlag=!1),this.buffer+=e}else{if(""===this.buffer)return this.parseError=!0,n;let t=O(this.buffer,b(this.url));if(t===n)return n;if(this.url.host=t,this.buffer="",this.state="port","hostname"===this.stateOverride)return!1}return!0},E.prototype["parse port"]=function(t,e){if(a(t))this.buffer+=e;else{if(!(isNaN(t)||47===t||63===t||35===t||b(this.url)&&92===t)&&!this.stateOverride)return this.parseError=!0,n;if(""!==this.buffer){let t=parseInt(this.buffer);if(t>65535)return this.parseError=!0,n;this.url.port=t===h[this.url.scheme]?null:t,this.buffer=""}if(this.stateOverride)return!1;this.state="path start",--this.pointer}return!0};let N=new Set([47,92,63,35]);E.prototype["parse file"]=function(t){if(this.url.scheme="file",47===t||92===t)92===t&&(this.parseError=!0),this.state="file slash";else if(null!==this.base&&"file"===this.base.scheme){if(isNaN(t))this.url.host=this.base.host,this.url.path=this.base.path.slice(),this.url.query=this.base.query;else if(63===t)this.url.host=this.base.host,this.url.path=this.base.path.slice(),this.url.query="",this.state="query";else if(35===t)this.url.host=this.base.host,this.url.path=this.base.path.slice(),this.url.query=this.base.query,this.url.fragment="",this.state="fragment";else{var e;this.input.length-this.pointer-1!=0&&(e=this.input[this.pointer+1],l(t)&&(58===e||124===e))&&(!(this.input.length-this.pointer-1>=2)||N.has(this.input[this.pointer+2]))?this.parseError=!0:(this.url.host=this.base.host,this.url.path=this.base.path.slice(),P(this.url)),this.state="path",--this.pointer}}else this.state="path",--this.pointer;return!0},E.prototype["parse file slash"]=function(t){if(47===t||92===t)92===t&&(this.parseError=!0),this.state="file host";else{if(null!==this.base&&"file"===this.base.scheme){var e;2===(e=this.base.path[0]).length&&l(e.codePointAt(0))&&":"===e[1]?this.url.path.push(this.base.path[0]):this.url.host=this.base.host}this.state="path",--this.pointer}return!0},E.prototype["parse file host"]=function(t,e){if(isNaN(t)||47===t||92===t||63===t||35===t){if(--this.pointer,!this.stateOverride&&c(this.buffer))this.parseError=!0,this.state="path";else if(""===this.buffer){if(this.url.host="",this.stateOverride)return!1;this.state="path start"}else{let t=O(this.buffer,b(this.url));if(t===n)return n;if("localhost"===t&&(t=""),this.url.host=t,this.stateOverride)return!1;this.buffer="",this.state="path start"}}else this.buffer+=e;return!0},E.prototype["parse path start"]=function(t){return b(this.url)?(92===t&&(this.parseError=!0),this.state="path",47!==t&&92!==t&&--this.pointer):this.stateOverride||63!==t?this.stateOverride||35!==t?void 0!==t&&(this.state="path",47!==t&&--this.pointer):(this.url.fragment="",this.state="fragment"):(this.url.query="",this.state="query"),!0},E.prototype["parse path"]=function(t){if(isNaN(t)||47===t||b(this.url)&&92===t||!this.stateOverride&&(63===t||35===t)){var e;if((b(this.url)&&92===t&&(this.parseError=!0),".."===(e=(e=this.buffer).toLowerCase())||"%2e."===e||".%2e"===e||"%2e%2e"===e)?(P(this.url),47===t||b(this.url)&&92===t||this.url.path.push("")):f(this.buffer)&&47!==t&&!(b(this.url)&&92===t)?this.url.path.push(""):f(this.buffer)||("file"===this.url.scheme&&0===this.url.path.length&&c(this.buffer)&&(""!==this.url.host&&null!==this.url.host&&(this.parseError=!0,this.url.host=""),this.buffer=this.buffer[0]+":"),this.url.path.push(this.buffer)),this.buffer="","file"===this.url.scheme&&(void 0===t||63===t||35===t))for(;this.url.path.length>1&&""===this.url.path[0];)this.parseError=!0,this.url.path.shift();63===t&&(this.url.query="",this.state="query"),35===t&&(this.url.fragment="",this.state="fragment")}else 37!==t||p(this.input[this.pointer+1])&&p(this.input[this.pointer+2])||(this.parseError=!0),this.buffer+=U(t,w);return!0},E.prototype["parse cannot-be-a-base-URL path"]=function(t){return 63===t?(this.url.query="",this.state="query"):35===t?(this.url.fragment="",this.state="fragment"):(isNaN(t)||37===t||(this.parseError=!0),37!==t||p(this.input[this.pointer+1])&&p(this.input[this.pointer+2])||(this.parseError=!0),isNaN(t)||(this.url.path[0]=this.url.path[0]+U(t,d))),!0},E.prototype["parse query"]=function(t,e){if(isNaN(t)||!this.stateOverride&&35===t){b(this.url)&&"ws"!==this.url.scheme&&"wss"!==this.url.scheme||(this.encodingOverride="utf-8");let e=new Buffer(this.buffer);for(let t=0;t<e.length;++t)e[t]<33||e[t]>126||34===e[t]||35===e[t]||60===e[t]||62===e[t]?this.url.query+=m(e[t]):this.url.query+=String.fromCodePoint(e[t]);this.buffer="",35===t&&(this.url.fragment="",this.state="fragment")}else 37!==t||p(this.input[this.pointer+1])&&p(this.input[this.pointer+2])||(this.parseError=!0),this.buffer+=e;return!0},E.prototype["parse fragment"]=function(t){return isNaN(t)||(0===t?this.parseError=!0:(37!==t||p(this.input[this.pointer+1])&&p(this.input[this.pointer+2])||(this.parseError=!0),this.url.fragment+=U(t,d))),!0},t.exports.serializeURL=function(t,e){let r=t.scheme+":";if(null!==t.host?(r+="//",(""!==t.username||""!==t.password)&&(r+=t.username,""!==t.password&&(r+=":"+t.password),r+="@"),r+=L(t.host),null!==t.port&&(r+=":"+t.port)):null===t.host&&"file"===t.scheme&&(r+="//"),t.cannotBeABaseURL)r+=t.path[0];else for(let e of t.path)r+="/"+e;return null!==t.query&&(r+="?"+t.query),e||null===t.fragment||(r+="#"+t.fragment),r},t.exports.serializeURLOrigin=function(e){switch(e.scheme){case"blob":try{return t.exports.serializeURLOrigin(t.exports.parseURL(e.path[0]))}catch(t){return"null"}case"ftp":case"gopher":case"http":case"https":case"ws":case"wss":var r;let s;return s=(r={scheme:e.scheme,host:e.host,port:e.port}).scheme+"://"+L(r.host),null!==r.port&&(s+=":"+r.port),s;case"file":return"file://";default:return"null"}},t.exports.basicURLParse=function(t,e){void 0===e&&(e={});let r=new E(t,e.baseURL,e.encodingOverride,e.url,e.stateOverride);return r.failure?"failure":r.url},t.exports.setTheUsername=function(t,e){t.username="";let r=s.ucs2.decode(e);for(let e=0;e<r.length;++e)t.username+=U(r[e],v)},t.exports.setThePassword=function(t,e){t.password="";let r=s.ucs2.decode(e);for(let e=0;e<r.length;++e)t.password+=U(r[e],v)},t.exports.serializeHost=L,t.exports.cannotHaveAUsernamePasswordPort=function(t){return null===t.host||""===t.host||t.cannotBeABaseURL||"file"===t.scheme},t.exports.serializeInteger=function(t){return String(t)},t.exports.parseURL=function(e,r){return void 0===r&&(r={}),t.exports.basicURLParse(e,{baseURL:r.baseURL,encodingOverride:r.encodingOverride})}},4069:t=>{t.exports.mixin=function(t,e){let r=Object.getOwnPropertyNames(e);for(let s=0;s<r.length;++s)Object.defineProperty(t,r[s],Object.getOwnPropertyDescriptor(e,r[s]))},t.exports.wrapperSymbol=Symbol("wrapper"),t.exports.implSymbol=Symbol("impl"),t.exports.wrapperForImpl=function(e){return e[t.exports.wrapperSymbol]},t.exports.implForWrapper=function(e){return e[t.exports.implSymbol]}}};