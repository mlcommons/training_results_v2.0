// Copyright 2016 The Closure Rules Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Example of how to use @export.


/**
 * Function that goes in binary, with name minimization, or possibly inlined
 * entirely, because it's part of the call graph of an @export function in a
 * module that is listed under entry_points in closure_js_binary().
 */
function myNameWillBeMinified() {
  console.log('hi');
}


/**
 * Function we want to be able to call from our HTML page.
 *
 * The Closure Compiler will see the @export JSDoc annotation below and
 * generate the following synthetic code in the global scope:
 *
 *   goog.exportSymbol('iWillGoIntoTheBinary', iWillGoIntoTheBinary);
 *
 * That makes the minified version of this function still available in the
 * global namespace. HOWEVER this function is still considered dead code,
 * because nothing in this file actually calls it. So you still need to list
 * this file under entry_points.
 *
 * You can also use @export for property names on classes, to prevent them
 * from being minified.
 *
 * @export
 */
function iWillGoIntoTheBinary() {
  myNameWillBeMinified();
}


/**
 * See export.js for more information.
 */
function iWillGetPrunedByTheCompiler() {
  console.log('no one loves me :(');
}
