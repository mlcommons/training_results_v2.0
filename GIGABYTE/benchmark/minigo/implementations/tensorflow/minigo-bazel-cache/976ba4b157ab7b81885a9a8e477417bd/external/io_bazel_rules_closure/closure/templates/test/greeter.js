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

goog.provide('io.bazel.rules.closure.Greeter');

goog.require('goog.soy');
goog.require('io.bazel.rules.closure.soy.greeter');



/**
 * Greeter page.
 * @param {string} name Name of person to greet.
 * @constructor
 * @final
 */
io.bazel.rules.closure.Greeter = function(name) {

  /**
   * Name of person to greet.
   * @private {string}
   * @const
   */
  this.name_ = name;
};


/**
 * Renders HTML greeting as document body.
 */
io.bazel.rules.closure.Greeter.prototype.greet = function() {
  goog.soy.renderElement(goog.global.document.body,
                         io.bazel.rules.closure.soy.greeter.greet,
                         {name: this.name_});
};
