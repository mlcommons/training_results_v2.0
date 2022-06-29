/*
 * Copyright 2016 The Closure Rules Authors. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.javascript.jscomp;

import java.util.ArrayList;
import java.util.List;

final class JsCheckerErrorManager extends BasicErrorManager {

  private final MessageFormatter formatter;
  final List<String> stderr = new ArrayList<>();

  JsCheckerErrorManager(MessageFormatter formatter) {
    this.formatter = formatter;
  }

  @Override
  public void println(CheckLevel level, JSError error) {
    stderr.add(error.format(level, formatter));
  }

  @Override
  public void printSummary() {
    if (getErrorCount() + getWarningCount() == 0) {
      return;
    }
    if (getTypedPercent() > 0.0) {
      stderr.add(
          String.format("%d error(s), %d warning(s), %.1f%% typed%n",
              getErrorCount(), getWarningCount(), getTypedPercent()));
    } else {
      stderr.add(
          String.format("%d error(s), %d warning(s)%n", getErrorCount(), getWarningCount()));
    }
  }
}
