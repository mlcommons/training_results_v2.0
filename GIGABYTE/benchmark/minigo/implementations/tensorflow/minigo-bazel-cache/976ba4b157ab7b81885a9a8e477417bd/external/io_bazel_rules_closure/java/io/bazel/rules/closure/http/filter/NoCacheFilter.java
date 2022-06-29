// Copyright 2017 The Closure Rules Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package io.bazel.rules.closure.http.filter;

import io.bazel.rules.closure.http.HttpResponse;
import javax.inject.Inject;

/** Filter that disables client and proxy caching on all responses by default. */
public final class NoCacheFilter {

  private final HttpResponse response;

  @Inject
  public NoCacheFilter(HttpResponse response) {
    this.response = response;
  }

  public void apply() {
    if (!response.getHeader("Cache-Control").isEmpty()
        || !response.getHeader("Expires").isEmpty()) {
      return;
    }
    response.setHeader("Cache-Control", "no-cache, must-revalidate");
    response.setHeader("Expires", "0");
  }
}
