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

package io.bazel.rules.closure.http;

import dagger.BindsInstance;

/** Component with lifecycle limited to individual HTTP request. */
public interface HttpRequestComponent<H extends HttpHandler> {

  /** Returns HTTP request handler. */
  H handler();

  /** Builder for {@link HttpRequestComponent}. */
  public interface Builder<
      H extends HttpHandler, R extends HttpRequestComponent<H>, B extends Builder<H, R, B>> {

    /** Binds HTTP request to request object graph. */
    @BindsInstance
    B request(HttpRequest request);

    /** Binds HTTP response to request object graph. */
    @BindsInstance
    B response(HttpResponse response);

    /** Returns the request object graph. */
    R build();
  }
}
