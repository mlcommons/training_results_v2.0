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

package io.bazel.rules.closure.webfiles.server;

import static com.google.common.base.MoreObjects.firstNonNull;
import static com.google.common.base.Strings.emptyToNull;

import io.bazel.rules.closure.http.HttpRequest;
import io.bazel.rules.closure.http.HttpResponse;
import javax.inject.Inject;

/**
 * Filter that disables CORS security features in the browser for the sake of convenience.
 *
 * <p>See <a href="http://www.w3.org/TR/cors/#access-control-allow-origin-response-header">W3C CORS
 * § 5.1 Access-Control-Allow-Origin Response Header</a> for more information.
 *
 * <p><b>Warning:</b> Please do not use this for production code.
 */
final class IWantToBeVulnerableToXssAttacksFilter {

  private final HttpRequest request;
  private final HttpResponse response;

  @Inject
  IWantToBeVulnerableToXssAttacksFilter(HttpRequest request, HttpResponse response) {
    this.request = request;
    this.response = response;
  }

  void apply() {
    // Certain proxies, such as the one Google uses on its corporate network, do not permit wildcard
    // for this header. We can weasel around this restriction as follows.
    response.setHeader(
        "Access-Control-Allow-Origin", firstNonNull(emptyToNull(request.getHeader("Origin")), "*"));
    response.setHeader("Access-Control-Allow-Credentials", "true");
  }
}
