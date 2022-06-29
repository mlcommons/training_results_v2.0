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

import io.bazel.rules.closure.http.HttpHandler;
import io.bazel.rules.closure.http.HttpRequest;
import io.bazel.rules.closure.http.HttpResponse;
import java.net.Socket;
import java.util.logging.Logger;
import javax.inject.Inject;

/** Filter that logs all requests and the responses. */
public final class LoggingFilter<T extends HttpHandler> implements HttpHandler {

  private static final Logger logger = Logger.getLogger(LoggingFilter.class.getName());

  private final HttpHandler delegate;
  private final HttpRequest request;
  private final HttpResponse response;
  private final Socket socket;

  @Inject
  public LoggingFilter(T delegate, HttpRequest request, HttpResponse response, Socket socket) {
    this.delegate = delegate;
    this.request = request;
    this.response = response;
    this.socket = socket;
  }

  @Override
  public void handle() throws Exception {
    long start = System.nanoTime();
    delegate.handle();
    logger.info(
        String.format(
            "%,dµs %s %d %s %s",
            (System.nanoTime() - start) / 1000,
            socket.getRemoteSocketAddress(),
            response.getStatus(),
            request.getMethod(),
            request.getUri()));
  }
}
