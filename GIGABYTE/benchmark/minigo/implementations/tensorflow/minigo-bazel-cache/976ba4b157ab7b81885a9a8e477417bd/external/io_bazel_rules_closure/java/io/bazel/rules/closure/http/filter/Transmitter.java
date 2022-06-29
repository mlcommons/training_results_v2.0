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

import com.google.common.collect.ImmutableSet;
import com.google.common.io.ByteStreams;
import com.google.common.net.MediaType;
import io.bazel.rules.closure.http.HttpHandler;
import io.bazel.rules.closure.http.HttpRequest;
import io.bazel.rules.closure.http.HttpResponse;
import java.io.IOException;
import java.io.InterruptedIOException;
import java.io.OutputStream;
import java.nio.channels.ClosedByInterruptException;
import java.nio.charset.StandardCharsets;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.inject.Inject;

/**
 * Delegate handler that copies response to socket and sends 500 when crashes happen.
 *
 * <p>The user should choose this as the outermost request handler, unless functionality like web
 * socket connection upgrading is desired.
 */
public final class Transmitter<T extends HttpHandler> implements HttpHandler {

  private static final Logger logger = Logger.getLogger(Transmitter.class.getName());

  private static final ImmutableSet<String> SUPPORTED_VERSIONS =
      ImmutableSet.of("HTTP/1.0", "HTTP/1.1");

  private final HttpHandler delegate;
  private final HttpRequest request;
  private final HttpResponse response;
  private final OutputStream output;

  @Inject
  public Transmitter(T delegate, HttpRequest request, HttpResponse response, OutputStream output) {
    this.delegate = delegate;
    this.request = request;
    this.response = response;
    this.output = output;
  }

  /** Handles HTTP request and returns {@code true} if connection can be re-used. */
  @Override
  public void handle() throws IOException {
    if (!SUPPORTED_VERSIONS.contains(request.getVersion())) {
      ByteStreams.copy(
          response.setStatus(505).setHeader("Connection", "close").openStream(), output);
      return;
    }
    try {
      delegate.handle();
    } catch (InterruptedIOException e) {
      throw e;
    } catch (InterruptedException | ClosedByInterruptException e) {
      throw new InterruptedIOException();
    } catch (Exception t) {
      logger.log(Level.SEVERE, "HTTP handling failed", t);
      response.setHeader("Connection", "close");
      ByteStreams.copy(
          request
              .newResponse()
              .setStatus(500)
              .setHeader("Connection", "close")
              .setContentType(MediaType.PLAIN_TEXT_UTF_8)
              .setPayload("500 Error :(".getBytes(StandardCharsets.UTF_8))
              .openStream(),
          output);
      return;
    }
    ByteStreams.exhaust(request.getPayload());
    ByteStreams.copy(response.openStream(), output);
  }
}
