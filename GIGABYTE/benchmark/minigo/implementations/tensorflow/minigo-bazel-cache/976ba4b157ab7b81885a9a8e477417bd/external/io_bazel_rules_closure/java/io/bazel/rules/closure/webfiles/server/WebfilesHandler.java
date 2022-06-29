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

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.net.MediaType;
import io.bazel.rules.closure.Webpath;
import io.bazel.rules.closure.http.HttpHandler;
import io.bazel.rules.closure.http.HttpRequest;
import io.bazel.rules.closure.http.HttpResponse;
import java.io.IOException;
import java.util.logging.Logger;
import javax.inject.Inject;

/** Web app for webfiles development web server. */
final class WebfilesHandler implements HttpHandler {

  private static final Logger logger = Logger.getLogger(WebfilesHandler.class.getName());

  private final HttpRequest request;
  private final HttpResponse response;
  private final FileServer fileServer;
  private final ListingPage listingPage;

  @Inject
  WebfilesHandler(
      HttpRequest request, HttpResponse response, FileServer fileServer, ListingPage listingPage) {
    this.request = request;
    this.response = response;
    this.fileServer = fileServer;
    this.listingPage = listingPage;
  }

  /** Handles HTTP request to webfiles dev web server. */
  @Override
  public void handle() throws IOException {
    if (!request.getMethod().equals("GET") && !request.getMethod().equals("HEAD")) {
      response.setStatus(405);
      response.setHeader("Allow", "GET, HEAD");
      serveError();
      return;
    }
    Webpath webpath = Webpath.get(request.getUri().getPath()).normalize();
    if (!webpath.isAbsolute()) {
      response.setStatus(400, "Bad Request URI Path");
      serveError();
      return;
    }
    if (!fileServer.serve(webpath)) {
      response.setStatus(404);
      listingPage.serve(webpath);
    }
  }

  private void serveError() {
    logger.info(
        String.format(
            "sending %d for %s: %s",
            response.getStatus(), request.getUri(), response.getMessage()));
    response.setContentType(MediaType.PLAIN_TEXT_UTF_8);
    response.setPayload(response.getMessage().getBytes(UTF_8));
  }
}
