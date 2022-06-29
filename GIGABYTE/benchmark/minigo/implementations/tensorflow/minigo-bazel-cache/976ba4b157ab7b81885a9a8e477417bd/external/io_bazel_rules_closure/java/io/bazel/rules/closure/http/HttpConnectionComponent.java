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
import java.io.OutputStream;
import java.net.Socket;

/** Component spanning multiple requests in HTTP keep-alive connection thread. */
public interface HttpConnectionComponent<
    H extends HttpHandler,
    R extends HttpRequestComponent<H>,
    B extends HttpRequestComponent.Builder<H, R, B>> {

  /** Returns new request component builder. */
  B newRequestComponentBuilder();

  /** Builder for {@link HttpConnectionComponent}. */
  public interface Builder<
      H extends HttpHandler,
      C extends HttpConnectionComponent<H, R, Z>,
      B extends HttpConnectionComponent.Builder<H, C, B, R, Z>,
      R extends HttpRequestComponent<H>,
      Z extends HttpRequestComponent.Builder<H, R, Z>> {

    /** Binds socket associated with HTTP request to request object graph. */
    @BindsInstance
    B socket(Socket socket);

    /**
     * Binds low level input stream to object graph.
     *
     * <p>This should only be injected to upgrade a connection or disable buffering. Otherwise
     * {@link HttpRequest#getPayload()} should be used.
     */
    @BindsInstance
    B input(UnbufferableInputStream input);

    /**
     * Binds low level input stream to object graph.
     *
     * <p>This should only be injected when upgrading a connection. Otherwise {@link
     * HttpResponse#setPayload(java.io.InputStream)} should be used.
     */
    @BindsInstance
    B output(OutputStream output);

    /** Returns the connection object graph. */
    C build();
  }
}
