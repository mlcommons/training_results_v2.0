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

import static com.google.common.io.ByteStreams.toByteArray;
import static com.google.common.net.MediaType.HTML_UTF_8;
import static com.google.common.net.MediaType.PLAIN_TEXT_UTF_8;
import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import dagger.BindsInstance;
import dagger.Component;
import dagger.Subcomponent;
import io.bazel.rules.closure.http.filter.Transmitter;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.URL;
import java.util.concurrent.Callable;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicReference;
import javax.inject.Inject;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link HttpServer}. */
@RunWith(JUnit4.class)
public class HttpServerTest {

  interface MagicHandler {
    void handle(HttpRequest request, HttpResponse response) throws Exception;
  }

  static class Handler implements HttpHandler {
    private final HttpRequest request;
    private final HttpResponse response;
    private final MagicHandler magic;

    @Inject
    Handler(HttpRequest request, HttpResponse response, MagicHandler magic) {
      this.request = request;
      this.response = response;
      this.magic = magic;
    }

    @Override
    public void handle() throws Exception {
      magic.handle(request, response);
    }
  }

  @Component
  interface Server
      extends HttpServerComponent<
          Transmitter<Handler>, Connection, Connection.Builder, Request, Request.Builder> {
    HttpServer<Server> httpServer();

    @Component.Builder
    interface Builder {
      @BindsInstance Builder executor(Executor executor);
      @BindsInstance Builder handler(MagicHandler magic);
      Server build();
    }
  }

  @Subcomponent
  interface Connection
      extends HttpConnectionComponent<Transmitter<Handler>, Request, Request.Builder> {
    @Subcomponent.Builder
    interface Builder
        extends HttpConnectionComponent.Builder<
            Transmitter<Handler>, Connection, Connection.Builder, Request, Request.Builder> {}
  }

  @Subcomponent
  interface Request extends HttpRequestComponent<Transmitter<Handler>> {
    @Subcomponent.Builder
    interface Builder
        extends HttpRequestComponent.Builder<Transmitter<Handler>, Request, Request.Builder> {}
  }

  private static final ExecutorService executor = Executors.newCachedThreadPool();
  private ServerSocket server;

  @Before
  public void createServer() throws Exception {
    server = new ServerSocket(0, 1, InetAddress.getByName("localhost"));
  }

  @After
  public void closeServer() throws Exception {
    server.close();
  }

  @AfterClass
  public static void closeExecutor() throws Exception {
    executor.shutdownNow();
  }

  private static HttpServer<Server> createHttpServer(MagicHandler magic) {
    return DaggerHttpServerTest_Server.builder()
        .executor(executor)
        .handler(magic)
        .build()
        .httpServer();
  }

  @Test
  public void simpleHtmlPage() throws Exception {
    final AtomicReference<HttpRequest> theRequest = new AtomicReference<>();
    final byte[] data = "<b>hello world</b>".getBytes(UTF_8);
    HttpServer<?> httpServer =
        createHttpServer(
            new MagicHandler() {
              @Override
              public void handle(HttpRequest request, HttpResponse response) throws IOException {
                theRequest.set(request);
                response.setContentType(HTML_UTF_8);
                response.setPayload(data);
              }
            });
    handleOneConnection(httpServer);
    HttpURLConnection connection = openConnection("/");
    connection.connect();
    assertThat(connection.getResponseCode()).isEqualTo(200);
    assertThat(connection.getContentType()).isEqualTo(HTML_UTF_8.toString());
    assertThat(toByteArray(connection.getInputStream())).isEqualTo(data);
    connection.disconnect();
  }

  @Test
  public void keepAlive() throws Exception {
    final AtomicReference<HttpRequest> theRequest = new AtomicReference<>();
    HttpServer<?> httpServer =
        createHttpServer(
            new MagicHandler() {
              @Override
              public void handle(HttpRequest request, HttpResponse response) throws IOException {
                theRequest.set(request);
                response.setContentType(PLAIN_TEXT_UTF_8);
                response.setPayload(request.getUri().getPath().getBytes(UTF_8));
              }
            });
    handleOneConnection(httpServer);
    HttpURLConnection connection = openConnection("");
    connection.connect();
    assertThat(toByteArray(connection.getInputStream())).isEqualTo("/".getBytes(UTF_8));
    connection = openConnection("/foo");
    connection.connect();
    assertThat(toByteArray(connection.getInputStream())).isEqualTo("/foo".getBytes(UTF_8));
    connection.disconnect();
  }

  private HttpURLConnection openConnection(String path) throws IOException {
    return (HttpURLConnection)
        new URL(String.format("http://localhost:%d%s", server.getLocalPort(), path))
            .openConnection();
  }

  private Future<Void> handleOneConnection(final HttpServer<?> httpServer) {
    return executor.submit(
        new Callable<Void>() {
          @Override
          public Void call() throws Exception {
            httpServer.handleOneConnection(server);
            return null;
          }
        });
  }
}
