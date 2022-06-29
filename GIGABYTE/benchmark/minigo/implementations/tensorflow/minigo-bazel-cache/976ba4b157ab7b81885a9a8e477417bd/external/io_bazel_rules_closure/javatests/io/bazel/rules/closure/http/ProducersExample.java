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
import static com.google.common.net.MediaType.PLAIN_TEXT_UTF_8;
import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Strings;
import com.google.common.util.concurrent.ListenableFuture;
import dagger.BindsInstance;
import dagger.Component;
import dagger.Subcomponent;
import dagger.producers.ProducerModule;
import dagger.producers.Produces;
import dagger.producers.Production;
import dagger.producers.ProductionSubcomponent;
import io.bazel.rules.closure.http.filter.Transmitter;
import java.net.HttpURLConnection;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.URL;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import javax.inject.Inject;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link HttpServer} that demonstrates the use of Dagger Producers. */
@RunWith(JUnit4.class)
public class ProducersExample {

  @Component
  interface Server
      extends HttpServerComponent<
          Transmitter<Handler>, Connection, Connection.Builder, Request, Request.Builder> {
    HttpServer<Server> httpServer();

    @Component.Builder
    interface Builder {
      @BindsInstance Builder executor(Executor executor);
      @BindsInstance Builder productionExecutor(@Production Executor executor);
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

  @ProductionSubcomponent(modules = ResponseModule.class)
  interface Request extends HttpRequestComponent<Transmitter<Handler>> {
    ListenableFuture<byte[]> payload();

    @ProductionSubcomponent.Builder
    interface Builder
        extends HttpRequestComponent.Builder<Transmitter<Handler>, Request, Request.Builder> {}
  }

  @ProducerModule
  static final class ResponseModule {

    @Produces
    static String produceGreeting() {
      return "Hello";
    }

    @Produces
    static Integer produceTimes() {
      return 2;
    }

    @Produces
    static byte[] producePayload(String greeting, int times) {
      return Strings.repeat(greeting, times).getBytes(UTF_8);
    }
  }

  static final class Handler implements HttpHandler {
    private final HttpResponse response;
    private final Request request;

    @Inject
    Handler(HttpResponse response, Request request) {
      this.response = response;
      this.request = request;
    }

    @Override
    public void handle() throws InterruptedException, ExecutionException {
      response.setContentType(PLAIN_TEXT_UTF_8);
      response.setPayload(request.payload().get());
    }
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

  @Test
  public void simpleHtmlPage() throws Exception {
    handleOneConnection(
        DaggerProducersExample_Server.builder()
            .executor(executor)
            .productionExecutor(executor)
            .build()
            .httpServer());
    HttpURLConnection connection =
        (HttpURLConnection)
            new URL(String.format("http://localhost:%d%s", server.getLocalPort(), "/"))
                .openConnection();
    connection.connect();
    assertThat(connection.getResponseCode()).isEqualTo(200);
    assertThat(connection.getContentType()).isEqualTo(PLAIN_TEXT_UTF_8.toString());
    assertThat(toByteArray(connection.getInputStream())).isEqualTo("HelloHello".getBytes(UTF_8));
    connection.disconnect();
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
