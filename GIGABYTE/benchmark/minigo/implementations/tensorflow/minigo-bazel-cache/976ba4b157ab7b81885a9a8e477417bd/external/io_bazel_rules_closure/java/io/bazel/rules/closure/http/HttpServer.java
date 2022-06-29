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

import com.google.common.base.Ascii;
import com.google.common.base.Strings;
import com.google.common.io.ByteStreams;
import io.bazel.rules.closure.http.HttpParser.HttpParserError;
import java.io.IOException;
import java.io.InterruptedIOException;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.channels.ClosedByInterruptException;
import java.util.concurrent.Executor;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.annotation.concurrent.GuardedBy;
import javax.inject.Inject;

/**
 * HTTP server.
 *
 * <p>This server was designed to follow the same design principles that made Google Web Server
 * great. We proudly use threads. There is absolutely no reflection. The config file is written in
 * Java rather than XML. We use generic typing extensively so our configuration can be validated by
 * the Java compiler.
 *
 * <p>This server will only send traffic to a single endpoint, which is defined by a type parameter
 * in the component configuration. Type parameters are also used to compose web server
 * functionality. It is the responsibility of the user to implement things like request dispatch.
 *
 * <p><b>Warning:</b> This server must never be used as a frontend that accepts connections from
 * untrusted sources. You have been warned.
 *
 * @param <T> Dagger component for HTTP server
 */
public final class HttpServer<T extends HttpServerComponent<?, ?, ?, ?, ?>> {

  private static final Logger logger = Logger.getLogger(HttpServer.class.getName());

  private final T component;
  private final Executor executor;

  @Inject
  public HttpServer(T component, Executor executor) {
    this.component = component;
    this.executor = executor;
  }

  /**
   * Blockingly accepts a single connection request and delegates it to a new thread.
   *
   * <p>This method blocks for two things. First, it blocks for a connection to come in over the
   * socket. Secondly, it blocks until the executor schedules our thread.
   */
  public void handleOneConnection(ServerSocket server) throws IOException {
    @SuppressWarnings("resource") // guaranteed to enter try-finally by spawn()
    Socket socket = server.accept();
    socket.setTcpNoDelay(true);
    socket.setKeepAlive(true);
    new ConnectionThread(socket).spawn();
  }

  private class ConnectionThread implements Runnable {

    private final Socket socket;

    @GuardedBy("this")
    private boolean isReady;

    ConnectionThread(Socket socket) {
      this.socket = socket;
    }

    void spawn() throws InterruptedIOException {
      synchronized (this) {
        executor.execute(this);
        try {
          while (!isReady) {
            wait();
          }
        } catch (InterruptedException e) {
          throw new InterruptedIOException();
        }
      }
    }

    @Override
    public void run() {
      try {
        synchronized (this) {
          isReady = true;
          notify();
        }
        handleConnection();
      } catch (InterruptedException | InterruptedIOException | ClosedByInterruptException e) {
        logger.info(e.toString());
      } catch (IOException e) {
        if (Ascii.equalsIgnoreCase(Strings.nullToEmpty(e.getMessage()), "connection reset")) {
          return;
        }
      } catch (Exception e) {
        logger.log(Level.SEVERE, "Unhandled HTTP handling error", e);
      } finally {
        try {
          socket.close();
        } catch (IOException e) {
          logger.log(Level.WARNING, "Close failed", e);
        }
      }
    }

    private void handleConnection() throws Exception {
      UnbufferableInputStream input = new UnbufferableInputStream(socket.getInputStream(), 8192);
      OutputStream output = socket.getOutputStream();
      HttpConnectionComponent<?, ?, ?> connectionComponent =
          component
              .newConnectionComponentBuilder()
              .socket(socket)
              .input(input)
              .output(output)
              .build();
      while (socket.isConnected() && !socket.isClosed() && !socket.isInputShutdown()) {
        HttpRequest request;
        try {
          request = HttpParser.readHttpRequest(input);
        } catch (HttpParserError e) {
          logger.warning(e.toString());
          ByteStreams.copy(
              new HttpResponse()
                  .setStatus(400, e.getMessage())
                  .setHeader("Connection", "close")
                  .openStream(),
              output);
          return;
        }
        if (request == null) {
          break;
        }
        HttpResponse response = request.newResponse();
        try {
          connectionComponent
              .newRequestComponentBuilder()
              .request(request)
              .response(response)
              .build()
              .handler()
              .handle();
        } finally {
          response.getPayload().close();
        }
        if (response.getHeader("Connection").equals("close")) {
          break;
        }
      }
    }
  }
}
