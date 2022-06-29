// Copyright 2017 The Closure Rules Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package io.bazel.rules.closure.http;

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import javax.annotation.Nullable;

/** Ultimo HTTP message parser. */
public final class HttpParser {

  /**
   * Parses request line and headers of HTTP request.
   *
   * <p>This parser is correct and extremely lax. This implementation is Θ(n) and the stream should
   * be buffered. All decoding is ISO-8859-1. A 1mB upper bound on memory is enforced.
   *
   * @throws IOException if reading failed or premature end of stream encountered
   * @throws HttpParserError if 400 error should be sent to client and connection must be closed
   */
  @Nullable
  public static HttpRequest readHttpRequest(InputStream stream) throws IOException {
    HttpRequest request = new HttpRequest();
    StringBuilder builder = new StringBuilder(256);
    State state = State.METHOD;
    String key = "";
    int toto = 0;
    try {
      while (true) {
        int c = stream.read();
        if (c == -1) {
          if (toto == 0) {
            return null;
          } else {
            throw new HttpParserError();  // RFC7230 § 3.4
          }
        }
        if (++toto == 1024 * 1024) {
          throw new HttpParserError();  // RFC7230 § 3.2.5
        }
        switch (state) {
          case METHOD:
            if (c == ' ') {
              if (builder.length() == 0) {
                throw new HttpParserError();
              }
              request.setMethod(builder);
              builder.setLength(0);
              state = State.URI;
            } else if (c == '\r' || c == '\n') {
              break;  // RFC7230 § 3.5
            } else {
              builder.append((char) c);
            }
            break;
          case URI:
            if (c == ' ') {
              if (builder.length() == 0) {
                throw new HttpParserError();
              }
              request.setUri(URI.create(builder.toString()));
              builder.setLength(0);
              state = State.VERSION;
            } else {
              builder.append((char) c);
            }
            break;
          case VERSION:
            if (c == '\r' || c == '\n') {
              request.setVersion(builder);
              builder.setLength(0);
              state = c == '\r' ? State.CR1 : State.LF1;
            } else {
              builder.append((char) c);
            }
            break;
          case CR1:
            if (c == '\n') {
              state = State.LF1;
              break;
            }
            throw new HttpParserError();
          case LF1:
            if (c == '\r') {
              state = State.LF2;
              break;
            } else if (c == '\n') {
              request.setPayload(stream);
              return request;
            } else if (c == ' ' || c == '\t') {
              throw new HttpParserError("Line folding unacceptable");  // RFC7230 § 3.2.4
            }
            state = State.HKEY;
            // epsilon transition
          case HKEY:
            if (c == ':') {
              key = builder.toString();
              builder.setLength(0);
              state = State.HSEP;
            } else {
              builder.append((char) c);
            }
            break;
          case HSEP:
            if (c == ' ' || c == '\t') {
              break;
            }
            state = State.HVAL;
            // epsilon transition
          case HVAL:
            if (c == '\r' || c == '\n') {
              request.setHeader(key, builder.toString());
              builder.setLength(0);
              state = c == '\r' ? State.CR1 : State.LF1;
            } else {
              builder.append((char) c);
            }
            break;
          case LF2:
            if (c == '\n') {
              request.setPayload(stream);
              return request;
            }
            throw new HttpParserError();
          default:
            throw new AssertionError();
        }
      }
    } catch (IllegalArgumentException e) {
      throw new HttpParserError();
    }
  }

  /** Exception thrown when {@link HttpParser} fails, with a user message. */
  public static final class HttpParserError extends IOException {

    HttpParserError() {
      this("Malformed Request");
    }

    HttpParserError(String messageForClient) {
      super(messageForClient);
    }
  }

  private enum State { METHOD, URI, VERSION, HKEY, HSEP, HVAL, CR1, LF1, LF2 }

  private HttpParser() {}
}
