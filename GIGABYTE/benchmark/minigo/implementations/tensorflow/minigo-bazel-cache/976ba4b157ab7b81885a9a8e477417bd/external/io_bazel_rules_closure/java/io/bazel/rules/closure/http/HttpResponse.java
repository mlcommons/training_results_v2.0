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

import static com.google.common.base.MoreObjects.firstNonNull;
import static com.google.common.base.Preconditions.checkArgument;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

/** HTTP request message. */
public final class HttpResponse extends HttpMessage<HttpResponse> {

  private static final ImmutableSet<String> NO_PAYLOAD_METHODS =
      ImmutableSet.of("HEAD", "DELETE", "CONNECT", "TRACE");

  private int status = 200;
  private String message = "OK";

  public int getStatus() {
    return status;
  }

  public String getMessage() {
    return message;
  }

  public HttpResponse setStatus(int status) {
    return setStatus(status, firstNonNull(DEFAULT_STATUS_MESSAGES.get(status), this.message));
  }

  public HttpResponse setStatus(int status, String message) {
    checkArgument(!message.isEmpty() && CRLF.matchesNoneOf(message));
    checkArgument(100 <= status && status < 1000);
    this.status = status;
    this.message = message;
    return this;
  }

  @Override
  void generateFirstLineOfMessage(StringBuilder builder) {
    builder.append(getVersion());
    builder.append(' ');
    builder.append(status);
    builder.append(' ');
    builder.append(message);
    builder.append("\r\n");
    if (getHeader("server").isEmpty()) {
      builder.append("server: carlyle v1.o\r\n");
    }
    if (getHeader("date").isEmpty()) {
      builder.append("date: ");
      builder.append(DATE_FORMAT.format(new Date()));
      builder.append("\r\n");
    }
  }

  @Override
  boolean isPayloadAllowed() {
    return !NO_PAYLOAD_METHODS.contains(getMethod());
  }

  private static final SimpleDateFormat DATE_FORMAT =
      new SimpleDateFormat("EEE, dd MMM yyyy HH:mm:ss z", Locale.US);

  private static final ImmutableMap<Integer, String> DEFAULT_STATUS_MESSAGES =
      new ImmutableMap.Builder<Integer, String>()
          .put(100, "Continue")
          .put(101, "Switching Protocols")
          .put(200, "OK")
          .put(201, "Created")
          .put(202, "Accepted")
          .put(203, "Non-Authoritative Information")
          .put(204, "No Content")
          .put(205, "Reset Content")
          .put(206, "Partial Content")
          .put(300, "Multiple Choice")
          .put(301, "Moved Permanently")
          .put(302, "Found")
          .put(303, "See Other")
          .put(304, "Not Modified")
          .put(305, "Use Proxy")
          .put(307, "Temporary Redirect")
          .put(308, "Permanent Redirect")
          .put(400, "Bad Request")
          .put(401, "Unauthorized")
          .put(402, "Payment Required")
          .put(403, "Forbidden")
          .put(404, "Not Found")
          .put(405, "Method Not Allowed")
          .put(406, "Not Acceptable")
          .put(407, "Proxy Authentication Required")
          .put(408, "Request Timeout")
          .put(409, "Conflict")
          .put(410, "Gone")
          .put(411, "Length Required")
          .put(412, "Precondition Failed")
          .put(413, "Payload Too Large")
          .put(414, "URI Too Long")
          .put(415, "Unsupported Media Type")
          .put(416, "Requested Range Not Satisfiable")
          .put(417, "Expectation Failed")
          .put(421, "Misdirected Request")
          .put(426, "Upgrade Required")
          .put(428, "Precondition Required")
          .put(429, "Too Many Requests")
          .put(431, "Request Header Fields Too Large")
          .put(451, "Unavailable For Legal Reasons")
          .put(500, "Internal Server Error")
          .put(501, "Not Implemented")
          .put(502, "Bad Gateway")
          .put(503, "Service Unavailable")
          .put(504, "Gateway Timeout")
          .put(505, "HTTP Version Not Supported")
          .put(506, "Variant Also Negotiates")
          .put(507, "Variant Also Negotiates")
          .put(511, "Network Authentication Required")
          .build();
}
