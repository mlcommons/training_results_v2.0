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
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Strings.emptyToNull;
import static com.google.common.base.Strings.nullToEmpty;

import com.google.common.base.Ascii;
import com.google.common.base.CharMatcher;
import com.google.common.collect.Ordering;
import com.google.common.io.ByteSource;
import com.google.common.io.ByteStreams;
import com.google.common.io.CharStreams;
import com.google.common.net.MediaType;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.SequenceInputStream;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.Map;
import java.util.TreeMap;
import javax.annotation.Nullable;

/**
 * HTTP message.
 *
 * @param <T> subclass type
 */
public abstract class HttpMessage<T extends HttpMessage<?>> extends ByteSource {

  static final CharMatcher CRLF = CharMatcher.anyOf("\r\n");
  static final ByteArrayInputStream emptyStream = new ByteArrayInputStream(new byte[0]);

  private final Map<String, String> headers = new TreeMap<>(Ordering.natural());

  private String method = "GET";
  private String version = "HTTP/1.1";
  @Nullable private MediaType contentType;
  private long contentLength = -1;
  private InputStream payload = emptyStream;

  HttpMessage() {}

  abstract void generateFirstLineOfMessage(StringBuilder builder);

  abstract boolean isPayloadAllowed();

  public final String getMethod() {
    return method;
  }

  public final String getVersion() {
    return version;
  }

  public final MediaType getContentType() {
    return firstNonNull(contentType, MediaType.OCTET_STREAM);
  }

  public final long getContentLength() {
    return contentLength;
  }

  public final String getHeader(String name) {
    return nullToEmpty(headers.get(Ascii.toLowerCase(name)));
  }

  public final Map<String, String> getHeaders() {
    return Collections.unmodifiableMap(headers);
  }

  public final InputStream getPayload() {
    return payload;
  }

  public final T setMethod(CharSequence method) {
    this.method = Ascii.toUpperCase(method);
    if (!isPayloadAllowed()) {
      setContentLength(0);
    }
    return castThis();
  }

  public final T setVersion(CharSequence version) {
    this.version = Ascii.toUpperCase(version);
    return castThis();
  }

  public final T setContentType(@Nullable MediaType contentType) {
    if (!isPayloadAllowed()) {
      contentType = null;
    }
    this.contentType = contentType;
    if (contentType == null) {
      headers.remove("content-type");
    } else {
      headers.put("content-type", contentType.toString());
    }
    return castThis();
  }

  public final T setContentLength(long contentLength) {
    checkArgument(contentLength >= -1);
    if (contentLength > 0 && !isPayloadAllowed()) {
      contentLength = -1;
    }
    this.contentLength = contentLength;
    if (contentLength == -1) {
      setHeader("Connection", "close");
    }
    if (contentLength >= 0) {
      headers.put("content-length", Long.toString(contentLength));
    } else {
      headers.remove("content-length");
    }
    return castThis();
  }

  public final T setHeader(String name, @Nullable String value) {
    checkArgument(CharMatcher.whitespace().matchesNoneOf(name));
    name = Ascii.toLowerCase(name);
    value = emptyToNull(value);
    switch (name) {
      case "content-type":
        setContentType(value == null ? null : MediaType.parse(value));
        break;
      case "content-length":
        setContentLength(value == null ? -1 : Long.parseLong(value));
        break;
      default:
        if (value == null) {
          headers.remove(name);
        } else {
          checkArgument(CRLF.matchesNoneOf(value));
          headers.put(name, checkNotNull(value));
        }
    }
    return castThis();
  }

  public final T setPayload(InputStream payload) {
    checkNotNull(payload);
    if (isPayloadAllowed()) {
      if (contentLength == -1) {
        setHeader("Connection", "close");
        this.payload = payload;
      } else {
        this.payload = ByteStreams.limit(payload, contentLength);
      }
    }
    return castThis();
  }

  public final T setPayload(byte[] data) {
    setContentLength(data.length);
    setPayload(new ByteArrayInputStream(data));
    return castThis();
  }

  @Override
  public final InputStream openStream() throws IOException {
    StringBuilder builder = new StringBuilder(1024);
    generateHeaders(builder);
    return new SequenceInputStream(
        new ByteArrayInputStream(builder.toString().getBytes(StandardCharsets.ISO_8859_1)),
        payload);
  }

  @Override
  public final int hashCode() {
    throw new UnsupportedOperationException();
  }

  @Override
  public final boolean equals(Object other) {
    throw new UnsupportedOperationException();
  }

  @Override
  public final String toString() {
    StringBuilder builder = new StringBuilder(1024);
    generateHeaders(builder);
    Charset charset = getContentType().charset().or(StandardCharsets.UTF_8);
    try {
      if (!(payload instanceof ByteArrayInputStream)) {
        payload = new ByteArrayInputStream(ByteStreams.toByteArray(payload));
      }
      payload.mark(-1);
      CharStreams.copy(new InputStreamReader(payload, charset), builder);
      payload.reset();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    return builder.toString();
  }

  private void generateHeaders(StringBuilder builder) {
    generateFirstLineOfMessage(builder);
    for (Map.Entry<String, String> header : getHeaders().entrySet()) {
      builder.append(header.getKey());
      builder.append(": ");
      builder.append(header.getValue());
      builder.append("\r\n");
    }
    builder.append("\r\n");
  }

  @SuppressWarnings("unchecked")
  private final T castThis() {
    return (T) this;
  }
}
