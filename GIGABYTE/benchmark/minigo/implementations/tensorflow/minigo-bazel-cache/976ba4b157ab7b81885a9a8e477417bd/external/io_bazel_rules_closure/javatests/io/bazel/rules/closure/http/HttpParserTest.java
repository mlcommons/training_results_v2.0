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
import static com.google.common.net.MediaType.OCTET_STREAM;
import static com.google.common.truth.Truth.assertThat;
import static io.bazel.rules.closure.http.HttpParser.readHttpRequest;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Joiner;
import com.google.common.base.Strings;
import io.bazel.rules.closure.http.HttpParser.HttpParserError;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.net.URI;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link HttpParser}. */
@RunWith(JUnit4.class)
public class HttpParserTest {

  @Rule
  public final ExpectedException thrown = ExpectedException.none();

  @Test
  public void emptyStream_returnsNull() throws Exception {
    assertThat(readHttpRequest(stream())).isNull();
  }

  @Test
  public void requestMessageWithNoHeadersOrPayload() throws Exception {
    HttpRequest request = readHttpRequest(stream("GET /foo HTTP/1.0", "", ""));
    assertThat(request.getMethod()).isEqualTo("GET");
    assertThat(request.getUri()).isEqualTo(URI.create("/foo"));
    assertThat(request.getVersion()).isEqualTo("HTTP/1.0");
    assertThat(request.getContentType()).isEqualTo(OCTET_STREAM);
    assertThat(request.getContentLength()).isEqualTo(0);
    assertThat(toByteArray(request.getPayload())).isEqualTo(new byte[0]);
  }

  @Test
  public void multipleMessagesInConnection() throws Exception {
    InputStream stream = stream("GET /foo HTTP/1.0", "", "", "GET /bar HTTP/1.0", "", "");
    assertThat(readHttpRequest(stream).getUri()).isEqualTo(URI.create("/foo"));
    assertThat(readHttpRequest(stream).getUri()).isEqualTo(URI.create("/bar"));
  }

  @Test
  public void postWithPayload() throws Exception {
    assertThat(
            toByteArray(readHttpRequest(stream("POST /foo HTTP/1.0", "", "lol")).getPayload()))
        .isEqualTo("lol".getBytes(UTF_8));
  }

  @Test
  public void postWithContentLengthAndPayload() throws Exception {
    assertThat(
            toByteArray(readHttpRequest(stream(
                "POST /foo HTTP/1.0",
                "Content-Length: 3",
                "",
                "lol")).getPayload()))
        .isEqualTo("lol".getBytes(UTF_8));
  }

  @Test
  public void getWithPayload_ignored() throws Exception {
    HttpRequest request =
        readHttpRequest(stream(
            "GET /foo HTTP/1.1",
            "Content-Length: 3",
            "",
            "lol"));
    assertThat(request.getHeader("Connection")).isEqualTo("close");
    assertThat(request.getContentLength()).isEqualTo(-1);
    assertThat(toByteArray(request.getPayload())).isEmpty();
  }

  @Test
  public void hugeMessage_failsBeforeOoming() throws Exception {
    thrown.expect(HttpParserError.class);
    readHttpRequest(stream(
        "GET /foo HTTP/1.0",
        Strings.repeat("lol", 1024 * 1024) + ": hi",
        "",
        ""));
  }

  @Test
  public void lineFolding_notAllowed() throws Exception {
    thrown.expect(HttpParserError.class);
    readHttpRequest(stream(
        "GET /foo HTTP/1.0",
        "Cache-Control: foo",
        "  bar",
        "Expires: 0",
        "",
        ""));
  }

  @Test
  public void malformedMessage_throwsError() throws Exception {
    thrown.expect(HttpParserError.class);
    readHttpRequest(stream("GET /foo", "", ""));
  }

  @Test
  public void unexpectedEndOfStream_throwsError() throws Exception {
    thrown.expect(HttpParserError.class);
    readHttpRequest(stream("GET /foo HTTP/1.0", "hi: there"));
  }

  @Test
  public void noCarriageReturn_isAllowed() throws Exception {
    assertThat(
            toByteArray(
                    readHttpRequest(stream("POST /foo HTTP/1.0\nContent-Length: 3\n\nlol"))
                .getPayload()))
        .isEqualTo("lol".getBytes(UTF_8));
  }

  private static InputStream stream(String... lines) {
    return new ByteArrayInputStream(Joiner.on("\r\n").join(lines).getBytes(UTF_8));
  }
}
