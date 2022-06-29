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

import static com.google.common.truth.Truth.assertThat;

import com.google.common.net.MediaType;
import com.google.common.testing.ClassSanityTester;
import java.nio.charset.StandardCharsets;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link HttpResponse}. */
@RunWith(JUnit4.class)
public class HttpResponseTest {

  @Rule public final ExpectedException thrown = ExpectedException.none();

  @Test
  public void setStatus_withoutMessage_knowsMessageForStandardCode() {
    HttpResponse response = new HttpResponse();
    response.setStatus(404);
    assertThat(response.getStatus()).isEqualTo(404);
    assertThat(response.getMessage()).isEqualTo("Not Found");
  }

  @Test
  public void setStatus_badMessage_throwsIae() {
    thrown.expect(IllegalArgumentException.class);
    new HttpResponse().setStatus(404, "\r\nX-Evil-Header: oh my goth");
  }

  @Test
  public void setHeader_contentLength_setsField() {
    HttpResponse response = new HttpResponse();
    response.setHeader("CONTENT-LENGTH", "123");
    assertThat(response.getContentLength()).isEqualTo(123);
    assertThat(response.getHeader("content-Length")).isEqualTo("123");
    assertThat(response.getHeaders()).containsEntry("content-length", "123");
  }

  @Test
  public void setContentLength_setsHeader() {
    HttpResponse response = new HttpResponse();
    response.setContentLength(456);
    assertThat(response.getContentLength()).isEqualTo(456);
    assertThat(response.getHeader("content-Length")).isEqualTo("456");
  }

  @Test
  public void nonAscii() {
    HttpResponse response = new HttpResponse();
    response.setContentType(MediaType.PLAIN_TEXT_UTF_8);
    response.setPayload("(◕‿◕)".getBytes(StandardCharsets.UTF_8));
    response.setHeader("x-lol", "héllo");
    assertThat(response.toString()).startsWith("HTTP/1.1 200 OK\r\n");
    assertThat(response.toString()).contains("x-lol: héllo");
    assertThat(response.toString()).contains("text/plain");
    assertThat(response.toString()).contains("(◕‿◕)");
  }

  @Test
  public void nulls() throws Exception {
    new ClassSanityTester().testNulls(HttpResponse.class);
  }
}
