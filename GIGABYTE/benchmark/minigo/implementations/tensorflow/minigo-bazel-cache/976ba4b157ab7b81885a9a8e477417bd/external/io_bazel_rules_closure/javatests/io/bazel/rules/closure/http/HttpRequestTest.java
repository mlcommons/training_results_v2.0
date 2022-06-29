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
import static java.nio.charset.StandardCharsets.US_ASCII;

import com.google.common.io.ByteStreams;
import com.google.common.testing.ClassSanityTester;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link HttpRequest}. */
@RunWith(JUnit4.class)
public class HttpRequestTest {

  @Rule public final ExpectedException thrown = ExpectedException.none();

  @Test
  public void setPayload_byteArray_setsContentLength() throws Exception {
    HttpRequest request = new HttpRequest().setMethod("POST");
    byte[] payload = "hello".getBytes(StandardCharsets.US_ASCII);
    request.setPayload(payload);
    assertThat(request.getContentLength()).isEqualTo(5);
    assertThat(ByteStreams.toByteArray(request.getPayload())).isEqualTo(payload);
  }

  @Test
  public void setPayload_partialContent_limitsStream() throws Exception {
    HttpRequest request = new HttpRequest().setMethod("POST");
    InputStream payload = new ByteArrayInputStream("hello".getBytes(US_ASCII));
    request.setContentLength(3);
    request.setPayload(payload);
    assertThat(ByteStreams.toByteArray(request.getPayload())).isEqualTo("hel".getBytes(US_ASCII));
    assertThat(ByteStreams.toByteArray(payload)).isEqualTo("lo".getBytes(US_ASCII));
  }

  @Test
  public void setPayload_whenMethodDoesntSupportIt_getsIgnored() throws Exception {
    HttpRequest request = new HttpRequest().setMethod("GET");
    byte[] payload = "hello".getBytes(StandardCharsets.US_ASCII);
    request.setPayload(payload);
    assertThat(ByteStreams.toByteArray(request.getPayload())).isEmpty();
  }

  @Test
  public void setPayload_noLengthSpecified_closesConnection() throws Exception {
    HttpRequest request = new HttpRequest().setMethod("POST");
    InputStream payload = new ByteArrayInputStream("hello".getBytes(US_ASCII));
    request.setPayload(payload);
    assertThat(request.getHeader("Connection")).isEqualTo("close");
    assertThat(request.getContentLength()).isEqualTo(-1);
    assertThat(ByteStreams.toByteArray(request.getPayload())).isEqualTo("hello".getBytes(US_ASCII));
  }

  @Test
  public void setPayload_lengthSpecified_keepsConnectionAlive() throws Exception {
    HttpRequest request = new HttpRequest().setMethod("POST");
    request.setPayload("hello".getBytes(US_ASCII));
    assertThat(request.getHeader("Connection")).isEmpty();
  }

  @Test
  public void nulls() throws Exception {
    new ClassSanityTester().testNulls(HttpRequest.class);
  }
}
