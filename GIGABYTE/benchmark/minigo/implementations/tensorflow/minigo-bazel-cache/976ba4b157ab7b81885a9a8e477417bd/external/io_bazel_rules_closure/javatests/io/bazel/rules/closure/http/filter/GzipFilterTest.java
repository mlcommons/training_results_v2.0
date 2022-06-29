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

package io.bazel.rules.closure.http.filter;

import static com.google.common.io.ByteStreams.toByteArray;
import static com.google.common.net.MediaType.OCTET_STREAM;
import static com.google.common.net.MediaType.PLAIN_TEXT_UTF_8;
import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.io.ByteStreams;
import io.bazel.rules.closure.http.HttpRequest;
import io.bazel.rules.closure.http.HttpResponse;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.zip.GZIPOutputStream;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link GzipFilter}. */
@RunWith(JUnit4.class)
public class GzipFilterTest {

  private static final byte[] hello = "hello".getBytes(UTF_8);

  @Test
  public void acceptsGzipForTextFile_encodesPayloadAndSetsHeader() throws Exception {
    HttpRequest request = new HttpRequest().setHeader("Accept-Encoding", "deflate, gzip");
    HttpResponse response = new HttpResponse().setContentType(PLAIN_TEXT_UTF_8).setPayload(hello);
    new GzipFilter(request, response).apply();
    assertThat(response.getHeader("Content-Encoding")).isEqualTo("gzip");
    assertThat(toByteArray(response.getPayload())).isEqualTo(gzipData(hello));
  }

  @Test
  public void binaryResponse_doesNotApplyGzipEvenIfRequested() throws Exception {
    HttpRequest request = new HttpRequest().setHeader("Accept-Encoding", "gzip");
    HttpResponse response = new HttpResponse().setContentType(OCTET_STREAM).setPayload(hello);
    new GzipFilter(request, response).apply();
    assertThat(response.getHeader("Content-Encoding")).isEmpty();
    assertThat(toByteArray(response.getPayload())).isEqualTo(hello);
  }

  @Test
  public void doesNotRequestContentEncoding_doesNotMutateResponse() throws Exception {
    HttpRequest request = new HttpRequest();
    HttpResponse response = new HttpResponse().setContentType(PLAIN_TEXT_UTF_8).setPayload(hello);
    new GzipFilter(request, response).apply();
    assertThat(response.getHeader("Content-Encoding")).isEmpty();
    assertThat(toByteArray(response.getPayload())).isEqualTo(hello);
  }

  private static byte[] gzipData(byte[] bytes) throws IOException {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    try (InputStream input = new ByteArrayInputStream(bytes);
        OutputStream output = new GZIPOutputStream(baos)) {
      ByteStreams.copy(input, output);
    }
    return baos.toByteArray();
  }
}
