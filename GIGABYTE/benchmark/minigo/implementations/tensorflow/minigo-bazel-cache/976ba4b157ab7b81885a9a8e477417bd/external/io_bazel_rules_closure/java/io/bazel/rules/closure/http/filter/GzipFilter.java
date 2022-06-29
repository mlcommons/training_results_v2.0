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

import com.google.common.collect.ImmutableSet;
import com.google.common.io.ByteStreams;
import com.google.common.net.MediaType;
import io.bazel.rules.closure.http.HttpRequest;
import io.bazel.rules.closure.http.HttpResponse;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.regex.Pattern;
import java.util.zip.GZIPOutputStream;
import javax.inject.Inject;

/** Filter that applies compression to responses when appropriate. */
public final class GzipFilter {

  private static final Pattern ALLOWS_GZIP =
      Pattern.compile("(?:^|,|\\s)(?:(?:x-)?gzip|\\*)(?!;q=0)(?:\\s|,|$)");

  private static final ImmutableSet<MediaType> COMPRESSIBLE =
      ImmutableSet.of(
          MediaType.ATOM_UTF_8.withoutParameters(),
          MediaType.CSS_UTF_8.withoutParameters(),
          MediaType.CSV_UTF_8.withoutParameters(),
          MediaType.DART_UTF_8.withoutParameters(),
          MediaType.EOT,
          MediaType.HTML_UTF_8.withoutParameters(),
          MediaType.JAVASCRIPT_UTF_8.withoutParameters(),
          MediaType.JSON_UTF_8.withoutParameters(),
          MediaType.KML,
          MediaType.KMZ,
          MediaType.MANIFEST_JSON_UTF_8.withoutParameters(),
          MediaType.PLAIN_TEXT_UTF_8.withoutParameters(),
          MediaType.POSTSCRIPT,
          MediaType.RDF_XML_UTF_8.withoutParameters(),
          MediaType.RTF_UTF_8.withoutParameters(),
          MediaType.SFNT,
          MediaType.SVG_UTF_8.withoutParameters(),
          MediaType.TAR,
          MediaType.TSV_UTF_8.withoutParameters(),
          MediaType.VCARD_UTF_8.withoutParameters(),
          MediaType.XHTML_UTF_8.withoutParameters(),
          MediaType.XML_UTF_8.withoutParameters());

  private final HttpRequest request;
  private final HttpResponse response;

  @Inject
  public GzipFilter(HttpRequest request, HttpResponse response) {
    this.request = request;
    this.response = response;
  }

  public void apply() throws IOException {
    if (!response.getHeader("Content-Encoding").isEmpty()
        || !COMPRESSIBLE.contains(response.getContentType().withoutParameters())
        || !ALLOWS_GZIP.matcher(request.getHeader("Accept-Encoding")).find()) {
      return;
    }
    try (ByteArrayOutputStream buffer = new ByteArrayOutputStream()) {
      try (InputStream input = response.getPayload();
          OutputStream output = new GZIPOutputStream(buffer, 8192)) {
        ByteStreams.copy(input, output);
      }
      response.setHeader("Content-Encoding", "gzip");
      response.setContentLength(buffer.size());
      response.setPayload(new ByteArrayInputStream(buffer.toByteArray()));
    }
  }
}
