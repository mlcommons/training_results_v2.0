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

package io.bazel.rules.closure.webfiles.server;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableSet;
import com.google.common.io.CharStreams;
import com.google.common.jimfs.Configuration;
import com.google.common.jimfs.Jimfs;
import com.google.common.net.MediaType;
import io.bazel.rules.closure.Webpath;
import io.bazel.rules.closure.http.HttpResponse;
import io.bazel.rules.closure.webfiles.server.BuildInfo.WebfilesServerInfo;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.FileSystem;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link ListingPage}. */
@RunWith(JUnit4.class)
public class ListingPageTest {

  private final FileSystem fs = Jimfs.newFileSystem(Configuration.unix());
  private final Metadata.Config config = mock(Metadata.Config.class);
  private final HttpResponse response = new HttpResponse();

  @Before
  public void before() throws Exception {
    when(config.get())
        .thenReturn(WebfilesServerInfo.newBuilder().setLabel("//omg/im/a/label").build());
  }

  @After
  public void after() throws Exception {
    fs.close();
  }

  @Test
  public void noWebpaths_stillShowsPage() throws Exception {
    new ListingPage(response, config, ImmutableSet.<Webpath>of()).serve(Webpath.get("/"));
    assertThat(response.getContentType()).isEqualTo(MediaType.HTML_UTF_8);
    assertThat(getResponsePayload()).contains("No srcs found");
  }

  @Test
  public void homePage_showsAllWebPaths() throws Exception {
    new ListingPage(
            response,
            config,
            ImmutableSet.of(
                Webpath.get("/omg/im/a/webpath"), Webpath.get("/omg/im/another/webpath")))
        .serve(Webpath.get("/"));
    String html = getResponsePayload();
    assertThat(html).contains("//omg/im/a/label");
    assertThat(html).contains("/omg/im/a/webpath");
    assertThat(html).contains("/omg/im/another/webpath");
  }

  @Test
  public void requestUriSubPath_filtersMatchingPrefixes() throws Exception {
    new ListingPage(
            response,
            config,
            ImmutableSet.of(
                Webpath.get("/omg/im/a/webpath"), Webpath.get("/omg/im/another/webpath")))
        .serve(Webpath.get("/omg/im/another"));
    String html = getResponsePayload();
    assertThat(html).doesNotContain("/omg/im/a/webpath");
    assertThat(html).contains("/omg/im/another/webpath");
  }

  private String getResponsePayload() throws IOException {
    return CharStreams.toString(new InputStreamReader(response.getPayload(), UTF_8));
  }
}
