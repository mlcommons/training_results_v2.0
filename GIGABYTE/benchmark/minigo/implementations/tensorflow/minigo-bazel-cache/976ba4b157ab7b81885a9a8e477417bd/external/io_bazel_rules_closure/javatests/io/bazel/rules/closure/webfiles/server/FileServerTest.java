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

import com.google.common.collect.ImmutableSortedMap;
import com.google.common.io.ByteStreams;
import com.google.common.jimfs.Configuration;
import com.google.common.jimfs.Jimfs;
import com.google.common.net.MediaType;
import io.bazel.rules.closure.Webpath;
import io.bazel.rules.closure.http.HttpResponse;
import java.io.IOException;
import java.nio.file.FileSystem;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.After;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link FileServer}. */
@RunWith(JUnit4.class)
public class FileServerTest {

  private final FileSystem fs = Jimfs.newFileSystem(Configuration.unix());
  private final HttpResponse response = new HttpResponse();

  @Rule
  public final ExpectedException thrown = ExpectedException.none();

  @After
  public void after() throws Exception {
    fs.close();
  }

  @Test
  public void noAssets_alwaysReturnsFalse() throws Exception {
    assertThat(
            new FileServer(response, ImmutableSortedMap.<Webpath, Path>of())
                .serve(Webpath.get("/foo.txt")))
        .isFalse();
  }

  @Test
  public void fileFound_servesFileWithContentTypeAndLength() throws Exception {
    String data = "fffffffuuuuuuuuuuuu";
    save(fs.getPath("/foo.txt"), data);
    assertThat(
            new FileServer(
                    response,
                    ImmutableSortedMap.of(Webpath.get("/foo.txt"), fs.getPath("/foo.txt")))
                .serve(Webpath.get("/foo.txt")))
        .isTrue();
    assertThat(response.getContentLength()).isEqualTo(19);
    assertThat(response.getContentType()).isEqualTo(MediaType.PLAIN_TEXT_UTF_8);
    assertThat(ByteStreams.toByteArray(response.getPayload())).isEqualTo(data.getBytes(UTF_8));
  }

  @Test
  public void unrecognizedExtension_usesOctetStreamContentType() throws Exception {
    save(fs.getPath("/foo.baz"), "haha");
    assertThat(
            new FileServer(
                    response,
                    ImmutableSortedMap.of(Webpath.get("/foo.baz"), fs.getPath("/foo.baz")))
                .serve(Webpath.get("/foo.baz")))
        .isTrue();
    assertThat(response.getContentType()).isEqualTo(MediaType.OCTET_STREAM);
  }

  @Test
  public void prefixPathExists_servesFile() throws Exception {
    save(fs.getPath("/foo.txt"), "fffffffuuuuuuuuuuuu");
    assertThat(
            new FileServer(response, ImmutableSortedMap.of(Webpath.get("/"), fs.getPath("/")))
                .serve(Webpath.get("/foo.txt")))
        .isTrue();
    assertThat(response.getContentLength()).isEqualTo(19);
  }

  @Test
  public void prefixPathDoesNotExists_returnsFalse() throws Exception {
    assertThat(
            new FileServer(response, ImmutableSortedMap.of(Webpath.get("/"), fs.getPath("/")))
                .serve(Webpath.get("/bar.txt")))
        .isFalse();
  }

  @Test
  public void prefixPathWithSubpath_stripsPrefix() throws Exception {
    save(fs.getPath("/runfiles/a/foo.txt"), "fffffffuuuuuuuuuuuu");
    assertThat(
            new FileServer(
                    response, ImmutableSortedMap.of(Webpath.get("/_"), fs.getPath("/runfiles")))
                .serve(Webpath.get("/_/a/foo.txt")))
        .isTrue();
    assertThat(response.getContentLength()).isEqualTo(19);
  }

  @Test
  public void multipleMatchingPrefixes_choosesLongerOne() throws Exception {
    save(fs.getPath("/goodfiles/foo.txt"), "fffffffuuuuuuuuuuuu");
    save(fs.getPath("/badfiles/b/foo.txt"), "ohno");
    assertThat(
            new FileServer(
                    response,
                    ImmutableSortedMap.of(
                        Webpath.get("/a/b"),
                        fs.getPath("/goodfiles"),
                        Webpath.get("/a"),
                        fs.getPath("/badfiles")))
                .serve(Webpath.get("/a/b/foo.txt")))
        .isTrue();
    assertThat(response.getContentLength()).isEqualTo(19);
  }

  @Test
  public void dottedPath_throwsIae() throws Exception {
    thrown.expect(IllegalArgumentException.class);
    new FileServer(response, ImmutableSortedMap.of(Webpath.get("/"), fs.getPath("/")))
        .serve(Webpath.get("/lol/../foo.txt"));
  }

  private static void save(Path path, String contents) throws IOException {
    Files.createDirectories(path.getParent());
    Files.write(path, contents.getBytes(UTF_8));
  }
}
