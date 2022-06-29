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

package io.bazel.rules.closure.webfiles;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.nio.file.StandardOpenOption.CREATE;
import static java.nio.file.StandardOpenOption.TRUNCATE_EXISTING;
import static java.nio.file.StandardOpenOption.WRITE;

import com.google.common.base.Strings;
import com.google.common.collect.Range;
import com.google.common.jimfs.Configuration;
import com.google.common.jimfs.Jimfs;
import com.google.common.testing.NullPointerTester;
import io.bazel.rules.closure.webfiles.BuildInfo.MultimapInfo;
import io.bazel.rules.closure.webfiles.BuildInfo.WebfileInfo;
import io.bazel.rules.closure.webfiles.BuildInfo.WebfileManifestInfo;
import java.nio.channels.SeekableByteChannel;
import java.nio.file.FileSystem;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.zip.Deflater;
import org.junit.Rule;
import org.junit.Test;
import org.junit.experimental.theories.DataPoint;
import org.junit.experimental.theories.Theories;
import org.junit.experimental.theories.Theory;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;

/** Unit tests for {@link WebfilesUtils}. */
@RunWith(Theories.class)
public class WebfilesUtilsTest {

  @Rule public final ExpectedException thrown = ExpectedException.none();
  @DataPoint public static FileSystem nixFileSystem = Jimfs.newFileSystem(Configuration.unix());
  @DataPoint public static FileSystem macFileSystem = Jimfs.newFileSystem(Configuration.osX());
  @DataPoint public static FileSystem winFileSystem = Jimfs.newFileSystem(Configuration.windows());

  @Test
  public void nulls() throws Exception {
    NullPointerTester npt = new NullPointerTester();
    npt.setDefault(Path.class, nixFileSystem.getPath("foo.pb"));
    npt.setDefault(WebfileManifestInfo.class, WebfileManifestInfo.getDefaultInstance());
    npt.testAllPublicStaticMethods(WebfilesUtils.class);
  }

  @Theory
  public void writeManifest_badExtension_throwsError(FileSystem fs) throws Exception {
    thrown.expect(IllegalArgumentException.class);
    WebfilesUtils.writeManifest(fs.getPath("man.omg"), WebfileManifestInfo.getDefaultInstance());
  }

  @Theory
  public void readManifest_badExtension_throwsError(FileSystem fs) throws Exception {
    thrown.expect(IllegalArgumentException.class);
    WebfilesUtils.readManifest(fs.getPath("man.omg"));
  }

  @Theory
  public void readManifest_roundTrip(FileSystem fs) throws Exception {
    Path path = fs.getPath("manifest.pb");
    WebfileManifestInfo manifest =
        WebfileManifestInfo.newBuilder()
            .addWebfile(WebfileInfo.newBuilder().setWebpath("/hello.js").build())
            .addWebfile(WebfileInfo.newBuilder().setWebpath("/hello.html").build())
            .addLink(MultimapInfo.newBuilder().setKey("/hello.html").addValue("/hello.js").build())
            .build();
    WebfilesUtils.writeManifest(path, manifest);
    assertThat(WebfilesUtils.readManifest(path)).isEqualTo(manifest);
  }

  @Test
  public void getWebfileNameInZip_makesNonAbsolute() throws Exception {
    assertThat(
            WebfilesUtils.getZipEntryName(WebfileInfo.newBuilder().setWebpath("/a/b.txt").build()))
        .isEqualTo("a/b.txt");
  }

  @Theory
  public void appendToBlob_htmlFile_usesCompression(FileSystem fs) throws Exception {
    Path izip = fs.getPath("blob.i.zip");
    try (SeekableByteChannel chan = Files.newByteChannel(izip, WRITE, CREATE, TRUNCATE_EXISTING);
        WebfilesWriter writer = new WebfilesWriter(chan, Deflater.BEST_SPEED)) {
      writer.writeWebfile(
          WebfileInfo.newBuilder().setWebpath("/foo.html").build(),
          Strings.repeat("LOL", 10000).getBytes(UTF_8));
    }
    assertThat(Files.size(izip)).isIn(Range.open(0L, 30000L));
  }

  @Theory
  public void appendToBlob_jpgFile_usesNoCompression(FileSystem fs) throws Exception {
    Path izip = fs.getPath("files.i.zip");
    WebfileInfo webfile = WebfileInfo.newBuilder().setWebpath("/foo.jpg").build();
    try (SeekableByteChannel chan = Files.newByteChannel(izip, WRITE, CREATE, TRUNCATE_EXISTING);
        WebfilesWriter writer = new WebfilesWriter(chan, Deflater.BEST_SPEED)) {
      webfile = writer.writeWebfile(webfile, Strings.repeat("lol", 100).getBytes(UTF_8));
    }
    assertThat(Files.size(izip)).isGreaterThan(300L);
  }
}
