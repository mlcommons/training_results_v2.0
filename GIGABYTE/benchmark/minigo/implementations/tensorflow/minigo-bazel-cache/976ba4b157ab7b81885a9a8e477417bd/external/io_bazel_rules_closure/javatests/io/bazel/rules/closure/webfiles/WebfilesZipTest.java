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
import static java.nio.file.StandardOpenOption.READ;
import static java.nio.file.StandardOpenOption.TRUNCATE_EXISTING;
import static java.nio.file.StandardOpenOption.WRITE;

import com.google.common.base.Strings;
import com.google.common.collect.Range;
import com.google.common.io.ByteStreams;
import com.google.common.jimfs.Configuration;
import com.google.common.jimfs.Jimfs;
import com.google.common.testing.NullPointerTester;
import io.bazel.rules.closure.webfiles.BuildInfo.WebfileInfo;
import io.bazel.rules.closure.webfiles.WebfilesReader.ZipEntryInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.channels.Channels;
import java.nio.channels.SeekableByteChannel;
import java.nio.file.FileSystem;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.FileTime;
import java.util.zip.Deflater;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import org.junit.Rule;
import org.junit.experimental.theories.DataPoint;
import org.junit.experimental.theories.Theories;
import org.junit.experimental.theories.Theory;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;

/** Unit tests for {@link WebfilesReader} and {@link WebfilesWriter}. */
@RunWith(Theories.class)
public class WebfilesZipTest {

  @Rule public final ExpectedException thrown = ExpectedException.none();
  @DataPoint public static FileSystem nixFileSystem = Jimfs.newFileSystem(Configuration.unix());
  @DataPoint public static FileSystem macFileSystem = Jimfs.newFileSystem(Configuration.osX());
  @DataPoint public static FileSystem winFileSystem = Jimfs.newFileSystem(Configuration.windows());

  @Theory
  public void nulls(FileSystem fs) throws Exception {
    Path path = fs.getPath("rule.i.zip");
    try (SeekableByteChannel chan = Files.newByteChannel(path, WRITE, CREATE, TRUNCATE_EXISTING)) {
      NullPointerTester npt = new NullPointerTester();
      npt.setDefault(Path.class, path);
      npt.setDefault(FileTime.class, FileTime.fromMillis(0));
      npt.setDefault(WebfileInfo.class, WebfileInfo.getDefaultInstance());
      npt.testAllPublicStaticMethods(WebfilesReader.class);
      npt.testAllPublicStaticMethods(WebfilesWriter.class);
      npt.testAllPublicConstructors(WebfilesReader.class);
      npt.testAllPublicConstructors(WebfilesWriter.class);
      npt.testAllPublicInstanceMethods(new WebfilesReader(chan));
      npt.testAllPublicInstanceMethods(new WebfilesWriter(chan, Deflater.BEST_SPEED));
    }
  }

  @Theory
  public void writer_htmlFile_usesCompression(FileSystem fs) throws Exception {
    Path path = fs.getPath("rule.i.zip");
    try (WebfilesWriter writer =
        new WebfilesWriter(
            Files.newByteChannel(path, WRITE, CREATE, TRUNCATE_EXISTING), Deflater.BEST_SPEED)) {
      writer.writeWebfile(
          WebfileInfo.newBuilder().setWebpath("/foo.html").build(),
          Strings.repeat("LOL", 10000).getBytes(UTF_8));
    }
    assertThat(Files.size(path)).isIn(Range.open(0L, 30000L));
  }

  @Theory
  public void writer_jpgFile_doesNotUseCompression(FileSystem fs) throws Exception {
    Path path = fs.getPath("rule.i.zip");
    WebfileInfo webfile = WebfileInfo.newBuilder().setWebpath("/foo.jpg").build();
    try (WebfilesWriter writer =
        new WebfilesWriter(
            Files.newByteChannel(path, WRITE, CREATE, TRUNCATE_EXISTING), Deflater.BEST_SPEED)) {
      webfile = writer.writeWebfile(webfile, Strings.repeat("lol", 100).getBytes(UTF_8));
    }
    assertThat(Files.size(path)).isGreaterThan(300L);
  }

  @Theory
  public void twoFiles_readOutOfOrder_works(FileSystem fs) throws Exception {
    Path path = fs.getPath("rule.i.zip");
    WebfileInfo rawFile = WebfileInfo.newBuilder().setWebpath("/foo.js").build();
    byte[] rawData = "Fanatics have their dreams, wherewith they weave".getBytes(UTF_8);
    WebfileInfo objectFile = WebfileInfo.newBuilder().setWebpath("/foo.js.o").build();
    byte[] objectData = "A paradise for a sect; the savage too".getBytes(UTF_8);
    try (WebfilesWriter writer =
        new WebfilesWriter(
            Files.newByteChannel(path, WRITE, CREATE, TRUNCATE_EXISTING),
            Deflater.BEST_COMPRESSION)) {
      rawFile = writer.writeWebfile(rawFile, rawData);
      objectFile = writer.writeWebfile(objectFile, objectData);
    }
    try (SeekableByteChannel chan = Files.newByteChannel(path);
        WebfilesReader zip = new WebfilesReader(chan)) {
      try (InputStream input = zip.openWebfile(objectFile)) {
        assertThat(ByteStreams.toByteArray(input)).isEqualTo(objectData);
      }
      try (InputStream input = zip.openWebfile(rawFile)) {
        assertThat(ByteStreams.toByteArray(input)).isEqualTo(rawData);
      }
    }
  }

  @Theory
  public void reader_corruptFile_throwsCrcError(FileSystem fs) throws Exception {
    Path path = fs.getPath("rule.i.zip");
    WebfileInfo webfile = WebfileInfo.newBuilder().setWebpath("/foo.jpg").build();
    try (WebfilesWriter writer =
        new WebfilesWriter(
            Files.newByteChannel(path, WRITE, CREATE, TRUNCATE_EXISTING), Deflater.BEST_SPEED)) {
      webfile =
          writer.writeWebfile(
              webfile, Strings.repeat("hello <b>world</b><br>", 1000).getBytes(UTF_8));
    }
    try (SeekableByteChannel chan = Files.newByteChannel(path, WRITE)) {
      chan.position(100);
      chan.write(ByteBuffer.wrap(new byte[] {6, 6, 6}));
    }
    try (SeekableByteChannel chan = Files.newByteChannel(path);
        WebfilesReader zip = new WebfilesReader(chan);
        ZipEntryInputStream input = zip.openWebfile(webfile)) {
      thrown.expect(IOException.class);
      thrown.expectMessage("CRC");
      ByteStreams.exhaust(input);
    }
  }

  @Theory
  public void fileHasDirectories_createsDirectoriesWithoutDupes(FileSystem fs) throws Exception {
    Path path = fs.getPath("rule.i.zip");
    WebfileInfo webfile1 = WebfileInfo.newBuilder().setWebpath("/a/b/c.txt").build();
    WebfileInfo webfile2 = WebfileInfo.newBuilder().setWebpath("/a/b/d.txt").build();
    try (WebfilesWriter writer =
        new WebfilesWriter(
            Files.newByteChannel(path, WRITE, CREATE, TRUNCATE_EXISTING), Deflater.BEST_SPEED)) {
      webfile1 = writer.writeWebfile(webfile1, "c".getBytes(UTF_8));
      webfile2 = writer.writeWebfile(webfile2, "d".getBytes(UTF_8));
    }
    try (SeekableByteChannel chan = Files.newByteChannel(path, READ);
        ZipInputStream zip = new ZipInputStream(Channels.newInputStream(chan))) {
      ZipEntry a = zip.getNextEntry();
      assertThat(a.getName()).isEqualTo("a/");
      assertThat(a.isDirectory()).isTrue();
      ZipEntry b = zip.getNextEntry();
      assertThat(b.getName()).isEqualTo("a/b/");
      assertThat(b.isDirectory()).isTrue();
      ZipEntry c = zip.getNextEntry();
      assertThat(c.getName()).isEqualTo("a/b/c.txt");
      assertThat(c.isDirectory()).isFalse();
      assertThat(zip.read()).isEqualTo('c');
      assertThat(zip.read()).isEqualTo(-1);
      assertThat(c.getSize()).isEqualTo(1);
      ZipEntry d = zip.getNextEntry();
      assertThat(d.getName()).isEqualTo("a/b/d.txt");
      assertThat(d.isDirectory()).isFalse();
      assertThat(zip.read()).isEqualTo('d');
      assertThat(zip.read()).isEqualTo(-1);
      assertThat(d.getSize()).isEqualTo(1);
    }
  }

  @Theory
  public void writeInputStreamThatIsntBytes_worksCorrectly(FileSystem fs) throws Exception {
    Path path = fs.getPath("rule.i.zip");
    WebfileInfo webfile = WebfileInfo.newBuilder().setWebpath("/foo.jpg").build();
    try (SeekableByteChannel chan = Files.newByteChannel(path, WRITE, CREATE, TRUNCATE_EXISTING);
        WebfilesWriter writer = new WebfilesWriter(chan, Deflater.BEST_SPEED)) {
      webfile = writer.writeWebfile(webfile, Strings.repeat("lol", 100).getBytes(UTF_8));
    }
    assertThat(Files.size(path)).isGreaterThan(300L);
  }
}
