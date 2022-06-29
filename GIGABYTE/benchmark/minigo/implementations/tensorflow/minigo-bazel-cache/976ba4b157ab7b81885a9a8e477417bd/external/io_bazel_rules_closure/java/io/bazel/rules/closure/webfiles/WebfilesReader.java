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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.base.Verify.verify;

import com.google.common.base.VerifyException;
import io.bazel.rules.closure.webfiles.BuildInfo.WebfileInfo;
import java.io.Closeable;
import java.io.FilterInputStream;
import java.io.IOException;
import java.nio.channels.Channels;
import java.nio.channels.SeekableByteChannel;
import java.util.zip.ZipEntry;
import java.util.zip.ZipError;
import java.util.zip.ZipInputStream;
import javax.annotation.CheckReturnValue;
import javax.annotation.Nullable;
import javax.annotation.WillCloseWhenClosed;
import javax.annotation.WillNotClose;

/**
 * Utility for reading zip files containing web files.
 *
 * <p>This helper class allows us to surgically extract individual files from a zip archive, in any
 * particular order, because the proto manifest data gives us the specific file offsets where
 * entries are stored.
 *
 * <p>This class is not intended for reading the full contents of a zip sequentially. For that use
 * case, the traditional Java tooling is more appropriate.
 */
public class WebfilesReader implements Closeable {

  private final SeekableByteChannel channel;
  @Nullable private ZipInputStream zip;
  private boolean isInUse;

  /**
   * Creates new helper for reading web files from a zip archive.
   *
   * @param channel Java 7 byte {@code channel} already opened with read permission
   */
  public WebfilesReader(@WillCloseWhenClosed SeekableByteChannel channel) {
    this.channel = checkNotNull(channel);
  }

  /**
   * Reads {@code webfile} from currently opened zip archive.
   *
   * <p>This method seeks to the appropriate position within the zip archive, verifies the name is
   * what we expect, and returns a stream for reading that particular file.
   *
   * <p><b>Warning:</b> Only one web file can be open at a time. The returned value must be closed
   * before this method can be called again. This is also not thread safe.
   *
   * @param webfile information about stored webfile
   * @return unbuffered stream of file within zip
   * @throws VerifyException if index provided by {@code webfile} pointed to an unrelated file
   * @throws IllegalArgumentException if {@code webfile} has an illegal name or isn't in the zip
   * @throws IllegalStateException if called when the previously returned value is not closed
   * @throws ZipError if zip data was corrupt
   * @throws IOException if i/o badness happened
   */
  @CheckReturnValue
  public ZipEntryInputStream openWebfile(WebfileInfo webfile) throws IOException {
    checkArgument(webfile.getInZip(), "Webfile says it's not stored in izip: %s", webfile);
    checkNotReadingEntry();
    String name = WebfilesUtils.getZipEntryName(webfile);
    // Because ZipInputStream has an internal buffer, it is not possible for us to hop around in the
    // byte channel without recreating this object. Thankfully, that is much less expensive than the
    // system overhead of closing and reopening the file. It also grants us the flexibility to
    // cherry-pick files in any particular order. Even though this is the case, we still enforce an
    // API contract that the returned value be fully read and closed before this method can be
    // called again. This ensures the CRC32 value is validated.
    if (channel.position() != webfile.getOffset()) {
      // Ideally we would use ZipFile#getEntry(String) which allows us to hop to the offset based on
      // the directory at the end of the zip file. Then we wouldn't need to store offsets in the
      // proto at all. However that native API does not work with a Java 7 NIO abstract FileSystem.
      channel.position(webfile.getOffset());
      zip = null;
    }
    if (zip == null) {
      zip = new ZipInputStream(Channels.newInputStream(channel));
    }
    // Please note that we do not call zip.close() for a reason.
    ZipEntry entry = zip.getNextEntry();
    String entryName = entry.getName();
    verify(entryName.equals(name), "Found %s in zip but expected %s", entryName, name);
    isInUse = true;
    return new ZipEntryInputStream(zip, entry);
  }

  @Override
  public void close() throws IOException {
    channel.close();
    checkNotReadingEntry();
  }

  void checkNotReadingEntry() {
    checkState(!isInUse, "Can't read another zip entry until previous one is closed");
  }

  /** Stream for reading a single file from a zip without closing the underlying file resource. */
  public final class ZipEntryInputStream extends FilterInputStream {
    private final ZipEntry entry;

    ZipEntryInputStream(@WillNotClose ZipInputStream zip, ZipEntry entry) {
      super(zip);
      this.entry = entry;
    }

    /** Returns metadata that was stored in the ZIP file associated with this file. */
    public ZipEntry getZipEntry() {
      return entry;
    }

    @Override
    public void close() throws IOException {
      isInUse = false;
      ((ZipInputStream) in).closeEntry();
    }
  }
}
