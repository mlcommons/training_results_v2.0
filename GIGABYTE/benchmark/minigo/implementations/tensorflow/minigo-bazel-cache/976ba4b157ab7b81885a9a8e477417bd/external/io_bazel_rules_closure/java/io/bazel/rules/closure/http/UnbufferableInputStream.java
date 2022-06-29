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

import static com.google.common.base.Preconditions.checkState;
import static com.google.common.base.Verify.verify;

import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.FilterInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.SequenceInputStream;

/** Buffered input stream that allows buffering to be turned off. */
public final class UnbufferableInputStream extends FilterInputStream {

  UnbufferableInputStream(InputStream delegate, int size) {
    super(new Buffer(delegate, size));
  }

  /** Turns off buffering, which can only be called once. */
  public void disableBuffering() throws IOException {
    checkState(in instanceof Buffer);
    Buffer buffer = (Buffer) in;
    InputStream original = buffer.getInput();
    int count = buffer.getNumberOfBytesInBuffer();
    if (count == 0) {
      in = original;
    } else {
      byte[] bridge = new byte[count];
      verify(in.read(bridge) == count);
      in = new SequenceInputStream(new ByteArrayInputStream(bridge), original);
    }
  }

  private static final class Buffer extends BufferedInputStream {
    Buffer(InputStream delegate, int size) {
      super(delegate, size);
    }

    InputStream getInput() throws IOException {
      InputStream input = in;
      if (input == null) {
        throw new IOException("Stream closed");
      }
      return input;
    }

    int getNumberOfBytesInBuffer() {
      return count - pos;
    }
  }
}
