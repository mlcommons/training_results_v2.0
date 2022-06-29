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

import java.nio.file.FileSystem;
import java.nio.file.Path;
import javax.inject.Inject;

// TODO(jart): Find way to make this work with blaze-run.sh
final class Runfiles {

  private static final String TEST_SRCDIR = "..";

  private final FileSystem fs;

  @Inject
  Runfiles(FileSystem fs) {
    this.fs = fs;
  }

  Path getPath(String longPath) {
    return fs.getPath(TEST_SRCDIR).normalize().resolve(longPath);
  }
}
