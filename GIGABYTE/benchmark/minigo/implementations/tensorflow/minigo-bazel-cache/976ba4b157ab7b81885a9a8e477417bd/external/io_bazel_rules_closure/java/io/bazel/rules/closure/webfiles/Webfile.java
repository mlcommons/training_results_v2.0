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

import com.google.auto.value.AutoValue;
import io.bazel.rules.closure.Webpath;
import io.bazel.rules.closure.webfiles.BuildInfo.WebfileInfo;
import java.nio.file.Path;

/** Wrapper around {@link WebfileInfo}. */
@AutoValue
public abstract class Webfile {

  /** Creates new instance. */
  public static Webfile create(Webpath webpath, Path zip, String label, WebfileInfo info) {
    return new AutoValue_Webfile(webpath, zip, label, info);
  }

  /** Returns wrapped equivalent of {@code info().getWebpath()}. */
  public abstract Webpath webpath();

  /** Returns path incremental zip file from which this web file can be loaded. */
  public abstract Path zip();

  /** Returns label of build rule that defined this web file. */
  public abstract String label();

  /** Returns wrapped protobuf. */
  public abstract WebfileInfo info();
}
