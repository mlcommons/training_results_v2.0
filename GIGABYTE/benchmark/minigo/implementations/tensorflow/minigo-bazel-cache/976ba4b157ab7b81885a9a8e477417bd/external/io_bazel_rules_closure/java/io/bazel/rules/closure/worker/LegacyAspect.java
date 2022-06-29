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

package io.bazel.rules.closure.worker;

import io.bazel.rules.closure.worker.Annotations.Action;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.inject.Inject;

/** Adapter of old {@link CommandLineProgram} to {@link Program} API. */
public final class LegacyAspect<T extends CommandLineProgram> implements Program {

  private final CommandLineProgram delegate;
  private final Iterable<String> arguments;
  private final AtomicBoolean failed;

  @Inject
  public LegacyAspect(T delegate, @Action List<String> arguments, @Action AtomicBoolean failed) {
    this.delegate = delegate;
    this.arguments = arguments;
    this.failed = failed;
  }

  @Override
  public void run() throws Exception {
    if (delegate.apply(arguments) != 0) {
      failed.set(true);
    }
  }
}
