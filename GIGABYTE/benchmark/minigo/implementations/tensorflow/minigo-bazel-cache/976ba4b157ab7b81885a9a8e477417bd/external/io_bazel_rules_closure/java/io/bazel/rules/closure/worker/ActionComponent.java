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

import com.google.common.hash.HashCode;
import com.google.common.io.Closer;
import dagger.BindsInstance;
import io.bazel.rules.closure.worker.Annotations.Action;
import java.io.PrintStream;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

/** Dagger component definition for Bazel ctx.action invocations. */
public interface ActionComponent<P extends Program> {

  /** Returns instance of {@link Program} which handles command invocations. */
  P program();

  /** Builder for {@link ActionComponent}. */
  public interface Builder<
      P extends Program,
      I extends ActionComponent<P>,
      B extends ActionComponent.Builder<P, I, B>> {

    /** Binds args to action component. */
    @BindsInstance
    B args(@Action List<String> args);

    /** Binds inputs to action component. */
    @BindsInstance
    B inputDigests(@Action Map<Path, HashCode> inputDigests);

    /** Binds return code setter. */
    @BindsInstance
    B failed(@Action AtomicBoolean failed);

    /** Binds standard error. */
    @BindsInstance
    B output(@Action PrintStream output);

    /** Binds automatic resource closer. */
    @BindsInstance
    B closer(@Action Closer closer);

    /** Creates instance of component. */
    I build();
  }
}
