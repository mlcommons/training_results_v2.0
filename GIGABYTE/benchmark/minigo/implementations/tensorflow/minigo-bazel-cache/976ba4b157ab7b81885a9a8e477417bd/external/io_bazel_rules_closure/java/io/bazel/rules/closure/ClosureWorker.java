/*
 * Copyright 2016 The Closure Rules Authors. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.bazel.rules.closure;

import com.google.common.collect.ImmutableList;
import com.google.javascript.jscomp.JsChecker;
import com.google.javascript.jscomp.JsCompiler;
import dagger.BindsInstance;
import dagger.Component;
import dagger.Subcomponent;
import io.bazel.rules.closure.webfiles.WebfilesValidatorProgram;
import io.bazel.rules.closure.worker.ActionComponent;
import io.bazel.rules.closure.worker.ActionModule;
import io.bazel.rules.closure.worker.Annotations.Action;
import io.bazel.rules.closure.worker.LegacyAspect;
import io.bazel.rules.closure.worker.PersistentWorker;
import io.bazel.rules.closure.worker.Prefixes;
import io.bazel.rules.closure.worker.Program;
import io.bazel.rules.closure.worker.WorkerComponent;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.inject.Inject;
import javax.inject.Singleton;

/** Bazel worker for all Closure Tools programs, some of which are modded. */
public final class ClosureWorker implements Program {

  private final Invocation action;
  private final PrintStream output;
  private final AtomicBoolean failed;
  private final List<String> arguments;

  @Inject
  ClosureWorker(
      Invocation action,
      @Action PrintStream output,
      @Action AtomicBoolean failed,
      @Action List<String> arguments) {
    this.action = action;
    this.failed = failed;
    this.output = output;
    this.arguments = arguments;
  }

  @Override
  public void run() throws Exception {
    String head = arguments.remove(0);
    // TODO(jart): Include Closure Templates and Stylesheets.
    switch (head) {
      case "JsChecker":
        action.jsChecker().run();
        break;
      case "JsCompiler":
        action.jsCompiler().run();
        break;
      case "WebfilesValidator":
        action.webfilesValidator().run();
        break;
      default:
        output.printf("\n%sFirst flag to ClosureWorker should be program to run\n", Prefixes.ERROR);
        failed.set(true);
    }
  }

  @Singleton
  @Component
  interface Server extends WorkerComponent<ClosureWorker, Invocation, Invocation.Builder> {
    PersistentWorker<Server> worker();

    @Component.Builder
    interface Builder {
      @BindsInstance Builder fs(FileSystem fs);
      Server build();
    }
  }

  @Subcomponent(modules = ActionModule.class)
  interface Invocation extends ActionComponent<ClosureWorker> {
    LegacyAspect<JsChecker.Program> jsChecker();
    LegacyAspect<JsCompiler> jsCompiler();
    LegacyAspect<WebfilesValidatorProgram> webfilesValidator();

    @Subcomponent.Builder
    interface Builder extends ActionComponent.Builder<ClosureWorker, Invocation, Builder> {}
  }

  public static void main(String[] args) throws IOException {
    System.exit(
        DaggerClosureWorker_Server.builder()
            .fs(FileSystems.getDefault())
            .build()
            .worker()
            .run(ImmutableList.copyOf(args)));
  }
}
