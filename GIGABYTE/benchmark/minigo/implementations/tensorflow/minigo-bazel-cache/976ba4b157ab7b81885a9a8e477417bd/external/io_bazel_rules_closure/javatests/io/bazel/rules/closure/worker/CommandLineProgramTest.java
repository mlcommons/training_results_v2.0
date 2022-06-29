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

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.jimfs.Jimfs;
import dagger.BindsInstance;
import dagger.Component;
import dagger.Subcomponent;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.file.FileSystem;
import javax.inject.Inject;
import org.junit.After;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Example of how to use {@link CommandLineProgram} for legacy programs. */
@RunWith(JUnit4.class)
public class CommandLineProgramTest {

  public static class Command implements CommandLineProgram {

    @Inject
    Command() {}

    @Override
    public Integer apply(Iterable<String> args) {
      String name = Iterables.getFirst(args, null);
      System.err.println("Hello " + name);
      return name.equals("Justine") ? 0 : 1;
    }
  }

  @Component
  interface Server extends WorkerComponent<LegacyAspect<Command>, Invocation, Invocation.Builder> {
    PersistentWorker<Server> worker();

    @Component.Builder
    interface Builder {
      @BindsInstance Builder fs(FileSystem mnemonic);
      Server build();
    }
  }

  @Subcomponent(modules = ActionModule.class)
  interface Invocation extends ActionComponent<LegacyAspect<Command>> {
    @Subcomponent.Builder
    interface Builder extends ActionComponent.Builder<LegacyAspect<Command>, Invocation, Builder> {}
  }

  @Rule public final StdioRestoreRule restoreStdio = new StdioRestoreRule();
  private final FileSystem fs = Jimfs.newFileSystem();
  private final ByteArrayOutputStream outputBytes = new ByteArrayOutputStream();
  private final PrintStream output = new PrintStream(outputBytes);

  @After
  public void closeFileSystem() throws Exception {
    fs.close();
  }

  @Test
  public void bypassPersistentWorker_success() throws Exception {
    System.setErr(output);
    assertThat(create().run(ImmutableList.of("Justine"))).isEqualTo(0);
    assertThat(getOutput()).contains("Hello Justine");
  }

  @Test
  public void bypassPersistentWorker_failure_codePropagatesToRun() throws Exception {
    System.setErr(output);
    assertThat(create().run(ImmutableList.of("Tobias"))).isEqualTo(1);
    assertThat(getOutput()).contains("Hello Tobias");
  }

  private PersistentWorker<Server> create() {
    return DaggerCommandLineProgramTest_Server.builder()
        .fs(fs)
        .build()
        .worker();
  }

  private String getOutput() {
    return new String(outputBytes.toByteArray(), UTF_8);
  }
}
