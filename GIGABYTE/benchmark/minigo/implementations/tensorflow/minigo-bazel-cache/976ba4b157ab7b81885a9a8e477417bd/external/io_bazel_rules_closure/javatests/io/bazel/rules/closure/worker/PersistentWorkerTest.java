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

import com.google.common.collect.ImmutableList;
import com.google.common.hash.HashCode;
import com.google.common.jimfs.Jimfs;
import com.google.devtools.build.lib.worker.WorkerProtocol.Input;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkRequest;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkResponse;
import com.google.protobuf.ByteString;
import dagger.BindsInstance;
import dagger.Component;
import dagger.Subcomponent;
import io.bazel.rules.closure.worker.Annotations.Action;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.file.FileSystem;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.inject.Inject;
import org.junit.After;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link PersistentWorker}. */
@RunWith(JUnit4.class)
public class PersistentWorkerTest {

  public static class Command implements Program {
    private final List<String> args;
    private final Map<Path, HashCode> inputDigests;
    private final AtomicBoolean failed;
    private final PrintStream output;

    @Inject
    Command(
        @Action List<String> args,
        @Action Map<Path, HashCode> inputDigests,
        @Action AtomicBoolean failed,
        @Action PrintStream output) {
      this.args = args;
      this.inputDigests = inputDigests;
      this.failed = failed;
      this.output = output;
    }

    @Override
    public void run() throws Exception {
      failed.set(Boolean.valueOf(args.get(0)));
      output.println(args.get(1));
      output.println(inputDigests);
    }
  }

  @Component
  interface Server extends WorkerComponent<Command, Invocation, Invocation.Builder> {
    PersistentWorker<Server> worker();

    @Component.Builder
    interface Builder {
      @BindsInstance Builder fs(FileSystem mnemonic);
      Server build();
    }
  }

  @Subcomponent(modules = ActionModule.class)
  interface Invocation extends ActionComponent<Command> {
    @Subcomponent.Builder
    interface Builder extends ActionComponent.Builder<Command, Invocation, Builder> {}
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
  public void persistentWorker_normalRequest() throws Exception {
    ByteArrayOutputStream requests = new ByteArrayOutputStream();
    WorkRequest.newBuilder()
        .addArguments("false")
        .addArguments("i will not fail")
        .addInputs(
            Input.newBuilder()
                .setPath("/doodle")
                .setDigest(ByteString.copyFrom(new byte[] {0}))
                .build())
        .build()
        .writeDelimitedTo(requests);
    System.setIn(new ByteArrayInputStream(requests.toByteArray()));
    System.setOut(output);
    assertThat(create().run(ImmutableList.of("--persistent_worker"))).isEqualTo(0);
    ByteArrayInputStream responses = new ByteArrayInputStream(outputBytes.toByteArray());
    WorkResponse response = WorkResponse.parseDelimitedFrom(responses);
    assertThat(response.getOutput()).contains("i will not fail");
    assertThat(response.getOutput()).contains("/doodle");
    assertThat(response.getExitCode()).isEqualTo(0);
  }

  @Test
  public void persistentWorker_failedRequest() throws Exception {
    ByteArrayOutputStream requests = new ByteArrayOutputStream();
    WorkRequest.newBuilder()
        .addArguments("true")
        .addArguments("i must fail")
        .build()
        .writeDelimitedTo(requests);
    System.setIn(new ByteArrayInputStream(requests.toByteArray()));
    System.setOut(output);
    assertThat(create().run(ImmutableList.of("--persistent_worker"))).isEqualTo(0);
    ByteArrayInputStream responses = new ByteArrayInputStream(outputBytes.toByteArray());
    WorkResponse response = WorkResponse.parseDelimitedFrom(responses);
    assertThat(response.getOutput()).contains("i must fail");
    assertThat(response.getExitCode()).isEqualTo(1);
  }

  private PersistentWorker<Server> create() {
    return DaggerPersistentWorkerTest_Server.builder()
        .fs(fs)
        .build()
        .worker();
  }
}
