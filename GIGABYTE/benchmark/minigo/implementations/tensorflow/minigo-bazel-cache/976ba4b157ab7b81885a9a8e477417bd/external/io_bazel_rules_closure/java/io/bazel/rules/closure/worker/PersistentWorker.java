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

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.CharMatcher;
import com.google.common.base.Joiner;
import com.google.common.collect.Iterables;
import com.google.common.hash.HashCode;
import com.google.common.io.Closer;
import com.google.devtools.build.lib.worker.WorkerProtocol.Input;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkRequest;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkResponse;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;
import java.nio.file.FileSystem;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.inject.Inject;

/**
 * Bazel worker runner.
 *
 * <p>This class adapts a Java command line program so it can be spawned by Bazel as a persistent
 * worker process that handles multiple invocations per JVM. It will also be backwards compatible
 * with being run as a normal single-invocation command.
 *
 * @param <T> worker component type
 */
public final class PersistentWorker<T extends WorkerComponent<?, ?, ?>> {

  private static final String FLAGFILE_ARG = "--flagfile=";
  private static final String PERSISTENT_WORKER_ARG = "--persistent_worker";

  private final T component;
  private final FileSystem fs;
  private final List<String> arguments = new ArrayList<>();
  private final Map<Path, HashCode> inputDigests = new HashMap<>();

  @Inject
  PersistentWorker(T component, FileSystem fs) {
    this.component = component;
    this.fs = fs;
  }

  /**
   * Runs persistent worker.
   *
   * <p>This method should be called from the main function. If {@code args} contains {@value
   * #PERSISTENT_WORKER_ARG} then it will be run as a persistent worker that consumes {@link
   * WorkRequest} protos from stdin until EOF. Otherwise, it will delegate a single invocation of
   * the program specified in the type parameter.
   *
   * <p>Since this method is intended to be invoked from main, it swallows exceptions, including
   * {@link InterruptedException}, and focuses on returning the result code.
   *
   * @return result code which should be passed to {@link System#exit(int)}
   */
  public int run(List<String> args) throws IOException {
    try {
      if (args.contains(PERSISTENT_WORKER_ARG)) {
        return runAsPersistentWorker();
      } else {
        arguments.addAll(args);
        loadArguments(false);
        return runProgram(System.err, new FakeInputDigestMap());
      }
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      return 1;
    }
  }

  private int runProgram(PrintStream output, Map<Path, HashCode> digests)
      throws InterruptedException {
    AtomicBoolean failed = new AtomicBoolean();
    try (Closer closer = Closer.create()) {
      component
          .newActionComponentBuilder()
          .args(new ArrayList<>(arguments))
          .closer(closer)
          .inputDigests(digests)
          .output(output)
          .failed(failed)
          .build()
          .program()
          .run();
    } catch (Exception e) {
      if (Utilities.wasInterrupted(e)) {
        throw new InterruptedException();
      }
      output.printf(
          "ERROR: Program threw uncaught exception with args: %s%n",
          Joiner.on(' ').join(arguments));
      e.printStackTrace(output);
      return 1;
    }
    return failed.get() ? 1 : 0;
  }

  private int runAsPersistentWorker() throws IOException, InterruptedException {
    InputStream realStdIn = System.in;
    PrintStream realStdOut = System.out;
    PrintStream realStdErr = System.err;
    try (InputStream emptyIn = new ByteArrayInputStream(new byte[0]);
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        PrintStream ps = new PrintStream(buffer)) {
      System.setIn(emptyIn);
      System.setOut(ps);
      System.setErr(ps);
      while (true) {
        WorkRequest request = WorkRequest.parseDelimitedFrom(realStdIn);
        if (request == null) {
          return 0;
        }
        for (Input input : request.getInputsList()) {
          inputDigests.put(
              fs.getPath(input.getPath()), HashCode.fromBytes(input.getDigest().toByteArray()));
        }
        arguments.addAll(request.getArgumentsList());
        loadArguments(true);
        int exitCode = runProgram(ps, Collections.unmodifiableMap(inputDigests));
        WorkResponse.newBuilder()
            .setOutput(new String(buffer.toByteArray(), UTF_8))
            .setExitCode(exitCode)
            .build()
            .writeDelimitedTo(realStdOut);
        realStdOut.flush();
        buffer.reset();
        arguments.clear();
        inputDigests.clear();
      }
    } finally {
      System.setIn(realStdIn);
      System.setOut(realStdOut);
      System.setErr(realStdErr);
    }
  }

  private void loadArguments(boolean isWorker) {
    try {
      String lastArg = Iterables.getLast(arguments, "");
      if (lastArg.startsWith("@")) {
        Path flagFile = fs.getPath(CharMatcher.is('@').trimLeadingFrom(lastArg));
        if ((isWorker && lastArg.startsWith("@@")) || Files.exists(flagFile)) {
          arguments.clear();
          arguments.addAll(Files.readAllLines(flagFile, UTF_8));
        }
      } else {
        List<String> newArguments = new ArrayList<>();
        for (String argument : arguments) {
          if (argument.startsWith(FLAGFILE_ARG)) {
            newArguments.addAll(
                Files.readAllLines(fs.getPath(argument.substring(FLAGFILE_ARG.length())), UTF_8));
          }
        }
        if (!newArguments.isEmpty()) {
          arguments.clear();
          arguments.addAll(newArguments);
        }
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
}
