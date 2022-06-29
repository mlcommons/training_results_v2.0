// Copyright 2016 The Closure Rules Authors. All Rights Reserved.
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

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Joiner;
import com.google.common.base.Supplier;
import com.google.common.base.Suppliers;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Multimap;
import com.google.common.collect.Ordering;
import com.google.common.collect.Sets;
import com.google.protobuf.TextFormat;
import io.bazel.rules.closure.webfiles.BuildInfo.Webfiles;
import io.bazel.rules.closure.worker.Annotations.Action;
import io.bazel.rules.closure.worker.CommandLineProgram;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.FileSystem;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import javax.inject.Inject;

/** CLI for {@link WebfilesValidator}. */
public final class WebfilesValidatorProgram implements CommandLineProgram {

  // Changing these values will break any build code that references them.
  private static final String SUPPRESS_EVERYTHING = "*";
  private static final String SUPERFLUOUS_SUPPRESS_ERROR = "superfluousSuppress";

  private static final ImmutableSet<String> NEVER_SUPERFLUOUS =
      ImmutableSet.of(SUPPRESS_EVERYTHING, SUPERFLUOUS_SUPPRESS_ERROR);

  private static final String RESET = "\u001b[0m";
  private static final String BOLD = "\u001b[1m";
  private static final String RED = "\u001b[31m";
  private static final String BLUE = "\u001b[34m";
  private static final String MAGENTA = "\u001b[35m";
  private static final String ERROR_PREFIX = String.format("%s%sERROR:%s ", BOLD, RED, RESET);
  private static final String WARNING_PREFIX = String.format("%sWARNING:%s ", MAGENTA, RESET);
  private static final String NOTE_PREFIX = String.format("%s%sNOTE:%s ", BOLD, BLUE, RESET);

  private final PrintStream output;
  private final FileSystem fs;
  private final WebfilesValidator validator;

  @Inject
  WebfilesValidatorProgram(@Action PrintStream output, FileSystem fs, WebfilesValidator validator) {
    this.output = output;
    this.fs = fs;
    this.validator = validator;
  }

  @Override
  public Integer apply(Iterable<String> args) {
    try {
      return run(args);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  private int run(Iterable<String> args) throws IOException {
    Webfiles target = null;
    List<Webfiles> directDeps = new ArrayList<>();
    final List<Path> transitiveDeps = new ArrayList<>();
    Iterator<String> flags = args.iterator();
    Set<String> suppress = new HashSet<>();
    while (flags.hasNext()) {
      String flag = flags.next();
      switch (flag) {
        case "--dummy":
          Files.write(fs.getPath(flags.next()), new byte[0]);
          break;
        case "--target":
          target = loadWebfilesPbtxt(fs.getPath(flags.next()));
          break;
        case "--direct_dep":
          directDeps.add(loadWebfilesPbtxt(fs.getPath(flags.next())));
          break;
        case "--transitive_dep":
          transitiveDeps.add(fs.getPath(flags.next()));
          break;
        case "--suppress":
          suppress.add(flags.next());
          break;
        default:
          throw new RuntimeException("Unexpected flag: " + flag);
      }
    }
    if (target == null) {
      output.println(ERROR_PREFIX + "Missing --target flag");
      return 1;
    }
    Multimap<String, String> errors =
        validator.validate(
            target,
            directDeps,
            Suppliers.memoize(
                new Supplier<ImmutableList<Webfiles>>() {
                  @Override
                  public ImmutableList<Webfiles> get() {
                    ImmutableList.Builder<Webfiles> builder = new ImmutableList.Builder<>();
                    for (Path path : transitiveDeps) {
                      try {
                        builder.add(loadWebfilesPbtxt(path));
                      } catch (IOException e) {
                        throw new RuntimeException(e);
                      }
                    }
                    return builder.build();
                  }
                }));
    Set<String> superfluous =
        Sets.difference(suppress, Sets.union(errors.keySet(), NEVER_SUPERFLUOUS));
    if (!superfluous.isEmpty()) {
      errors.put(
          SUPERFLUOUS_SUPPRESS_ERROR, "Superfluous suppress codes: " + joinWords(superfluous));
    }
    return displayErrors(suppress, errors);
  }

  private int displayErrors(Set<String> suppress, Multimap<String, String> errors) {
    int exitCode = 0;
    for (String category : errors.keySet()) {
      boolean ignored = suppress.contains(category) || suppress.contains(SUPPRESS_EVERYTHING);
      String prefix = ignored ? WARNING_PREFIX : ERROR_PREFIX;
      for (String error : errors.get(category)) {
        output.println(prefix + error);
      }
      if (!ignored) {
        exitCode = 1;
        output.printf(
            "%sUse suppress=[\"%s\"] to make the errors above warnings%n", NOTE_PREFIX, category);
      }
    }
    return exitCode;
  }

  private static String joinWords(Iterable<String> words) {
    return Joiner.on(", ").join(Ordering.natural().immutableSortedCopy(words));
  }

  private static Webfiles loadWebfilesPbtxt(Path path) throws IOException {
    Webfiles.Builder build = Webfiles.newBuilder();
    TextFormat.getParser().merge(new String(Files.readAllBytes(path), UTF_8), build);
    return build.build();
  }
}
