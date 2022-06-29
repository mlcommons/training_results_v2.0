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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.base.Joiner;
import com.google.common.base.Optional;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Ordering;
import com.google.common.collect.Sets;
import io.bazel.rules.closure.worker.Annotations.Action;
import io.bazel.rules.closure.worker.Annotations.ActionScope;
import io.bazel.rules.closure.worker.Annotations.Suppress;
import java.io.PrintStream;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.inject.Inject;

/** Error reporter for Bazel actions. */
@ActionScope
public class ErrorReporter {

  /** Wrapper around a program that prints errors afterwards. */
  public static final class Aspect<T extends Program> implements Program {

    private final Program delegate;
    private final ErrorReporter reporter;
    private final PrintStream output;

    @Inject
    Aspect(T delegate, ErrorReporter reporter, @Action PrintStream output) {
      this.delegate = delegate;
      this.reporter = reporter;
      this.output = output;
    }

    @Override
    public void run() throws Exception {
      delegate.run();
      reporter.finish();
      displayMessages(Prefixes.ERROR, reporter.errors);
      displayMessages(Prefixes.WARNING, reporter.warnings);
    }

    private void displayMessages(String prefix, Set<String> messages) {
      for (String message : messages) {
        output.println(prefix + message);
        String note = reporter.notes.get(message);
        if (note != null) {
          output.println(Prefixes.NOTE + note);
        }
      }
    }
  }

  private static final String SUPPRESS_EVERYTHING = "*";
  private static final String SUPERFLUOUS_SUPPRESS_ERROR = "superfluousSuppress";
  private static final ImmutableSet<String> NEVER_SUPERFLUOUS =
      ImmutableSet.of(SUPPRESS_EVERYTHING, SUPERFLUOUS_SUPPRESS_ERROR);

  private final Optional<String> label;
  private final AtomicBoolean failed;
  private final Optional<ImmutableSet<String>> suppress;
  private final Set<String> suppressed = new LinkedHashSet<>();
  private final Set<String> errors = new LinkedHashSet<>();
  private final Set<String> warnings = new LinkedHashSet<>();
  private final Map<String, String> notes = new HashMap<>();

  @Inject
  public ErrorReporter(
      @Action Optional<String> label,
      @Action AtomicBoolean failed,
      @Suppress Optional<ImmutableSet<String>> suppress) {
    this.label = label;
    this.failed = failed;
    this.suppress = suppress;
  }

  /** Reports an error message with its associated category code. */
  public void report(String code, String message) {
    checkNotNull(code);
    checkArgument(!message.isEmpty());
    Set<String> destination;
    if (suppress.isPresent()) {
      if (code.isEmpty()) {
        if (suppress.get().contains(SUPPRESS_EVERYTHING)) {
          destination = warnings;
        } else {
          destination = errors;
          failed.set(true);
        }
      } else {
        if (suppress.get().contains(code)) {
          destination = null;
          suppressed.add(code);
        } else {
          if (suppress.get().contains(SUPPRESS_EVERYTHING)) {
            destination = warnings;
          } else {
            destination = errors;
            failed.set(true);
          }
          if (label.isPresent()) {
            notes.put(
                message,
                String.format(
                    "To make this go away add suppress=[\"%s\"] to %s", code, label.get()));
          }
        }
      }
    } else {
      destination = errors;
      failed.set(true);
    }
    if (destination != null) {
      destination.add(message);
    }
  }

  /** Returns errors reported by program. */
  public Set<String> getErrors() {
    finish();
    return Collections.unmodifiableSet(errors);
  }

  /** Returns warnings reported by program. */
  public Set<String> getWarnings() {
    finish();
    return Collections.unmodifiableSet(warnings);
  }

  private void finish() {
    if (suppress.isPresent()) {
      Set<String> superfluous =
          Sets.difference(suppress.get(), Sets.union(suppressed, NEVER_SUPERFLUOUS));
      if (!superfluous.isEmpty()) {
        report(SUPERFLUOUS_SUPPRESS_ERROR, "Superfluous suppress codes: " + joinWords(superfluous));
      }
    }
  }

  private static String joinWords(Iterable<String> words) {
    return Joiner.on(", ").join(Ordering.natural().immutableSortedCopy(words));
  }
}
