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

import com.google.common.base.Optional;
import com.google.common.collect.ImmutableSet;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link ErrorReporter}. */
@RunWith(JUnit4.class)
public class ErrorReporterTest {

  private final ByteArrayOutputStream outputBytes = new ByteArrayOutputStream();
  private final PrintStream output = new PrintStream(outputBytes);
  private final AtomicBoolean failed = new AtomicBoolean();

  @Test
  public void nothingReported_doesNotFailOrPrintAnything() throws Exception {
    ErrorReporter reporter = create(null, null);
    print(reporter);
    assertThat(failed.get()).isFalse();
    assertThat(getOutput()).isEmpty();
  }

  @Test
  public void errorReported_failsAndPrintsError() throws Exception {
    ErrorReporter reporter = create(null, null);
    reporter.report("doodle", "oh no something happened");
    print(reporter);
    assertThat(failed.get()).isTrue();
    assertThat(getOutput()).containsMatch("ERROR[^\n]*oh no something happened");
  }

  @Test
  public void noLabelOrSuppress_doesntPrintSuppressHelpText() throws Exception {
    ErrorReporter reporter = create(null, null);
    reporter.report("doodle", "oh no something happened");
    print(reporter);
    assertThat(getOutput()).doesNotContain("doodle");
    assertThat(getOutput()).doesNotContain("suppress");
  }

  @Test
  public void hasLabelAndSuppress_showsHowToSuppressError() throws Exception {
    ErrorReporter reporter = create("//a:b", ImmutableSet.<String>of());
    reporter.report("doodle", "oh no something happened");
    print(reporter);
    assertThat(getOutput()).containsMatch("suppress.*doodle.*//a:b");
  }

  @Test
  public void errorIsExplicitlySuppressed_doesntPrintOrFail() throws Exception {
    ErrorReporter reporter = create("//a:b", ImmutableSet.of("doodle"));
    reporter.report("doodle", "oh no something happened");
    print(reporter);
    assertThat(failed.get()).isFalse();
    assertThat(getOutput()).isEmpty();
  }

  @Test
  public void errorIsWildcardSuppressed_printsButDoesntFail() throws Exception {
    ErrorReporter reporter = create("//a:b", ImmutableSet.of("*"));
    reporter.report("doodle", "oh no something happened");
    print(reporter);
    assertThat(failed.get()).isFalse();
    assertThat(getOutput()).containsMatch("WARNING[^\n]*oh no something happened");
    assertThat(getOutput()).containsMatch("suppress.*doodle.*//a:b");
  }

  @Test
  public void wildcardAndExplicitSuppression_showsWarningsUnlessExplicit() throws Exception {
    ErrorReporter reporter = create("//a:b", ImmutableSet.of("doodle", "*"));
    reporter.report("doodle", "oh no something happened");
    reporter.report("another", "sssh no more tears");
    print(reporter);
    assertThat(failed.get()).isFalse();
    assertThat(getOutput()).containsMatch("WARNING[^\n]*sssh no more tears");
    assertThat(getOutput()).doesNotContainMatch("oh no something happened");
  }

  @Test
  public void superfluousSuppressCode_showsError() throws Exception {
    ErrorReporter reporter = create("//a:b", ImmutableSet.of("doodle"));
    print(reporter);
    assertThat(failed.get()).isTrue();
    assertThat(getOutput()).containsMatch("ERROR[^\n]*[Ss]uperfluous");
  }

  @Test
  public void suppressSuperfluousSuppressCode_showsWarning() throws Exception {
    ErrorReporter reporter = create("//a:b", ImmutableSet.of("doodle", "superfluousSuppress"));
    print(reporter);
    assertThat(failed.get()).isFalse();
    assertThat(getOutput()).isEmpty();
  }

  private ErrorReporter create(@Nullable String label, @Nullable ImmutableSet<String> suppress) {
    return new ErrorReporter(Optional.fromNullable(label), failed, Optional.fromNullable(suppress));
  }

  private void print(ErrorReporter reporter) throws Exception {
    new ErrorReporter.Aspect<>(
            new Program() {
              @Override
              public void run() throws Exception {}
            },
            reporter,
            output)
        .run();
  }

  private String getOutput() {
    return new String(outputBytes.toByteArray(), UTF_8);
  }
}
