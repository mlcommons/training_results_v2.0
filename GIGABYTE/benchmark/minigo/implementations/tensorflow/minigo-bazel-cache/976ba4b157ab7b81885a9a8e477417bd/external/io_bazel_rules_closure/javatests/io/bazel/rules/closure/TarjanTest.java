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

package io.bazel.rules.closure;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import io.bazel.rules.closure.Tarjan.Result;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link Tarjan}. */
@RunWith(JUnit4.class)
public class TarjanTest {

  @Test
  public void testEmpty_hasNoCycles() {
    Result<Object> result = Tarjan.run(ImmutableMultimap.of());
    assertThat(result.getReverseTopologicallyOrderedVertices()).isEmpty();
    assertThat(result.getStronglyConnectedComponents()).isEmpty();
  }

  @Test
  public void testDagWithOneEdge_hasNoCycles() {
    Result<String> result = Tarjan.run(ImmutableMultimap.of("A", "B"));
    assertThat(result.getStronglyConnectedComponents()).isEmpty();
    assertThat(result.getReverseTopologicallyOrderedVertices()).containsExactly("B", "A").inOrder();
  }

  @Test
  public void testMultipleEdges() {
    Result<String> result =
        Tarjan.run(
            new ImmutableMultimap.Builder<String, String>()
                .put("A", "B")
                .put("B", "D")
                .put("B", "C")
                .put("A", "C")
                .put("A", "D")
                .build());
    assertThat(result.getStronglyConnectedComponents()).isEmpty();
    assertThat(result.getReverseTopologicallyOrderedVertices())
        .containsExactly("D", "C", "B", "A")
        .inOrder();
  }

  @Test
  public void testSelfReferential_returnsVertex() {
    assertThat(
            Tarjan.run(ImmutableMultimap.of("A", "A"))
                .getStronglyConnectedComponents())
        .containsExactly(ImmutableSet.of("A"));
  }

  @Test
  public void testTwoVerticesInCycle_returnsSingleSubsetWithBothVertices() {
    assertThat(
            Tarjan.run(
                    ImmutableMultimap.of(
                        "A", "B",
                        "B", "A"))
                .getStronglyConnectedComponents())
        .containsExactly(ImmutableSet.of("A", "B"));
  }

  @Test
  public void testDisjointCycles_returnsDisjointSets() {
    assertThat(
            Tarjan.run(
                    ImmutableMultimap.of(
                        "C", "D",
                        "D", "C",
                        "A", "B",
                        "B", "A"))
                .getStronglyConnectedComponents())
        .containsExactly(ImmutableSet.of("A", "B"), ImmutableSet.of("C", "D"));
  }

  @Test
  public void testWikipediaExample() {
    assertThat(
            Tarjan.run(
                    new ImmutableMultimap.Builder<String, String>()
                        .put("F", "B")
                        .put("G", "C")
                        .put("H", "D")
                        .put("E", "B")
                        .put("F", "E")
                        .put("F", "G")
                        .put("G", "F")
                        .put("H", "G")
                        .put("H", "H")
                        .put("B", "A")
                        .put("C", "B")
                        .put("D", "C")
                        .put("C", "D")
                        .put("A", "E")
                        .build())
                .getStronglyConnectedComponents())
        .containsExactly(
            ImmutableSet.of("B", "E", "A"),
            ImmutableSet.of("D", "C"),
            ImmutableSet.of("G", "F"),
            ImmutableSet.of("H"))
        .inOrder();
  }

  @Test
  public void disjointGraphWithCycle_detectsCycleAndVertexListIsWeirdCompleteAndDeterministic() {
    Result<String> result =
        Tarjan.run(
            new ImmutableMultimap.Builder<String, String>()
                .put("E", "F")
                .put("B", "C")
                .put("D", "E")
                .put("A", "B")
                .put("A", "C")
                .put("F", "E")
                .build());
    assertThat(result.getStronglyConnectedComponents()).containsExactly(ImmutableSet.of("E", "F"));
    assertThat(result.getReverseTopologicallyOrderedVertices())
        .containsExactly("E", "F", "C", "B", "D", "A")
        .inOrder();
  }
}
