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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.common.collect.Multimap;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** Class implementing Tarjan's strongly connected components algorithm. */
public final class Tarjan {

  /**
   * Runs Tarjan's strongly connected components algorithm which is O(|V| + |E|).
   *
   * <p>This algorithm is advantageous when one wishes to do a reverse topological sort in linear
   * time while detecting all cycles for free.
   *
   * <p>A "strongly connected component" is a subset of vertices within a graph where every vertex
   * is reachable from every other vertex. It's a cluster of cycles basically. We use the term
   * "cluster" because the ordering of each cycle might not intuitively reflect the shape of a
   * directed graph that produced these edges.
   *
   * <p><b>Note:</b> Since we use {@link Multimap} to represent the graph, the user may need to take
   * additional steps to handle situations where edgeless vertices exist.
   *
   * <p><b>Note:</b> Topological orderings can be expressed in multiple ways and this implementation
   * does not provide any ordering guarantees beyond topological. However it does guarantee that it
   * will be deterministic, so long as the underlying implementation of {@code edges} is linked.
   *
   * @param edges set of vertex connections in graph
   * @param <V> vertex type, which should have a fast {@code hashCode()} method
   */
  public static <V> Result<V> run(Multimap<V, V> edges) {
    return new Finder<>(checkNotNull(edges)).getComponents();
  }

  /** Result of {@link Tarjan#run(Multimap)}. */
  @AutoValue
  public abstract static class Result<V> {

    /** Returns strongly connected components in graph or empty if it was acyclic. */
    public abstract ImmutableSet<ImmutableSet<V>> getStronglyConnectedComponents();

    /**
     * Returns vertices associated with edges in reverse topological order.
     *
     * <p><b>Note:</b> This might not contain all vertices that exist in the graph, because this
     * implementation only infers vertices from edges and therefore has no awareness of vertices
     * without any edges. Consider {@link com.google.common.collect.Sets#union(java.util.Set,
     * java.util.Set) Sets.union()} for this purpose.
     *
     * <p><b>Warning:</b> If {@link #getStronglyConnectedComponents()} is non-empty, then this
     * ordering will only apply to non-cyclic vertices. The cyclic ones will be inserted wherever
     * they happen to be encountered.
     */
    public abstract ImmutableSet<V> getReverseTopologicallyOrderedVertices();

    Result() {}
  }

  private static final class Vertex<V> {
    final V vertex;
    final int index;
    int lowlink;
    boolean onStack;
    boolean selfReferential;

    Vertex(V vertex, int index) {
      this.vertex = vertex;
      this.index = index;
      this.lowlink = index;
    }
  }

  private static final class Finder<V> {
    private final ImmutableSet.Builder<ImmutableSet<V>> result = new ImmutableSet.Builder<>();
    private final Multimap<V, V> edges;
    private final Map<V, Vertex<V>> vertices;
    private final List<Vertex<V>> stack;
    private final ImmutableSet.Builder<V> sorted = new ImmutableSet.Builder<>();
    private int index;

    Finder(Multimap<V, V> edges) {
      this.edges = edges;
      int wildGuessForInitialCapacity = Math.max(16, edges.size() / 4);
      this.stack = new ArrayList<>(wildGuessForInitialCapacity);
      this.vertices = Maps.newLinkedHashMapWithExpectedSize(wildGuessForInitialCapacity);
    }

    Result<V> getComponents() {
      for (V vertex : edges.values()) {
        if (!vertices.containsKey(vertex)) {
          connectStrongly(vertex);
        }
      }
      for (V vertex : edges.keySet()) {
        if (!vertices.containsKey(vertex)) {
          connectStrongly(vertex);
        }
      }
      return new AutoValue_Tarjan_Result<>(result.build(), sorted.build());
    }

    private Vertex<V> connectStrongly(V vertex) {
      // Set the depth index for v to the smallest unused index.
      Vertex<V> v = new Vertex<>(vertex, index++);
      vertices.put(vertex, v);
      stack.add(v);
      v.onStack = true;

      // Consider successors of v.
      for (V vertex2 : edges.get(v.vertex)) {
        Vertex<V> w = vertices.get(vertex2);
        if (w == null) {
          // Successor w has not yet been visited; recurse on it.
          w = connectStrongly(vertex2);
          v.lowlink = Math.min(v.lowlink, w.lowlink);
        } else if (w.onStack) {
          // Successor w is in stack and hence in the current SCC.
          v.lowlink = Math.min(v.lowlink, w.index);
        }
        if (w.equals(v)) {
          w.selfReferential = true;
        }
      }

      // If v is a root node, pop the stack and generate an SCC.
      if (v.lowlink == v.index) {
        if (!v.selfReferential && v.equals(stack.get(stack.size() - 1))) {
          sorted.add(stack.remove(stack.size() - 1).vertex);
          v.onStack = false;
        } else {
          ImmutableSet.Builder<V> scc = new ImmutableSet.Builder<>();
          Vertex<V> w;
          do {
            w = stack.remove(stack.size() - 1);
            w.onStack = false;
            sorted.add(w.vertex);
            scc.add(w.vertex);
          } while (!w.equals(v));
          result.add(scc.build());
        }
      }

      return v;
    }
  }
}
