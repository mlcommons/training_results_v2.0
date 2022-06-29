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

package io.bazel.rules.closure.webfiles;

import static com.google.common.base.Preconditions.checkArgument;

import com.google.auto.value.AutoValue;
import com.google.common.base.Functions;
import com.google.common.collect.Collections2;
import com.google.common.collect.Iterables;
import com.google.common.collect.LinkedHashMultimap;
import com.google.common.collect.Maps;
import com.google.common.collect.Multimap;
import com.google.common.collect.Sets;
import io.bazel.rules.closure.Webpath;
import io.bazel.rules.closure.WebpathInterner;
import io.bazel.rules.closure.webfiles.BuildInfo.MultimapInfo;
import io.bazel.rules.closure.webfiles.BuildInfo.WebfileInfo;
import io.bazel.rules.closure.webfiles.BuildInfo.WebfileManifestInfo;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

/**
 * Mutable graph of web files.
 *
 * <p>This class is a Java wrapper around a set of {@link
 * io.bazel.rules.closure.webfiles.BuildInfo.WebfileManifestInfo WebfileManifestInfo} protos. In the
 * case of an incremental compile, those might be only directly dependent manifests, in which case
 * this is a segment of the graph.
 *
 * <p><b>Note:</b> All {@link Webpath} instances within an instance of this class are guaranteed to
 * be interned by the same interner, unless the user mutates {@link #webfiles()} or {@link
 * #links()}.
 */
@AutoValue
public abstract class Webset {

  /**
   * Loads graph of web files from proto manifests.
   *
   * @param manifests set of web rule target proto files in reverse topological order
   * @return set of web files and relationships between them, which could be mutated, although
   *     adding a single key will most likely result in a full rehash
   */
  public static Webset load(Map<Path, WebfileManifestInfo> manifests, WebpathInterner interner) {
    int webfileCapacity = 0;
    int unlinkCapacity = 16; // LinkedHashMultimap#DEFAULT_KEY_CAPACITY
    for (WebfileManifestInfo manifest : manifests.values()) {
      webfileCapacity += manifest.getWebfileCount();
      unlinkCapacity = Math.max(unlinkCapacity, manifest.getUnlinkCount());
    }
    Map<Webpath, Webfile> webfiles = Maps.newLinkedHashMapWithExpectedSize(webfileCapacity);
    Multimap<Webpath, Webpath> links = LinkedHashMultimap.create(webfileCapacity, 4);
    Multimap<Webpath, Webpath> unlinks = LinkedHashMultimap.create(unlinkCapacity, 4);
    for (Map.Entry<Path, WebfileManifestInfo> entry : manifests.entrySet()) {
      Path manifestPath = entry.getKey();
      Path zipPath = WebfilesUtils.getIncrementalZipPath(manifestPath);
      WebfileManifestInfo manifest = entry.getValue();
      String label = manifest.getLabel();
      for (WebfileInfo info : manifest.getWebfileList()) {
        Webpath webpath = interner.get(info.getWebpath());
        webfiles.put(webpath, Webfile.create(webpath, zipPath, label, info));
      }
      for (MultimapInfo mapping : manifest.getLinkList()) {
        Webpath from = interner.get(mapping.getKey());
        for (Webpath to : Iterables.transform(mapping.getValueList(), interner)) {
          // When compiling web_library rules, if the strict dependency checking invariant holds
          // true, we can choose to only load adjacent manifests, rather than transitive ones. The
          // adjacent manifests may contain links to transitive web files which will not be in the
          // webfiles map.
          if (webfiles.containsKey(to)) {
            links.put(from, to);
            checkArgument(!unlinks.containsEntry(from, to),
                "Has a use case for resurrected links been discovered? %s -> %s", from, to);
          }
        }
      }
      for (MultimapInfo mapping : manifest.getUnlinkList()) {
        unlinks.putAll(
            interner.get(mapping.getKey()),
            Collections2.transform(mapping.getValueList(), interner));
      }
    }
    for (Map.Entry<Webpath, Webpath> entry : unlinks.entries()) {
      links.remove(entry.getKey(), entry.getValue());
    }
    unlinks.clear();
    return new AutoValue_Webset(webfiles, links, interner);
  }

  /**
   * Returns mutable reverse topologically ordered set of web files in graph.
   *
   * <p>This keys in this map is the canonical set of vertices in the graph.
   */
  public abstract Map<Webpath, Webfile> webfiles();

  /**
   * Returns mutable set of edges between web files in reverse topological order.
   *
   * <p><b>Warning:</b> This data structure is allowed to contain superfluous edges to vertices that
   * do not exist in {@link #webfiles()}.
   */
  public abstract Multimap<Webpath, Webpath> links();

  abstract WebpathInterner interner();

  /**
   * Mutates graph to remove web files not reachable from set of entry points.
   *
   * <p>This method fully prunes {@link #webfiles()}. Entries might be removed from {@link #links()}
   * on a best effort basis.
   *
   * @param entryPoints set of paths that should be considered tree tips
   * @throws IllegalArgumentException if {@code entryPoints} aren't defined by {@code manifests}
   * @return {@code this}
   */
  public final Webset removeWebfilesNotReachableFrom(Iterable<Webpath> entryPoints) {
    Deque<Webpath> bfs = new ArrayDeque<>();
    Set<Webpath> visited = Sets.newIdentityHashSet();
    for (Webpath entryPoint :
        Iterables.transform(
            Iterables.transform(entryPoints, Functions.toStringFunction()), interner())) {
      checkArgument(webfiles().containsKey(entryPoint), "Not found: %s", entryPoint);
      bfs.add(entryPoint);
    }
    while (!bfs.isEmpty()) {
      Webpath path = bfs.removeLast();
      if (visited.add(path)) {
        for (Webpath dest : links().get(path)) {
          if (webfiles().containsKey(dest)) {
            bfs.addFirst(dest);
          }
        }
      }
    }
    Iterator<Webpath> webfilesIterator = webfiles().keySet().iterator();
    while (webfilesIterator.hasNext()) {
      Webpath key = webfilesIterator.next();
      if (!visited.contains(key)) {
        webfilesIterator.remove();
        links().removeAll(key);
      }
    }
    return this;
  }

  Webset() {}
}
