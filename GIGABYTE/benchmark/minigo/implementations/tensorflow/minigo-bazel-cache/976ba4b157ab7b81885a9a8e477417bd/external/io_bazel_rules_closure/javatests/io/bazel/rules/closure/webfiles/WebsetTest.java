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

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.jimfs.Configuration;
import com.google.common.jimfs.Jimfs;
import com.google.common.testing.NullPointerTester;
import io.bazel.rules.closure.Webpath;
import io.bazel.rules.closure.WebpathInterner;
import io.bazel.rules.closure.webfiles.BuildInfo.MultimapInfo;
import io.bazel.rules.closure.webfiles.BuildInfo.WebfileInfo;
import io.bazel.rules.closure.webfiles.BuildInfo.WebfileManifestInfo;
import java.nio.file.FileSystem;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.Map;
import java.util.Set;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link Webset}. */
@RunWith(JUnit4.class)
public class WebsetTest {

  private static final FileSystem fs = Jimfs.newFileSystem(Configuration.unix());

  private static final String LABEL_A = "//:a";
  private static final Path MANIFEST_A = fs.getPath("bazel-bin/a.pb");

  private static final String LABEL_B = "//:b";
  private static final Path MANIFEST_B = fs.getPath("bazel-bin/b.pb");

  private static final String LABEL_C = "//:c";
  private static final Path MANIFEST_C = fs.getPath("bazel-bin/c.pb");

  private static final Webfile A_JPG = makeWebfile(MANIFEST_A, LABEL_A, "/a.jpg");
  private static final Webfile A_JS = makeWebfile(MANIFEST_A, LABEL_A, "/a.js");
  private static final Webfile B_HTML = makeWebfile(MANIFEST_B, LABEL_B, "/b.html");
  private static final Webfile C_HTML = makeWebfile(MANIFEST_C, LABEL_C, "/c.html");
  private static final Webfile C2_HTML = makeWebfile(MANIFEST_C, LABEL_C, "/c2.html");

  private static final WebfileManifestInfo MANIFEST_INFO_A =
      WebfileManifestInfo.newBuilder()
          .setLabel(LABEL_A)
          .addWebfile(A_JPG.info())
          .addWebfile(A_JS.info())
          .build();

  private static final WebfileManifestInfo MANIFEST_INFO_B =
      WebfileManifestInfo.newBuilder()
          .setLabel(LABEL_B)
          .addWebfile(B_HTML.info())
          .addLink(makeLink("/b.html", "/a.jpg"))
          .build();

  private static final WebfileManifestInfo MANIFEST_INFO_C =
      WebfileManifestInfo.newBuilder()
          .setLabel(LABEL_C)
          .addWebfile(C_HTML.info())
          .addWebfile(C2_HTML.info())
          .addLink(makeLink("/c.html", "/b.html", "/a.jpg", "/a.js"))
          .addLink(makeLink("/c2.html", "/b.html"))
          .build();

  private static final ImmutableMap<Path, WebfileManifestInfo> MANIFESTS =
      ImmutableMap.of(
          MANIFEST_A, MANIFEST_INFO_A,
          MANIFEST_B, MANIFEST_INFO_B,
          MANIFEST_C, MANIFEST_INFO_C);

  private final WebpathInterner interner = new WebpathInterner();

  @Rule
  public final ExpectedException thrown = ExpectedException.none();

  @Test
  public void nulls() throws Exception {
    NullPointerTester npt = new NullPointerTester();
    npt.setDefault(Map.class, Collections.emptyMap());
    npt.setDefault(Set.class, Collections.emptySet());
    npt.setDefault(WebpathInterner.class, interner);
    npt.testAllPublicStaticMethods(Webset.class);
  }

  @Test
  public void load_emptyInputs_emptyOutputs() throws Exception {
    Webset webset =
        Webset.load(Collections.<Path, WebfileManifestInfo>emptyMap(), interner)
            .removeWebfilesNotReachableFrom(Collections.<Webpath>emptySet());
    assertThat(webset.webfiles()).isEmpty();
    assertThat(webset.links()).isEmpty();
  }

  @Test
  public void load_nothingReachable_returnsEmpty() throws Exception {
    Webset webset =
        Webset.load(MANIFESTS, interner)
            .removeWebfilesNotReachableFrom(Collections.<Webpath>emptySet());
    assertThat(webset.webfiles()).isEmpty();
    assertThat(webset.links()).isEmpty();
  }

  @Test
  public void load_leafReachable_returnsJustLeaf() throws Exception {
    Webset webset =
        Webset.load(MANIFESTS, interner)
            .removeWebfilesNotReachableFrom(ImmutableSet.of(Webpath.get("/a.jpg")));
    assertThat(webset.webfiles()).containsExactly(Webpath.get("/a.jpg"), A_JPG);
    assertThat(webset.links()).isEmpty();
  }

  @Test
  public void load_branchReachable_returnsBranchAndLeaves() throws Exception {
    Webset webset =
        Webset.load(MANIFESTS, interner)
            .removeWebfilesNotReachableFrom(ImmutableSet.of(Webpath.get("/b.html")));
    assertThat(webset.webfiles())
        .containsExactly(
            A_JPG.webpath(), A_JPG,
            B_HTML.webpath(), B_HTML)
        .inOrder();
    assertThat(webset.links())
        .containsExactlyEntriesIn(
            ImmutableMultimap.of(Webpath.get("/b.html"), Webpath.get("/a.jpg")));
  }

  @Test
  public void load_trunkReachable_returnsTrunkBranchAndLeaves() throws Exception {
    Webset webset =
        Webset.load(MANIFESTS, interner)
            .removeWebfilesNotReachableFrom(ImmutableSet.of(Webpath.get("/c2.html")));
    assertThat(webset.webfiles())
        .containsExactly(
            A_JPG.webpath(), A_JPG,
            B_HTML.webpath(), B_HTML,
            C2_HTML.webpath(), C2_HTML)
        .inOrder();
    assertThat(webset.links())
        .containsExactlyEntriesIn(
            ImmutableMultimap.of(
                Webpath.get("/b.html"), Webpath.get("/a.jpg"),
                Webpath.get("/c2.html"), Webpath.get("/b.html")))
        .inOrder();
  }

  @Test
  public void load_doubleDipping_orderingMakesSense() throws Exception {
    Webset webset =
        Webset.load(MANIFESTS, interner)
            .removeWebfilesNotReachableFrom(
                ImmutableSet.of(Webpath.get("/c.html"), Webpath.get("/c2.html")));
    assertThat(webset.webfiles())
        .containsExactly(
            A_JPG.webpath(), A_JPG,
            A_JS.webpath(), A_JS,
            B_HTML.webpath(), B_HTML,
            C_HTML.webpath(), C_HTML,
            C2_HTML.webpath(), C2_HTML)
        .inOrder();
    assertThat(webset.links())
        .containsExactlyEntriesIn(
            ImmutableMultimap.of(
                Webpath.get("/b.html"), Webpath.get("/a.jpg"),
                Webpath.get("/c.html"), Webpath.get("/b.html"),
                Webpath.get("/c.html"), Webpath.get("/a.jpg"),
                Webpath.get("/c.html"), Webpath.get("/a.js"),
                Webpath.get("/c2.html"), Webpath.get("/b.html")))
        .inOrder();
  }

  private static Webfile makeWebfile(Path manifest, String label, String webpath) {
    return Webfile.create(
        Webpath.get(webpath),
        WebfilesUtils.getIncrementalZipPath(manifest),
        label,
        WebfileInfo.newBuilder().setWebpath(webpath).build());
  }

  private static MultimapInfo makeLink(String key, String... values) {
    return MultimapInfo.newBuilder().setKey(key).addAllValue(Arrays.asList(values)).build();
  }
}
