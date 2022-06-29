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

package io.bazel.rules.closure.webfiles.server;

import static com.google.common.io.Resources.getResource;

import com.google.common.base.Functions;
import com.google.common.base.Predicate;
import com.google.common.collect.FluentIterable;
import com.google.common.collect.ImmutableSet;
import com.google.common.net.MediaType;
import com.google.template.soy.SoyFileSet;
import com.google.template.soy.data.SoyListData;
import com.google.template.soy.data.SoyMapData;
import com.google.template.soy.tofu.SoyTofu;
import io.bazel.rules.closure.Webpath;
import io.bazel.rules.closure.http.HttpResponse;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import javax.inject.Inject;

/**
 * Web page listing webpaths in transitive closure in topological order.
 *
 * <p>The webfiles server uses this as the directory listing and 404 page.
 */
class ListingPage {

  private static final SoyTofu TOFU =
      SoyFileSet.builder()
          .add(getResource(ListingSoyInfo.class, ListingSoyInfo.getInstance().getFileName()))
          .build()
          .compileToTofu();

  private final HttpResponse response;
  private final Metadata.Config config;
  private final ImmutableSet<Webpath> webpaths;

  @Inject
  ListingPage(HttpResponse response, Metadata.Config config, ImmutableSet<Webpath> webpaths) {
    this.response = response;
    this.config = config;
    this.webpaths = webpaths;
  }

  void serve(final Webpath webpath) throws IOException {
    response.setContentType(MediaType.HTML_UTF_8);
    response.setPayload(
        TOFU.newRenderer(ListingSoyInfo.LISTING)
            .setData(
                new SoyMapData(
                    ListingSoyInfo.ListingSoyTemplateInfo.LABEL,
                    config.get().getLabel(),
                    ListingSoyInfo.ListingSoyTemplateInfo.PATHS,
                    new SoyListData(
                        FluentIterable.from(webpaths)
                            .filter(
                                new Predicate<Webpath>() {
                                  @Override
                                  public boolean apply(Webpath path) {
                                    return path.startsWith(webpath);
                                  }
                                })
                            .transform(Functions.toStringFunction()))))
            .render()
            .getBytes(StandardCharsets.UTF_8));
  }
}
