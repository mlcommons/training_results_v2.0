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

package io.bazel.rules.closure.http.filter;

import static com.google.common.truth.Truth.assertThat;

import io.bazel.rules.closure.http.HttpResponse;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link NoCacheFilter}. */
@RunWith(JUnit4.class)
public class NoCacheFilterTest {

  @Test
  public void setsCacheControlAndExpires() throws Exception {
    HttpResponse response = new HttpResponse();
    new NoCacheFilter(response).apply();
    assertThat(response.getHeader("Cache-Control")).contains("no-cache");
    assertThat(response.getHeader("Expires")).isEqualTo("0");
  }

  @Test
  public void cacheControlAlreadySet_doesNothing() throws Exception {
    HttpResponse response = new HttpResponse().setHeader("cache-control", "doodle");
    new NoCacheFilter(response).apply();
    assertThat(response.getHeader("Cache-Control")).isEqualTo("doodle");
    assertThat(response.getHeader("Expires")).isEmpty();
  }

  @Test
  public void expiresAlreadySet_doesNothing() throws Exception {
    HttpResponse response = new HttpResponse().setHeader("expires", "doodle");
    new NoCacheFilter(response).apply();
    assertThat(response.getHeader("Cache-Control")).isEmpty();
    assertThat(response.getHeader("Expires")).isEqualTo("doodle");
  }
}
