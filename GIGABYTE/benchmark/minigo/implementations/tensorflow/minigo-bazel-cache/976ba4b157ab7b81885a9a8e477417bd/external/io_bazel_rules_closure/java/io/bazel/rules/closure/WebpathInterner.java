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

package io.bazel.rules.closure;

import static com.google.common.base.Preconditions.checkArgument;

import com.google.common.base.Function;
import java.util.HashMap;
import java.util.Map;
import javax.inject.Inject;

/**
 * Registry of {@link Webpath} instances.
 *
 * <p>This class should be used when a nontrivial number of web paths are being used in data
 * structures. This is because {@link Webpath#equals(Object)}, {@link Webpath#hashCode()}, and
 * {@link Webpath#compareTo(Webpath)} use memoization internally for performance. Therefore it is
 * nice, but not a requirement, to not have multiple instances of the same web path.
 *
 * <p>See <a href="https://en.wikipedia.org/wiki/String_interning">string interning</a> for more
 * information.
 */
public final class WebpathInterner implements Function<String, Webpath> {

  private final Map<String, Webpath> pool = new HashMap<>(256);

  @Inject
  public WebpathInterner() {}

  /**
   * Delegates to {@link Webpath#get(String)} with regional interning.
   *
   * @throws IllegalArgumentException if {@code path} has superfluous slashes
   */
  public Webpath get(String path) {
    Webpath result = pool.get(path);
    if (result == null) {
      checkArgument(!path.contains("//"), "Interned webpath with superfluous slashes: %s", path);
      result = Webpath.get(path);
      pool.put(path, result);
    }
    return result;
  }

  @Override
  @Deprecated
  public Webpath apply(String input) {
    return get(input);
  }
}
