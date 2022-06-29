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

import com.google.auto.value.AutoValue;
import com.google.common.base.Function;
import com.google.common.cache.LoadingCache;
import com.google.common.hash.HashCode;
import io.bazel.rules.closure.worker.Annotations.Action;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import javax.inject.Inject;

/**
 * Helper for loading files that are cached between ctx.action invocations.
 *
 * <p>In order for this class to be injectable, the user must define a module method providing a
 * {@link LoadingCache} that turns {@link Key InputCache.Key} into a strongly typed value. For
 * example, if {@code @Singleton LoadingCache<InputCache.Key, Foo>} is provided, then {@code
 * InputCache<Foo>} can be injected.
 */
public final class InputCache<T> implements Function<Path, T> {

  /** Key for items in an {@link InputCache}. */
  @AutoValue
  public abstract static class Key {

    /** Hash value of contents, passed to us by Bazel, as an opaque token. */
    abstract HashCode digest();

    /** Path at which this file resides on disk. */
    public abstract Path path();

    Key() {}
  }

  private final LoadingCache<Key, T> cache;
  private final Map<Path, HashCode> digests;

  @Inject
  public InputCache(LoadingCache<Key, T> cache, @Action Map<Path, HashCode> digests) {
    this.cache = cache;
    this.digests = digests;
  }

  /** Loads resource from cache, if available, or from disk. */
  public T load(Path path) throws IOException {
    try {
      return cache.get(makeKey(path));
    } catch (ExecutionException e) {
      String message = "Error reading: " + path;
      if (e.getCause() instanceof IOException) {
        throw new IOException(message, e.getCause());
      } else {
        throw new RuntimeException(message, e);
      }
    }
  }

  @Override
  @Deprecated
  public T apply(Path path) {
    try {
      return load(path);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  private Key makeKey(Path path) {
    HashCode hashCode = digests.get(path);
    if (hashCode == null) {
      throw new IllegalArgumentException("Not declared in ctx.action inputs: " + path);
    }
    AutoValue_InputCache_Key key = new AutoValue_InputCache_Key(hashCode, path);
    return key;
  }
}
