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

import com.google.common.hash.HashCode;
import java.nio.file.Path;
import java.util.Collection;
import java.util.Map;
import java.util.Set;

/** Fake hash map so {@link InputCache} can work when not run as a persistent worker. */
public final class FakeInputDigestMap implements Map<Path, HashCode> {

  private static final HashCode FAKE_DIGEST = HashCode.fromInt(0);

  @Override
  public int size() {
    return 1;
  }

  @Override
  public boolean isEmpty() {
    return false;
  }

  @Override
  public boolean containsKey(Object key) {
    return true;
  }

  @Override
  public boolean containsValue(Object value) {
    return true;
  }

  @Override
  public HashCode get(Object key) {
    return FAKE_DIGEST;
  }

  @Override
  public HashCode put(Path key, HashCode value) {
    throw new UnsupportedOperationException();
  }

  @Override
  public HashCode remove(Object key) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void putAll(Map<? extends Path, ? extends HashCode> m) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void clear() {
    throw new UnsupportedOperationException();
  }

  @Override
  public Set<Path> keySet() {
    throw new UnsupportedOperationException();
  }

  @Override
  public Collection<HashCode> values() {
    throw new UnsupportedOperationException();
  }

  @Override
  public Set<Entry<Path, HashCode>> entrySet() {
    throw new UnsupportedOperationException();
  }
}
