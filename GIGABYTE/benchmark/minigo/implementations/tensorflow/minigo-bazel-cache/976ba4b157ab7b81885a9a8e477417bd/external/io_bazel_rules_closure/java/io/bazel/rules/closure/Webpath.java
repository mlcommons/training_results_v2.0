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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.base.Joiner;
import com.google.common.base.Splitter;
import com.google.common.collect.Iterators;
import com.google.common.collect.Lists;
import com.google.common.collect.Ordering;
import com.google.common.collect.PeekingIterator;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import javax.annotation.Nullable;
import javax.annotation.concurrent.Immutable;

/**
 * Web server path.
 *
 * <p>This class is a de facto implementation of the {@link java.nio.file.Path} API. That interface
 * is not formally implemented because it would not be desirable to have web paths accidentally
 * intermingle with file system paths.
 *
 * <p>This implementation is almost identical to {@code sun.nio.fs.UnixPath}. The main difference is
 * that this implementation goes to greater lengths to preserve trailing slashes, since their
 * presence has a distinct meaning in web applications.
 *
 * <p><b>Note:</b> This code might not play nice with <a
 * href="http://docs.oracle.com/javase/tutorial/i18n/text/supplementaryChars.html">Supplementary
 * Characters as Surrogates</a>.
 */
@Immutable
public final class Webpath implements CharSequence, Comparable<Webpath> {

  private static final char SEPARATOR = '/';
  private static final String ROOT = "/";
  private static final String CURRENT_DIR = ".";
  private static final String PARENT_DIR = "..";
  private static final Webpath EMPTY_PATH = new Webpath("");
  private static final Webpath ROOT_PATH = new Webpath(ROOT);

  private static final Splitter SPLITTER = Splitter.on(SEPARATOR).omitEmptyStrings();
  private static final Joiner JOINER = Joiner.on(SEPARATOR);
  private static final Ordering<Iterable<String>> ORDERING = Ordering.natural().lexicographical();

  /** Returns new path of {@code first}. */
  public static Webpath get(String path) {
    if (path.isEmpty()) {
      return EMPTY_PATH;
    } else if (isRootInternal(path)) {
      return ROOT_PATH;
    }
    return new Webpath(path);
  }

  private final String path;
  private int lazyHashCode;
  private List<String> lazyStringParts;

  private Webpath(String path) {
    this.path = checkNotNull(path);
  }

  /** Returns {@code true} consists only of {@code separator}. */
  public boolean isRoot() {
    return isRootInternal(path);
  }

  private static boolean isRootInternal(String path) {
    return path.length() == 1 && path.charAt(0) == SEPARATOR;
  }

  /** Returns {@code true} if path starts with {@code separator}. */
  public boolean isAbsolute() {
    return isAbsoluteInternal(path);
  }

  private static boolean isAbsoluteInternal(String path) {
    return !path.isEmpty() && path.charAt(0) == SEPARATOR;
  }

  /** Returns {@code true} if path ends with {@code separator}. */
  public boolean hasTrailingSeparator() {
    return hasTrailingSeparatorInternal(path);
  }

  private static boolean hasTrailingSeparatorInternal(CharSequence path) {
    return path.length() != 0 && path.charAt(path.length() - 1) == SEPARATOR;
  }

  /** Returns {@code true} if path ends with a trailing slash, or would after normalization. */
  public boolean seemsLikeADirectory() {
    int length = path.length();
    return path.isEmpty()
        || path.charAt(length - 1) == SEPARATOR
        || (path.endsWith(".") && (length == 1 || path.charAt(length - 2) == SEPARATOR))
        || (path.endsWith("..") && (length == 2 || path.charAt(length - 3) == SEPARATOR));
  }

  /**
   * Returns last component in {@code path}.
   *
   * @see java.nio.file.Path#getFileName()
   */
  @Nullable
  public Webpath getFileName() {
    if (path.isEmpty()) {
      return EMPTY_PATH;
    } else if (isRoot()) {
      return null;
    } else {
      List<String> parts = getParts();
      String last = parts.get(parts.size() - 1);
      return parts.size() == 1 && path.equals(last) ? this : new Webpath(last);
    }
  }

  /**
   * Returns parent directory (including trailing separator) or {@code null} if no parent remains.
   *
   * @see java.nio.file.Path#getParent()
   */
  @Nullable
  public Webpath getParent() {
    if (path.isEmpty() || isRoot()) {
      return null;
    }
    int index =
        hasTrailingSeparator()
            ? path.lastIndexOf(SEPARATOR, path.length() - 2)
            : path.lastIndexOf(SEPARATOR);
    if (index == -1) {
      return isAbsolute() ? ROOT_PATH : null;
    } else {
      return new Webpath(path.substring(0, index + 1));
    }
  }

  /**
   * Returns root component if an absolute path, otherwise {@code null}.
   *
   * @see java.nio.file.Path#getRoot()
   */
  @Nullable
  public Webpath getRoot() {
    return isAbsolute() ? ROOT_PATH : null;
  }

  /**
   * Returns specified range of sub-components in path joined together.
   *
   * @see java.nio.file.Path#subpath(int, int)
   */
  public Webpath subpath(int beginIndex, int endIndex) {
    if (path.isEmpty() && beginIndex == 0 && endIndex == 1) {
      return this;
    }
    checkArgument(beginIndex >= 0 && endIndex > beginIndex);
    List<String> subList;
    try {
      subList = getParts().subList(beginIndex, endIndex);
    } catch (IndexOutOfBoundsException e) {
      throw new IllegalArgumentException();
    }
    return new Webpath(JOINER.join(subList));
  }

  /**
   * Returns number of components in {@code path}.
   *
   * @see java.nio.file.Path#getNameCount()
   */
  public int getNameCount() {
    if (path.isEmpty()) {
      return 1;
    } else if (isRoot()) {
      return 0;
    } else {
      return getParts().size();
    }
  }

  /**
   * Returns component in {@code path} at {@code index}.
   *
   * @see java.nio.file.Path#getName(int)
   */
  public Webpath getName(int index) {
    if (path.isEmpty()) {
      checkArgument(index == 0);
      return this;
    }
    try {
      return new Webpath(getParts().get(index));
    } catch (IndexOutOfBoundsException e) {
      throw new IllegalArgumentException();
    }
  }

  /**
   * Returns path without extra separators or {@code .} and {@code ..}, preserving trailing slash.
   *
   * @see java.nio.file.Path#normalize()
   */
  public Webpath normalize() {
    List<String> parts = new ArrayList<>();
    boolean mutated = false;
    int resultLength = 0;
    int mark = 0;
    int index;
    do {
      index = path.indexOf(SEPARATOR, mark);
      String part = path.substring(mark, index == -1 ? path.length() : index + 1);
      switch (part) {
        case CURRENT_DIR:
        case CURRENT_DIR + SEPARATOR:
          if (!parts.isEmpty()) {
            if (parts.get(parts.size() - 1).equals(CURRENT_DIR + SEPARATOR)) {
              resultLength -= parts.remove(parts.size() - 1).length();
            }
            mutated = true;
            break;
          }
          // fallthrough
        case PARENT_DIR:
        case PARENT_DIR + SEPARATOR:
          if (!parts.isEmpty()) {
            if (parts.size() == 1 && parts.get(0).equals(ROOT)) {
              mutated = true;
              break;
            }
            String last = parts.get(parts.size() - 1);
            if (last.equals(CURRENT_DIR + SEPARATOR)) {
              resultLength -= parts.remove(parts.size() - 1).length();
              mutated = true;
            } else if (!last.equals(PARENT_DIR + SEPARATOR)) {
              resultLength -= parts.remove(parts.size() - 1).length();
              mutated = true;
              break;
            }
          }
          // fallthrough
        default:
          if (index != mark || index == 0) {
            parts.add(part);
            resultLength = part.length();
          } else {
            mutated = true;
          }
      }
      mark = index + 1;
    } while (index != -1);
    if (!mutated) {
      return this;
    }
    StringBuilder result = new StringBuilder(resultLength);
    for (String part : parts) {
      result.append(part);
    }
    return new Webpath(result.toString());
  }

  /**
   * Returns {@code other} appended to {@code path}.
   *
   * @see java.nio.file.Path#resolve(java.nio.file.Path)
   */
  public Webpath resolve(Webpath other) {
    if (other.path.isEmpty()) {
      return this;
    } else if (isEmpty() || other.isAbsolute()) {
      return other;
    } else if (hasTrailingSeparator()) {
      return new Webpath(path + other.path);
    } else {
      return new Webpath(path + SEPARATOR + other.path);
    }
  }

  /**
   * Returns {@code other} resolved against parent of {@code path}.
   *
   * @see java.nio.file.Path#resolveSibling(java.nio.file.Path)
   */
  public Webpath resolveSibling(Webpath other) {
    checkNotNull(other);
    Webpath parent = getParent();
    return parent == null ? other : parent.resolve(other);
  }

  /** Returns absolute path of {@code reference} relative to {@code file}. */
  public Webpath lookup(Webpath reference) {
    return getParent().resolve(reference).normalize();
  }

  /**
   * Returns {@code other} made relative to {@code path}.
   *
   * @see java.nio.file.Path#relativize(java.nio.file.Path)
   */
  public Webpath relativize(Webpath other) {
    checkArgument(isAbsolute() == other.isAbsolute(), "'other' is different type of Path");
    if (path.isEmpty()) {
      return other;
    }
    PeekingIterator<String> left = Iterators.peekingIterator(split());
    PeekingIterator<String> right = Iterators.peekingIterator(other.split());
    while (left.hasNext() && right.hasNext()) {
      if (!left.peek().equals(right.peek())) {
        break;
      }
      left.next();
      right.next();
    }
    StringBuilder result = new StringBuilder(path.length() + other.path.length());
    while (left.hasNext()) {
      result.append(PARENT_DIR);
      result.append(SEPARATOR);
      left.next();
    }
    while (right.hasNext()) {
      result.append(right.next());
      result.append(SEPARATOR);
    }
    if (result.length() > 0 && !other.hasTrailingSeparator()) {
      result.deleteCharAt(result.length() - 1);
    }
    return new Webpath(result.toString());
  }

  /**
   * Returns {@code true} if {@code path} starts with {@code other}.
   *
   * @see java.nio.file.Path#startsWith(java.nio.file.Path)
   */
  public boolean startsWith(Webpath other) {
    Webpath me = removeTrailingSeparator();
    other = other.removeTrailingSeparator();
    if (other.path.length() > me.path.length()) {
      return false;
    } else if (me.isAbsolute() != other.isAbsolute()) {
      return false;
    } else if (!me.path.isEmpty() && other.path.isEmpty()) {
      return false;
    }
    return startsWith(split(), other.split());
  }

  private static boolean startsWith(Iterator<String> lefts, Iterator<String> rights) {
    while (rights.hasNext()) {
      if (!lefts.hasNext() || !rights.next().equals(lefts.next())) {
        return false;
      }
    }
    return true;
  }

  /**
   * Returns {@code true} if {@code path} ends with {@code other}.
   *
   * @see java.nio.file.Path#endsWith(java.nio.file.Path)
   */
  public boolean endsWith(Webpath other) {
    Webpath me = removeTrailingSeparator();
    other = other.removeTrailingSeparator();
    if (other.path.length() > me.path.length()) {
      return false;
    } else if (!me.path.isEmpty() && other.path.isEmpty()) {
      return false;
    } else if (other.isAbsolute()) {
      return me.isAbsolute() && me.path.equals(other.path);
    }
    return startsWith(me.splitReverse(), other.splitReverse());
  }

  /** Converts relative path to an absolute path. */
  public Webpath toAbsolutePath(Webpath currentWorkingDirectory) {
    checkArgument(currentWorkingDirectory.isAbsolute());
    return isAbsolute() ? this : currentWorkingDirectory.resolve(this);
  }

  /** Returns {@code toAbsolutePath(ROOT_PATH)}. */
  public Webpath toAbsolutePath() {
    return toAbsolutePath(ROOT_PATH);
  }

  /** Removes beginning separator from path, if an absolute path. */
  public Webpath removeBeginningSeparator() {
    return isAbsolute() ? new Webpath(path.substring(1)) : this;
  }

  /** Adds trailing separator to path, if it isn't present. */
  public Webpath addTrailingSeparator() {
    return hasTrailingSeparator() ? this : new Webpath(path + SEPARATOR);
  }

  /** Removes trailing separator from path, unless it's root. */
  public Webpath removeTrailingSeparator() {
    if (!isRoot() && hasTrailingSeparator()) {
      return new Webpath(path.substring(0, path.length() - 1));
    } else {
      return this;
    }
  }

  /** Splits path into components, excluding separators and empty strings. */
  public Iterator<String> split() {
    return getParts().iterator();
  }

  /** Splits path into components in reverse, excluding separators and empty strings. */
  public Iterator<String> splitReverse() {
    return Lists.reverse(getParts()).iterator();
  }

  /**
   * Compares two paths lexicographically for ordering.
   *
   * @see java.nio.file.Path#compareTo(java.nio.file.Path)
   */
  @Override
  public int compareTo(Webpath other) {
    if (isEmpty()) {
      if (!other.isEmpty()) {
        return -1;
      }
    } else {
      if (other.isEmpty()) {
        return 1;
      }
    }
    if (isAbsolute()) {
      if (!other.isAbsolute()) {
        return 1;
      }
    } else {
      if (other.isAbsolute()) {
        return -1;
      }
    }
    int result = ORDERING.compare(getParts(), other.getParts());
    if (result == 0) {
      if (hasTrailingSeparator()) {
        if (!other.hasTrailingSeparator()) {
          return 1;
        }
      } else {
        if (other.hasTrailingSeparator()) {
          return -1;
        }
      }
    }
    return result;
  }

  /** Returns {@code true} if equal, without taking duplicate slashes into consideration. */
  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }
    if (!(other instanceof Webpath) || hashCode() != other.hashCode()) {
      return false;
    }
    String path2 = ((Webpath) other).path;
    int i = 0;
    int i2 = 0;
    while (true) {
      if (i == path.length()) {
        return i2 == path2.length();
      }
      if (i2 == path2.length()) {
        return false;
      }
      char c = path.charAt(i++);
      if (c == SEPARATOR) {
        while (i < path.length() && path.charAt(i) == SEPARATOR) {
          i++;
        }
      }
      char c2 = path2.charAt(i2++);
      if (c != c2) {
        return false;
      }
      if (c2 == SEPARATOR) {
        while (i2 < path2.length() && path2.charAt(i2) == SEPARATOR) {
          i2++;
        }
      }
    }
  }

  /** Returns hash code, without taking duplicate slashes into consideration. */
  @Override
  public int hashCode() {
    int h = lazyHashCode;
    if (h == 0) {
      boolean previousWasSlash = false;
      for (int i = 0; i < path.length(); i++) {
        char c = path.charAt(i);
        if (c != SEPARATOR || !previousWasSlash) {
          h = 31 * h + (c & 0xffff);
        }
        previousWasSlash = c == SEPARATOR;
      }
      lazyHashCode = h;
    }
    return h;
  }

  /** Returns path as a string. */
  @Override
  public String toString() {
    return path;
  }

  @Override
  public int length() {
    return path.length();
  }

  @Override
  public char charAt(int index) {
    return path.charAt(index);
  }

  @Override
  public CharSequence subSequence(int start, int end) {
    return path.subSequence(start, end);
  }

  /** Returns {@code true} if this path is an empty string. */
  public boolean isEmpty() {
    return path.isEmpty();
  }

  /** Returns list of path components, excluding slashes. */
  private List<String> getParts() {
    List<String> result = lazyStringParts;
    return result != null
        ? result
        : (lazyStringParts =
            path.isEmpty() || isRoot()
                ? Collections.<String>emptyList()
                : SPLITTER.splitToList(path));
  }
}
