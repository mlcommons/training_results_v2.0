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

package io.bazel.rules.closure.webfiles.compiler;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.css.SourceCode;
import com.google.common.css.compiler.ast.CssBlockNode;
import com.google.common.css.compiler.ast.CssRootNode;
import com.google.common.css.compiler.ast.CssTree;
import com.google.common.css.compiler.ast.GssParserCC;
import com.google.common.css.compiler.ast.GssParserCCConstants;
import com.google.common.css.compiler.ast.GssParserException;
import com.google.common.css.compiler.ast.StringCharStream;
import com.google.common.css.compiler.ast.Token;
import java.util.Set;
import javax.inject.Inject;

/**
 * Wrapper parser class around {@link GssParserCC} that does not abort due to syntax errors.
 *
 * <p>The parse method retries parsing individual erroneous blocks with some workarounds, failing
 * which it ignores the erroneous block and continues parsing the rest of the CSS. Workarounds have
 * been implemented for some known cases that GssParser does not support. Note, this intends to
 * preserve as much CSS as possible when presented with erroneous stylesheets and does NOT support
 * spec compliant error handling/recovery.
 */
public final class CssParser {

  public static final String CSS_SYNTAX_ERROR = "cssSyntax";

  private static final ImmutableMap<Integer, Integer> BRACE_MAP =
      ImmutableMap.of(
          GssParserCCConstants.LEFTBRACE, GssParserCCConstants.RIGHTBRACE,
          GssParserCCConstants.LEFTROUND, GssParserCCConstants.RIGHTROUND,
          GssParserCCConstants.LEFTSQUARE, GssParserCCConstants.RIGHTSQUARE);

  /** Returns {@code true} if {@code path} is a CSS file. */
  static boolean isCssFile(Object path) {
    String strPath = path.toString();
    return strPath.endsWith(".css") || strPath.endsWith(".gss");
  }

  @Inject
  public CssParser() {}

  /** Parses stylesheet. */
  public CssTree parse(String path, String content) {
    SourceCode source = new SourceCode(path, content);
    StringCharStream stream = new StringCharStream(source.getFileContents());
    CssBlockNode globalBlock = new CssBlockNode(false /* isEnclosedWithBraces */);
    CssTree tree = new CssTree(source, new CssRootNode(globalBlock));
    GssParserCC parser =
        new GssParserCC(stream, globalBlock, source, false /* enableErrorRecovery */);
    int endOfLastError = 0;
    while (true) {
      try {
        parser.parse();
        break;
      } catch (GssParserException e) {
        // TODO(jart): Re-enable when csscomp supports --foo syntax.
        //errorReporter.report(CSS_SYNTAX_ERROR, e.getGssError().format());
        // Generic strategy to retry parsing by getting back to top level
        // NOTE: +1 for end-index because a location is represented as a closed range
        int begin =
            globalBlock.isEmpty()
                ? endOfLastError
                : Math.max(
                    endOfLastError,
                    globalBlock.getLastChild().getSourceCodeLocation().getEndCharacterIndex() + 1);
        endOfLastError = skipCurrentStatement(parser, stream, begin);
      }
    }
    return tree;
  }

  /**
   * Skips current statement (a ruleset or an at-rule) and gets back to the "top level" so that it
   * can continue to parse. See <a href="http://www.w3.org/TR/css-syntax-3/#error-handling">W3C CSS3
   * Error Handling</a>.
   */
  private static int skipCurrentStatement(GssParserCC parser, StringCharStream stream, int begin) {
    // Back up to the beginning of the current statement
    // NOTE: StringCharStream points 0-based index of the last read char, that's like (-1)-based
    stream.backup(stream.getCharIndex() - begin + 1);
    // Clear prefetched token
    parser.token.next = null;
    // Skip until the end of current statement
    Token t = parser.getToken(1);
    // At the "top level" of a stylesheet, an <at-keyword-token> starts an at-rule. Anything else
    // starts a qualified rule.
    if (t.kind == GssParserCCConstants.ATKEYWORD
        || t.kind == GssParserCCConstants.ATLIST
        || t.kind == GssParserCCConstants.ATRULESWITHDECLBLOCK) {
      // An at-rule ends with a semicolon or a block
      t = skipUntil(parser, GssParserCCConstants.LEFTBRACE, GssParserCCConstants.SEMICOLON);
      if (t.kind == GssParserCCConstants.LEFTBRACE) {
        // An opening curly-brace starts the at-rule’s body. The at-rule seeks forward, matching
        // blocks (content surrounded by (), {}, or []) until it finds a closing curly-brace that
        // isn’t matched by anything else or inside of another block.
        t = skipUntil(parser, GssParserCCConstants.RIGHTBRACE);
      } else {
        // A semicolon ends the at-rule immediately
      }
    } else {
      // A qualified ruleset ends with a block
      t = skipUntil(parser, GssParserCCConstants.LEFTBRACE);
      if (t.kind == GssParserCCConstants.LEFTBRACE) {
        t = skipUntil(parser, GssParserCCConstants.RIGHTBRACE);
      }
    }
    // NOTE: +1 for end-index because a location is represented as a closed range
    return stream.convertToCharacterIndex(t.endLine, t.endColumn) + 1;
  }

  /** Consumes tokens until any of specified tokens except ones within matched braces. */
  private static Token skipUntil(GssParserCC parser, Integer... kinds) {
    Set<Integer> kindset =
        ImmutableSet.<Integer>builder().add(kinds).add(GssParserCCConstants.EOF).build();
    while (true) {
      Token t = parser.getNextToken();
      if (kindset.contains(t.kind)) {
        return t;
      } else if (BRACE_MAP.containsKey(t.kind)) {
        t = skipUntil(parser, BRACE_MAP.get(t.kind));
        if (t.kind == GssParserCCConstants.EOF) {
          return t;
        }
      }
    }
  }
}
