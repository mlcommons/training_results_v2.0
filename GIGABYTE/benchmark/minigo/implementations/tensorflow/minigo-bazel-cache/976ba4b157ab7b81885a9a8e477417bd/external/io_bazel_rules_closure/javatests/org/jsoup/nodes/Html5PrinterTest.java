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

package org.jsoup.nodes;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import org.jsoup.Jsoup;
import org.jsoup.parser.Parser;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link Html5Printer}. */
@RunWith(JUnit4.class)
public final class Html5PrinterTest {

  @Test
  public void simpleDocument_printsLikeHtml5() throws Exception {
    assertThat(Html5Printer.stringify(parse("hello"))).isEqualTo("<!doctype html>hello");
  }

  @Test
  public void paragraph_omitsClosingTag() throws Exception {
    assertThat(Html5Printer.stringify(parse("<p>hello</p>"))).isEqualTo("<!doctype html><p>hello");
  }

  @Test
  public void preservesNecessaryWhitespace() throws Exception {
    assertThat(Html5Printer.stringify(parse(" <p> hello</p>")))
        .isEqualTo("<!doctype html><p> hello");
  }

  @Test
  public void htmlHeadBody_withoutAttributes_omitsThem() throws Exception {
    assertThat(
            Html5Printer.stringify(
                parse("<html><head><link rel=doodle></head><body>hello</body></html>")))
        .isEqualTo("<!doctype html><link rel=\"doodle\">hello");
  }

  @Test
  public void htmlWithLang_preservesOpeningHtmlTag() throws Exception {
    assertThat(
            Html5Printer.stringify(
                parse("<html lang=en><head><link rel=doodle></head><body>hello</body></html>")))
        .isEqualTo("<!doctype html><html lang=\"en\"><link rel=\"doodle\">hello");
  }

  @Test
  public void table_omitsTagsWeDontWant() throws Exception {
    assertThat(
            Html5Printer.stringify(
                parse(
                    "<!doctype html>"
                        + "<table>"
                        + "<tbody>"
                        + "<tr>"
                        + "<td>hi</td>"
                        + "</tr>"
                        + "</tbody>"
                        + "</table>")))
        .isEqualTo("<!doctype html><table><tr><td>hi</table>");
  }

  private static Document parse(String html) throws IOException {
    Parser parser = Parser.htmlParser();
    Document doc = Jsoup.parse(new ByteArrayInputStream(html.getBytes(UTF_8)), null, "", parser);
    doc.outputSettings().indentAmount(0);
    doc.outputSettings().prettyPrint(false);
    return doc;
  }
}
