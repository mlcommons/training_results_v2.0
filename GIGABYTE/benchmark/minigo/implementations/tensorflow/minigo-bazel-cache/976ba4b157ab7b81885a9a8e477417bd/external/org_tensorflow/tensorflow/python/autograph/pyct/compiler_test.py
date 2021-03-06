# coding=utf-8
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for compiler module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import textwrap

import gast

from tensorflow.python.autograph.pyct import compiler
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.platform import test
from tensorflow.python.util import tf_inspect


class CompilerTest(test.TestCase):

  def test_parser_compile_identity(self):

    def test_fn(x):
      a = True
      b = ''
      if a:
        b = x + 1
      return b

    node, _ = parser.parse_entity(test_fn, future_features=())
    module, _, _ = compiler.ast_to_object(node)

    self.assertEqual(
        textwrap.dedent(tf_inspect.getsource(test_fn)),
        tf_inspect.getsource(module.test_fn))

  def test_ast_to_source(self):
    node = gast.If(
        test=gast.Num(1),
        body=[
            gast.Assign(
                targets=[gast.Name('a', gast.Store(), None)],
                value=gast.Name('b', gast.Load(), None))
        ],
        orelse=[
            gast.Assign(
                targets=[gast.Name('a', gast.Store(), None)],
                value=gast.Str('c'))
        ])

    source = compiler.ast_to_source(node, indentation='  ')
    self.assertEqual(
        textwrap.dedent("""
            # coding=utf-8
            if 1:
              a = b
            else:
              a = 'c'
        """).strip(), source.strip())

  def test_ast_to_object(self):
    node = gast.FunctionDef(
        name='f',
        args=gast.arguments(
            args=[gast.Name('a', gast.Param(), None)],
            vararg=None,
            kwonlyargs=[],
            kwarg=None,
            defaults=[],
            kw_defaults=[]),
        body=[
            gast.Return(
                gast.BinOp(
                    op=gast.Add(),
                    left=gast.Name('a', gast.Load(), None),
                    right=gast.Num(1)))
        ],
        decorator_list=[],
        returns=None)

    module, source, _ = compiler.ast_to_object(node)

    expected_source = """
      # coding=utf-8
      def f(a):
        return a + 1
    """
    self.assertEqual(
        textwrap.dedent(expected_source).strip(),
        source.strip())
    self.assertEqual(2, module.f(1))
    with open(module.__file__, 'r') as temp_output:
      self.assertEqual(
          textwrap.dedent(expected_source).strip(),
          temp_output.read().strip())

  def test_source_to_entity(self):
    test_source = textwrap.dedent(u"""
      # coding=utf-8
      def f(a):
        '????????? ??????? ??? ????????????? + ???Q(s???, a???)(r??? + ??????????? max Q(???))'
        return a + 1
    """)
    module, _ = compiler.source_to_entity(test_source, delete_on_exit=True)
    self.assertEqual(module.f(1), 2)
    self.assertEqual(
        module.f.__doc__, '????????? ??????? ??? ????????????? + ???Q(s???, a???)(r??? + ??????????? max Q(???))')


if __name__ == '__main__':
  test.main()
