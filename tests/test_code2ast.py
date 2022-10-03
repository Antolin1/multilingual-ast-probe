import unittest

import matplotlib.pyplot as plt
import networkx as nx

from src.data.code2ast import (code2ast, get_tokens_ast)
from src.data.data_loading import PARSER_OBJECT_BY_NAME
from src.data.utils import (remove_comments_and_docstrings_python,
                            remove_comments_and_docstrings_java_js)


def node_match_type_atts(n1, n2):
    return n1['type'] == n2['type']


code = """'''Compute the maximum'''
def max(a,b):
    s = "string"
    #compare a and b
    if a > b:
        return a
    return b
"""

code_pre_expected = """def max(a,b):
    s = "string"
    if a > b:
        return a
    return b"""

code_js = """function myFunction(p1, p2) {
/* multi-line
comments */
s = "string"
return p1 * p2;// The function returns the product of p1 and p2
}"""
plt.show()
code_js_pre_expected = """function myFunction(p1, p2) {

s = "string"
return p1 * p2;
}"""

code_go = """// Function to add two numbers
func addTwoNumbers(x, y int) int {
/*
sdsd
sdsds
sdsdsd
*/
s = "str"
sum := x + y
return sum
}"""

code_php = """function writeMsg() {
  echo 'Hello world!';
}"""

code_ruby = """def initialize(n, a)
# this is a comment
@name = n
# this is another comment
@surname = "smith"
@age  = a * DOG_YEARS

#!/usr/bin/ruby -w
# This is a single line comment.

=begin
This is a multiline comment and con spwan as many lines as you
like.
=end
end"""

code_java = """public void myMethod() {
    String mystr = "mystr";
}"""

code_csharp = "public override string ToString(){return this.GetType().Name + \"" \
              "(compressionMode=\" + compressionMode + \", chunkSize=\" + chunkSize + \")\";}\n"


class Code2ast(unittest.TestCase):

    def test_code2ast_java(self):
        plt.figure()
        plt.title('test_code2ast_java')
        parser = PARSER_OBJECT_BY_NAME['java']
        G, pre_code = code2ast(code_java, parser, lang='java')
        print(pre_code)
        nx.draw(nx.Graph(G), labels=nx.get_node_attributes(G, 'type'), with_labels=True)
        plt.show()
        print(G.nodes(data=True))
        tokens = get_tokens_ast(G, pre_code)
        print(tokens)
        self.assertTrue('"mystr"' in tokens)

    def test_code2ast_csharp(self):
        plt.figure()
        plt.title('test_code2ast_csharp')
        parser = PARSER_OBJECT_BY_NAME['csharp']
        G, pre_code = code2ast(code_csharp, parser, lang='csharp')
        print(pre_code)
        nx.draw(nx.Graph(G), labels=nx.get_node_attributes(G, 'type'), with_labels=True)
        plt.show()
        print(G.nodes(data=True))
        tokens = get_tokens_ast(G, pre_code)
        print(tokens)
        self.assertTrue('"(compressionMode="' in tokens)

    def test_code2ast_php(self):
        plt.figure()
        plt.title('test_code2ast_php')
        parser = PARSER_OBJECT_BY_NAME['php']
        G, pre_code = code2ast(code_php, parser, lang='php')
        print(pre_code)
        nx.draw(nx.Graph(G), labels=nx.get_node_attributes(G, 'type'), with_labels=True)
        plt.show()
        print(G.nodes(data=True))
        tokens = get_tokens_ast(G, pre_code)
        print(tokens)
        self.assertTrue("'Hello world!'" in tokens)

    def test_code2ast_ruby(self):
        plt.figure()
        plt.title('test_code2ast_ruby')
        parser = PARSER_OBJECT_BY_NAME['ruby']
        G, pre_code = code2ast(code_ruby, parser, lang='ruby')
        print(pre_code)
        nx.draw(nx.Graph(G), labels=nx.get_node_attributes(G, 'type'), with_labels=True)
        plt.show()
        print(G.nodes(data=True))
        tokens = get_tokens_ast(G, pre_code)
        print(tokens)
        self.assertTrue('"smith"' in tokens)

    def test_code2ast_go(self):
        plt.figure()
        plt.title('test_code2ast_go')
        parser = PARSER_OBJECT_BY_NAME['go']
        G, pre_code = code2ast(code_go, parser, lang='go')
        print(pre_code)
        nx.draw(nx.Graph(G), labels=nx.get_node_attributes(G, 'type'), with_labels=True)
        plt.show()
        print(G.nodes(data=True))
        tokens = get_tokens_ast(G, pre_code)
        print(tokens)
        self.assertTrue('"str"' in tokens)

    def test_preprocessing(self):
        code_pre = remove_comments_and_docstrings_python(code)
        self.assertEqual(code_pre_expected, code_pre)
        code_pre = remove_comments_and_docstrings_java_js(code_js)
        self.assertEqual(code_js_pre_expected, code_pre)

    def test_code2ast_python(self):
        plt.figure()
        plt.title('test_code2ast_python')
        parser = PARSER_OBJECT_BY_NAME['python']
        G, pre_code = code2ast(code, parser)
        nx.draw(nx.Graph(G), labels=nx.get_node_attributes(G, 'type'), with_labels=True)
        plt.show()
        self.assertEqual(31, len(G))
        tokens = get_tokens_ast(G, pre_code)
        print(tokens)
        self.assertTrue('"string"' in tokens)

    def test_js(self):
        plt.figure()
        plt.title('test_js I')
        parser = PARSER_OBJECT_BY_NAME['javascript']
        G, pre_code = code2ast(code_js, parser, 'javascript')
        print(pre_code)
        nx.draw(G, labels=nx.get_node_attributes(G, 'type'), with_labels=True)
        plt.show()
        tokens = get_tokens_ast(G, pre_code)
        print(tokens)
        self.assertTrue('"string"' in tokens)

    def test_str_ast(self):
        code = """def split_phylogeny(p, level="s"):
    level = level+"__"
    result = p.split(level)
    return result[0]+level+result[1].split(";")[0]"""
        parser = PARSER_OBJECT_BY_NAME['python']
        G, pre_code = code2ast(code, parser)
        tokens = get_tokens_ast(G, pre_code)
        self.assertTrue('"__"' in tokens)
        self.assertTrue('";"' in tokens)
        self.assertTrue('"s"' in tokens)
        plt.figure()
        plt.title('test_str_ast')
        nx.draw(nx.Graph(G), labels=nx.get_node_attributes(G, 'type'), with_labels=True)
        plt.show()
