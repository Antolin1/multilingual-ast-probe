import unittest
from tree_sitter import Language, Parser
from src.data.code2ast import (code2ast, enrich_ast_with_deps,
                               get_dependency_tree, get_matrix_and_tokens_dep,
                               get_tree_from_distances, get_uas,
                               remplace_non_terminals,
                               remove_useless_non_terminals,
                               get_matrix_tokens_ast, get_depths_tokens_ast, get_tokens_ast)
from src.data.utils import (remove_comments_and_docstrings_python,
                            remove_comments_and_docstrings_java_js)
import networkx as nx
import matplotlib.pyplot as plt


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

code_csharp = "public override string ToString(){return this.GetType().Name + \"(compressionMode=\" + compressionMode + \", chunkSize=\" + chunkSize + \")\";}\n"


PY_LANGUAGE = Language('grammars/languages.so', 'python')
JS_LANGUAGE = Language('grammars/languages.so', 'javascript')
GO_LANGUAGE = Language('grammars/languages.so', 'go')
PHP_LANGUAGE = Language('grammars/languages.so', 'php')
RUBY_LANGUAGE = Language('grammars/languages.so', 'ruby')
JAVA_LANGUAGE = Language('grammars/languages.so', 'java')
CSHARP_LANGUAGE = Language('grammars/languages.so', 'c_sharp')
C_LANGUAGE = Language('grammars/languages.so', 'c')
parser = Parser()
parser.set_language(PY_LANGUAGE)


class Code2ast(unittest.TestCase):

    def test_code2ast_java(self):
        plt.figure()
        plt.title('test_code2ast_java')
        parser = Parser()
        parser.set_language(JAVA_LANGUAGE)
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
        parser = Parser()
        parser.set_language(CSHARP_LANGUAGE)
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
        parser = Parser()
        parser.set_language(PHP_LANGUAGE)
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
        parser = Parser()
        parser.set_language(RUBY_LANGUAGE)
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
        parser = Parser()
        parser.set_language(GO_LANGUAGE)
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
        G, pre_code = code2ast(code, parser)
        nx.draw(nx.Graph(G), labels=nx.get_node_attributes(G, 'type'), with_labels=True)
        plt.show()
        self.assertEqual(31, len(G))
        tokens = get_tokens_ast(G, pre_code)
        print(tokens)
        self.assertTrue('"string"' in tokens)

    def test_dependency(self):
        plt.figure()
        plt.title('test_dependency')
        G, code_pre = code2ast(code, parser)
        enrich_ast_with_deps(G)
        T = get_dependency_tree(G)
        nx.draw(nx.Graph(T), labels=nx.get_node_attributes(T, 'type'), with_labels=True)
        plt.show()
        self.assertEqual(len(T), len(get_tokens_ast(G, code_pre)))

    def test_inverse(self):
        plt.figure()
        plt.title('test_inverse')
        G, pre_code = code2ast(code, parser)
        enrich_ast_with_deps(G)
        T = get_dependency_tree(G)
        matrix, code_tokens = get_matrix_and_tokens_dep(T, pre_code)
        T2 = get_tree_from_distances(matrix, code_tokens)
        nx.draw(T2, labels=nx.get_node_attributes(T2, 'type'), with_labels=True)
        plt.show()
        self.assertEqual(len(T), len(T2))
        self.assertTrue(nx.is_isomorphic(T2, nx.Graph(T)))

    def test_js(self):
        plt.figure()
        plt.title('test_js I')
        parser = Parser()
        parser.set_language(JS_LANGUAGE)
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
        G, pre_code = code2ast(code, parser)
        tokens = get_tokens_ast(G, pre_code)
        self.assertTrue('"__"' in tokens)
        self.assertTrue('";"' in tokens)
        self.assertTrue('"s"' in tokens)
        plt.figure()
        plt.title('test_str_ast')
        nx.draw(nx.Graph(G), labels=nx.get_node_attributes(G, 'type'), with_labels=True)
        plt.show()

    def test_replace_functions_and_labels(self):
        code = """def my_func(a,b):
            if a > b:
                c = 0 + 1
                return a + b - c
            return b/a"""

        def comparison_operator_head(G, n):
            l = ['<', '>', '==', '<>',
                 '!=', '<=', '>=', 'in', 'is']
            nodes = [m for _, m in G.out_edges(n) if G.nodes[m]['type'] in l]
            return nodes[0]

        def binary_operator_head(G, n):
            l = ['+', '-', '/', '*',
                 '**', '@', '%', '//',
                 '<<', '>>', '^', '&', '|']
            nodes = [m for _, m in G.out_edges(n) if G.nodes[m]['type'] in l]
            return nodes[0]

        conf = {'comparison_operator': comparison_operator_head,
                'binary_operator': binary_operator_head}
        G, _ = code2ast(code, parser)
        g = remplace_non_terminals(remove_useless_non_terminals(G), conf)
        plt.figure()
        plt.title('test_replace_functions_and_labels I')
        nx.draw(nx.Graph(G), labels=nx.get_node_attributes(G, 'type'), with_labels=True)
        plt.show()
        plt.figure()
        plt.title('test_replace_functions_and_labels II')
        to_plot = nx.Graph(g)
        pos = nx.spring_layout(to_plot)
        nx.draw(to_plot, pos, labels=nx.get_node_attributes(to_plot, 'type'), with_labels=True)
        nx.draw_networkx_edge_labels(to_plot, pos, edge_labels=nx.get_edge_attributes(to_plot, 'label'))
        plt.show()
