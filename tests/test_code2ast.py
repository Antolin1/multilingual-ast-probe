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

def node_match_type_atts(n1,n2):
    return n1['type'] == n2['type']

code = """'''Compute the maximum'''
def max(a,b):
    #compare a and b
    if a > b:
        return a
    return b
"""

code_pre_expected = """def max(a,b):
    if a > b:
        return a
    return b"""

code_js = """function myFunction(p1, p2) {
/* multi-line
comments */
return p1 * p2;// The function returns the product of p1 and p2
}"""
plt.show()
code_js_pre_expected = """function myFunction(p1, p2) {

return p1 * p2;
}"""

PY_LANGUAGE = Language('grammars/languages.so', 'python')
JS_LANGUAGE = Language('grammars/languages.so', 'javascript')
parser = Parser()
parser.set_language(PY_LANGUAGE)

class Code2ast(unittest.TestCase):

    def test_preprocessing(self):
        code_pre = remove_comments_and_docstrings_python(code)
        self.assertEqual(code_pre_expected, code_pre)
        code_pre = remove_comments_and_docstrings_java_js(code_js)
        self.assertEqual(code_js_pre_expected, code_pre)

    def test_code2ast(self):
        plt.figure()
        plt.title('test_code2ast')
        G,_ = code2ast(code, parser)
        nx.draw(nx.Graph(G), labels=nx.get_node_attributes(G,'type'), with_labels=True)
        plt.show()
        self.assertEqual(26, len(G))
    
    def test_dependency(self):
        plt.figure()
        plt.title('test_dependency')
        G, code_pre = code2ast(code, parser)
        enrich_ast_with_deps(G)
        T = get_dependency_tree(G)
        nx.draw(nx.Graph(T), labels=nx.get_node_attributes(T,'type'), with_labels=True)
        plt.show()
        self.assertEqual(len(T), len(get_tokens_ast(G, code_pre)))
    
    def test_distance_tokens_dep(self):
        G, pre_code = code2ast(code, parser)
        enrich_ast_with_deps(G)
        T = get_dependency_tree(G)
        matrix, code_tokens = get_matrix_and_tokens_dep(T, pre_code)
        self.assertEqual(len(code_tokens), matrix.shape[0])
        first_row = [0, 1, 1, 2, 2, 2, 2, 1, 1, 2, 3, 3, 2, 2, 3, 2, 3]
        self.assertEqual(first_row, list(matrix[0]))
        
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
    
    def test_eval(self):
        G, pre_code = code2ast(code, parser)
        enrich_ast_with_deps(G)
        T = get_dependency_tree(G)
        matrix, code_toks = get_matrix_and_tokens_dep(T, pre_code)
        T2 = get_tree_from_distances(matrix, code_toks)
        T_pred = nx.Graph(T2)
        T_pred.remove_edge(8,15)
        T_pred.add_edge(15,14)
        self.assertAlmostEqual(get_uas(T2, T_pred),
                               float(len(T_pred.edges)-1) / float(len(T_pred.edges)))

    def test_js(self):
        plt.figure()
        plt.title('test_js I')
        parser = Parser()
        parser.set_language(JS_LANGUAGE)
        G, _ = code2ast(code_js, parser, 'javascript')
        nx.draw(G, labels=nx.get_node_attributes(G, 'type'), with_labels=True)
        plt.show()
        plt.figure()
        plt.title('test_js II')
        enrich_ast_with_deps(G)
        T = get_dependency_tree(G)
        nx.draw(T, labels=nx.get_node_attributes(T, 'type'), with_labels=True)
        plt.show()

    def test_distance_ast(self):
        G, pre_code = code2ast(code, parser)
        first_row = [0, 2, 3, 3, 3, 3, 3, 2, 4, 5, 5, 5, 4, 6, 6, 4, 4]
        depths = [2, 2, 3, 3, 3, 3, 3, 2, 4, 5, 5, 5, 4, 6, 6, 4, 4]
        self.assertEqual(first_row, list(get_matrix_tokens_ast(G, pre_code)[0][0]))
        self.assertEqual(depths, list(get_depths_tokens_ast(G, pre_code)[0]))

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