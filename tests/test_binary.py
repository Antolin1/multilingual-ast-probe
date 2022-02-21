import unittest
from tree_sitter import Language, Parser
from src.data.code2ast import code2ast, get_tokens_ast
from src.data.binary_tree import ast2binary, tree_to_distance, distance_to_tree, remove_empty_nodes, extend_complex_nodes
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout


code = """'''Compute the maximum'''
def max(a,b):
    #compare a and b
    if a > b:
        return a
    return b
"""

PY_LANGUAGE = Language('grammars/languages.so', 'python')
parser = Parser()
parser.set_language(PY_LANGUAGE)


class TestBinary(unittest.TestCase):
    def test_binary(self):
        G, pre_code = code2ast(code, parser)
        binary_ast = ast2binary(G)
        nx.draw(nx.Graph(binary_ast), labels=nx.get_node_attributes(binary_ast, 'type'), with_labels=True)
        plt.show()

        self.assertTrue(nx.is_tree(binary_ast))

        print([binary_ast.out_degree(n) for n in binary_ast])
        d, c, _ = tree_to_distance(binary_ast, 0)
        binary_ast_recov = distance_to_tree(d, c, get_tokens_ast(G, pre_code))
        self.assertTrue(nx.is_tree(binary_ast_recov))

        print(len(binary_ast_recov))
        print(len(binary_ast))
        print(d, c)
        nx.draw(nx.Graph(binary_ast_recov), labels=nx.get_node_attributes(binary_ast_recov, 'type'), with_labels=True)
        plt.show()

        binary_ast_recov_full = extend_complex_nodes(remove_empty_nodes(binary_ast_recov))
        nx.draw(nx.Graph(binary_ast_recov_full), labels=nx.get_node_attributes(binary_ast_recov_full, 'type'),
                with_labels=True)
        plt.show()
        #print(binary_ast.nodes(data=True))

if __name__ == '__main__':
    unittest.main()
