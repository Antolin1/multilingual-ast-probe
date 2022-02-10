import unittest
from tree_sitter import Language, Parser
from src.data.code2ast import (code2ast, enrichAstWithDeps,
                               getDependencyTree, remove_useless_non_terminals,
                               remplace_non_terminals)
from src.data.utils import (remove_comments_and_docstrings_python,
                            remove_comments_and_docstrings_java_js)
import networkx as nx
import matplotlib.pyplot as plt



PY_LANGUAGE = Language('grammars/languages.so', 'python')

parser = Parser()
parser.set_language(PY_LANGUAGE)

def comparison_operator_head(G, n):
    l = ['<','>','==','!=', '<=', '>=']
    nodes = [m for _, m in G.out_edges(n) if G.nodes[m]['type'] in l]
    return nodes[0]

def binary_operator_head(G, n):
    l = ['+','-','/','*', '**']
    nodes = [m for _, m in G.out_edges(n) if G.nodes[m]['type'] in l]
    return nodes[0]

conf = {'comparison_operator': comparison_operator_head,
        'binary_operator': binary_operator_head}

class Code2ast(unittest.TestCase):

    def test_dependency(self):
        code = """def max(a,b):
    if a == b:
        return a
    else:
        return a+b+c/2**2
    return b"""

        G, _ = code2ast(code, parser)
        g = remplace_non_terminals(remove_useless_non_terminals(G), conf)
        G_ast = nx.Graph(G)
        enrichAstWithDeps(G)
        T = getDependencyTree(G)
        nx.draw(nx.Graph(g), labels=nx.get_node_attributes(g, 'type'), with_labels=True)
        plt.show()
        plt.savefig("figure.png")

