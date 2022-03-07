import unittest

from tree_sitter import Language, Parser
from src.data.code2ast import code2ast
import networkx as nx
import matplotlib.pyplot as plt

PY_LANGUAGE = Language('grammars/languages.so', 'python')
parser = Parser()
parser.set_language(PY_LANGUAGE)

class RunningExample(unittest.TestCase):
    def test_running_example(self):
        code = """
        for e in l:
            if e > 0:
                c+=1
            else:
                break"""
        G, _ = code2ast(code, parser)
        plt.figure()
        plt.title('test_code2ast')
        nx.draw(nx.Graph(G), labels=nx.get_node_attributes(G, 'type'), with_labels=True)
        plt.show()


if __name__ == '__main__':
    unittest.main()
