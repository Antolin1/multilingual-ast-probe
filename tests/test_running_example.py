import unittest

import matplotlib.pyplot as plt
import networkx as nx

from src.data.binary_tree import ast2binary, tree_to_distance
from src.data.code2ast import code2ast
from src.data.data_loading import PARSER_OBJECT_BY_NAME

parser = PARSER_OBJECT_BY_NAME['python']


class RunningExample(unittest.TestCase):
    def test_running_example(self):
        code = """
        for element in l:
            if element > 0:
                c+=1
            else:
                selected = element
                break"""
        G, _ = code2ast(code, parser)
        plt.figure()
        plt.title('test_code2ast')
        nx.draw(nx.Graph(G), labels=nx.get_node_attributes(G, 'type'), with_labels=True)
        plt.show()

        binary_ast = ast2binary(G)
        nx.draw(nx.Graph(binary_ast), labels=nx.get_node_attributes(binary_ast, 'type'), with_labels=True)
        plt.show()

        d, c, _, u = tree_to_distance(binary_ast, 0)
        print(d)
        print(c)
        print(u)


if __name__ == '__main__':
    unittest.main()
