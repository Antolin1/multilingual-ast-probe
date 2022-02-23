import unittest
from tree_sitter import Language, Parser
from src.data.code2ast import code2ast, get_tokens_ast
from src.data.binary_tree import ast2binary, tree_to_distance, \
    distance_to_tree, remove_empty_nodes, extend_complex_nodes, \
    get_multiset_ast, get_precision_recall_f1, add_unary
import networkx as nx
import matplotlib.pyplot as plt


code = """'''Compute the maximum'''
def max(a,b):
    #compare a and b
    if a > b:
        return
    return b
"""

PY_LANGUAGE = Language('grammars/languages.so', 'python')
parser = Parser()
parser.set_language(PY_LANGUAGE)


class TestBinary(unittest.TestCase):

    def test_random(self):
        d = [12.749075889587402, 12.086353302001953, 3.5092380046844482, 2.8773586750030518, 1.8203082084655762,
         1.105534553527832, 16.498552322387695, 14.007007598876953, 10.105134963989258, 8.575981140136719,
         9.284015655517578, 10.76449203491211, 9.185318946838379, 6.908447265625, 2.8699421882629395,
         1.5084803104400635, 3.9279606342315674, 1.9249317646026611, 1.1106932163238525, 5.392538070678711,
         1.335496187210083]
        c = ['module#function_definition', '<empty>', 'parameters', '<empty>', '<empty>', '<empty>', '<empty>', '<empty>',
         'block#for_statement', '<empty>', '<empty>', '<empty>', '<empty>', 'block#if_statement', 'comparison_operator',
         '<empty>', '<empty>', '<empty>', 'block#return_statement', '<empty>', 'return_statement']
        u = ['<empty>', '<empty>', '<empty>', '<empty>', '<empty>', '<empty>', '<empty>', '<empty>', '<empty>', '<empty>',
         '<empty>', '<empty>', '<empty>', '<empty>', '<empty>', '<empty>', '<empty>', '<empty>', '<empty>', '<empty>',
         '<empty>', '<empty>']

        binary_ast_recov = distance_to_tree(d, c, u, [str(i) for i in range(len(u))])

        nx.draw(nx.Graph(binary_ast_recov), labels=nx.get_node_attributes(binary_ast_recov, 'type'), with_labels=True)
        plt.show()

        binary_ast_recov_full =  extend_complex_nodes(add_unary(remove_empty_nodes(binary_ast_recov)))

        nx.draw(nx.Graph(binary_ast_recov_full), labels=nx.get_node_attributes(binary_ast_recov_full, 'type'),
                with_labels=True)
        plt.show()

    def test_binary(self):
        G, pre_code = code2ast(code, parser)
        binary_ast = ast2binary(G)
        nx.draw(nx.Graph(binary_ast), labels=nx.get_node_attributes(binary_ast, 'type'), with_labels=True)
        plt.show()

        self.assertTrue(nx.is_tree(binary_ast))

        print([binary_ast.out_degree(n) for n in binary_ast])
        d, c, _, u = tree_to_distance(binary_ast, 0)
        self.assertTrue(len(u), len(get_tokens_ast(G, pre_code)))
        print(u)
        binary_ast_recov = distance_to_tree(d, c, u, get_tokens_ast(G, pre_code))

        self.assertTrue(nx.is_tree(binary_ast_recov))
        self.assertEqual(len(binary_ast_recov), len(binary_ast))

        nx.draw(nx.Graph(binary_ast_recov), labels=nx.get_node_attributes(binary_ast_recov, 'type'), with_labels=True)
        plt.show()

        print(binary_ast_recov.nodes(data=True))
        binary_ast_recov_full = extend_complex_nodes(add_unary(remove_empty_nodes(binary_ast_recov)))
        nx.draw(nx.Graph(binary_ast_recov_full), labels=nx.get_node_attributes(binary_ast_recov_full, 'type'),
                with_labels=True)
        plt.show()

        print(get_precision_recall_f1(binary_ast_recov_full, binary_ast_recov_full))

        perturbed = nx.DiGraph(binary_ast_recov)
        for n in perturbed:
            if perturbed.nodes[n]['type'] == 'comparison_operator':
                perturbed.nodes[n]['type'] = 'binary_operator'
        perturbed = extend_complex_nodes(add_unary(remove_empty_nodes(perturbed)))
        print(get_precision_recall_f1(binary_ast_recov_full, perturbed))

        nx.draw(nx.Graph(perturbed), labels=nx.get_node_attributes(perturbed, 'type'),
                with_labels=True)
        plt.show()
        #print(binary_ast.nodes(data=True))

if __name__ == '__main__':
    unittest.main()
