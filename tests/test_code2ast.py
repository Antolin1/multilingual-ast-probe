#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 08:34:53 2022

@author: Jose Antonio
"""

import unittest
from tree_sitter import Language, Parser
from src.data.code2ast import (code2ast, enrich_ast_with_deps,
                               get_dependency_tree, get_matrix_and_tokens_dep,
                               get_tree_from_distances, get_uas, get_spear,
                               label_dep_tree, from_label_dep_tree_to_ast,
                               remove_useless_non_terminals, get_tuples_from_labeled_dep_tree,
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
        G,_ = code2ast(code, parser)
        nx.draw(G, labels=nx.get_node_attributes(G,'type'), with_labels = True)
        plt.show()
    
    def test_dependency(self):
        G,_ = code2ast(code, parser)
        enrich_ast_with_deps(G)
        T = get_dependency_tree(G)
        nx.draw(T, labels=nx.get_node_attributes(T,'type'), with_labels = True)
        plt.show()
    
    def test_distanceToks(self):
        G, pre_code = code2ast(code, parser)
        enrich_ast_with_deps(G)
        T = get_dependency_tree(G)
        matrix, code_toks = get_matrix_and_tokens_dep(T, pre_code)
        print(matrix)
        print(code_toks)
        self.assertEqual(len(code_toks), matrix.shape[0])
        first_row = [0,1,1,2,2,2,2,1,1,2,3,3,2,2,3,2,3]
        self.assertEqual(first_row, list(matrix[0]))
        
    def test_inverse(self):
        G, pre_code = code2ast(code, parser)
        enrich_ast_with_deps(G)
        T = get_dependency_tree(G)
        matrix, code_toks = get_matrix_and_tokens_dep(T, pre_code)
        T2 = get_tree_from_distances(matrix, code_toks)
        nx.draw(T2, labels=nx.get_node_attributes(T2,'type'), with_labels = True)
        plt.show()
    
    def test_Eval(self):
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
        print(get_spear(matrix, matrix))

    def test_js(self):
        parser = Parser()
        parser.set_language(JS_LANGUAGE)
        G,_ = code2ast(code_js, parser, 'javascript')
        nx.draw(G, labels=nx.get_node_attributes(G, 'type'), with_labels=True)
        plt.show()
        enrich_ast_with_deps(G)
        T = get_dependency_tree(G)
        nx.draw(T, labels=nx.get_node_attributes(T,'type'), with_labels = True)
        plt.show()

    def test_labelEdges(self):
        G, pre_code = code2ast(code, parser)
        G_not_enr = nx.DiGraph(G)
        enrich_ast_with_deps(G)
        T = get_dependency_tree(G)
        label_dep_tree(G_not_enr, T)
        T_ast = from_label_dep_tree_to_ast(T)
        #print(list(T.edges(data=True)))
        print(list((*edge, d['complex_edge_str']) for *edge, d in T.edges(data=True)))
        print('-'*100)
        self.assertEqual(len(T_ast), len(G))
        self.assertEqual(len(T_ast.edges), len(G_not_enr.edges))

        print([T_ast.nodes[n]['type'] for n in T_ast if not T_ast.nodes[n]['is_terminal']])
        print([G.nodes[n]['type'] for n in G if not G.nodes[n]['is_terminal']])
        print([T_ast.nodes[n]['type'] for n in T_ast if T_ast.nodes[n]['is_terminal']])
        print([G.nodes[n]['type'] for n in G if G.nodes[n]['is_terminal']])
        self.assertTrue(nx.is_isomorphic(T_ast, nx.Graph(G_not_enr), node_match_type_atts))
        print(get_tuples_from_labeled_dep_tree(T, pre_code))
        nx.draw(T_ast, labels=nx.get_node_attributes(T_ast, 'type'), with_labels=True)
        plt.show()

    def test_distane_ast(self):
        G, pre_code = code2ast(code, parser)
        print(get_matrix_tokens_ast(G, pre_code))
        print(get_depths_tokens_ast(G, pre_code))

    def test_str_ast(self):
        code = """def split_phylogeny(p, level="s"):
    level = level+"__"
    result = p.split(level)
    return result[0]+level+result[1].split(";")[0]"""
        G, pre_code = code2ast(code, parser)
        for n in G:
            print(G.nodes[n])
        g = remove_useless_non_terminals(G)
        print('------')
        for n in g:
            print(g.nodes[n])
        self.assertTrue(nx.is_connected(nx.Graph(g)))
        print(get_tokens_ast(G, pre_code))

        print(get_matrix_tokens_ast(G, pre_code))
        print(get_depths_tokens_ast(G, pre_code))

