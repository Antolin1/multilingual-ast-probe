#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 08:34:53 2022

@author: Jose Antonio
"""

import unittest
from tree_sitter import Language, Parser
from src.data.code2ast import (code2ast, enrichAstWithDeps, 
                          getDependencyTree, getMatrixAndTokens,
                          getTreeFromDistances, getUAS, getSpear)
from src.data.utils import (remove_comments_and_docstrings_python,
                            remove_comments_and_docstrings_java_js)
import networkx as nx
import matplotlib.pyplot as plt


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
        enrichAstWithDeps(G)
        T = getDependencyTree(G)
        nx.draw(T, labels=nx.get_node_attributes(T,'type'), with_labels = True)
        plt.show()
    
    def test_distanceToks(self):
        G, pre_code = code2ast(code, parser)
        enrichAstWithDeps(G)
        T = getDependencyTree(G)
        matrix, code_toks = getMatrixAndTokens(T,pre_code)
        print(matrix)
        print(code_toks)
        self.assertEqual(len(code_toks), matrix.shape[0])
        first_row = [0,1,1,2,2,2,2,1,1,2,3,3,2,2,3,2,3]
        self.assertEqual(first_row, list(matrix[0]))
        
    def test_inverse(self):
        G, pre_code = code2ast(code, parser)
        enrichAstWithDeps(G)
        T = getDependencyTree(G)
        matrix, code_toks = getMatrixAndTokens(T,pre_code)
        T2 = getTreeFromDistances(matrix, code_toks)
        nx.draw(T2, labels=nx.get_node_attributes(T2,'type'), with_labels = True)
        plt.show()
    
    def test_Eval(self):
        G, pre_code = code2ast(code, parser)
        enrichAstWithDeps(G)
        T = getDependencyTree(G)
        matrix, code_toks = getMatrixAndTokens(T,pre_code)
        T2 = getTreeFromDistances(matrix, code_toks)
        
        T_pred = nx.Graph(T2)
        T_pred.remove_edge(8,15)
        T_pred.add_edge(15,14)
        self.assertAlmostEqual(getUAS(T2,T_pred), 
                               float(len(T_pred.edges)-1)/float(len(T_pred.edges)))
        print(getSpear(matrix,matrix))

    def test_js(self):
        parser = Parser()
        parser.set_language(JS_LANGUAGE)
        G,_ = code2ast(code_js, parser, 'javascript')
        nx.draw(G, labels=nx.get_node_attributes(G, 'type'), with_labels=True)
        plt.show()
        enrichAstWithDeps(G)
        T = getDependencyTree(G)
        nx.draw(T, labels=nx.get_node_attributes(T,'type'), with_labels = True)
        plt.show()
        