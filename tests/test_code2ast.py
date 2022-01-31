#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 08:34:53 2022

@author: Jose Antonio
"""

import unittest
from tree_sitter import Language, Parser
from src.data.code2ast import (code2ast, enrichAstWithDeps, 
                          getDependencyTree, getMatrixAndTokens)
from src.data.preprocessing import remove_comments_and_docstrings_python
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

PY_LANGUAGE = Language('grammars/languages.so', 'python')
parser = Parser()
parser.set_language(PY_LANGUAGE)

class Code2ast(unittest.TestCase):
    
    def test_preprocessing(self):
        code_pre = remove_comments_and_docstrings_python(code)
        self.assertEqual(code_pre_expected, code_pre)
    
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