#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 16:08:24 2022

@author: Jose Antonio
"""

from src.data.preprocessing import remove_comments_and_docstrings_python
import networkx as nx

#aux function
def getId(G):
    if len(G) == 0:
        return 0
    return max(list(G)) + 1

#aux function
def getGraphFromTree(node, G, id_father):
    #traverse children
    for child in node.children:
        is_terminal_child = (len(child.children) == 0)
        id_child = getId(G)
        G.add_node(id_child, type = child.type,
                      is_terminal = is_terminal_child,
                      start = child.start_byte,
                      end = child.end_byte)
        G.add_edge(id_father,id_child)
        getGraphFromTree(child, G, id_child)

#aux function
def getToken(code, start, end):
    return bytes(code, "utf8")[start:end].decode("utf-8") 

#preprocess code, obtain the ast and returns a network graph 
def code2ast(code, parser, lang='python'):
    if lang == 'python':
        #preprocess
        code = remove_comments_and_docstrings_python(code)
        tree = parser.parse(bytes(code, "utf8"))
        
        G = nx.DiGraph()
        #add root
        G.add_node(0, type = tree.root_node.type,
                      is_terminal = False,
                      start = tree.root_node.start_byte,
                      end = tree.root_node.end_byte)
        getGraphFromTree(tree.root_node, G, 0)
        return G

def selectHead(G, n):
    return

def enrichAstWithDeps(G):
    return 

#get distances