#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 16:08:24 2022

@author: Jose Antonio
"""

from utils import remove_comments_and_docstrings_python
import networkx as nx

#aux function, get a new id in the graph
def getId(G):
    if len(G) == 0:
        return 0
    return max(list(G)) + 1

#aux function used to get the graph associated to the ast
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

#get token given the code, the start byte and the end byte
def getToken(code, start, end):
    return bytes(code, "utf8")[start:end].decode("utf-8") 


#get the possible candiate as head given a nonterminal level
def getCandidate(G, level, start):
    #reachable nodes
    nodes = list(nx.single_source_shortest_path_length(G, level).keys())
    #filter non-terminals
    nodes = [n for n in nodes if G.nodes[n]['is_terminal']]
    #filter right ones
    nodes = [n for n in nodes if G.nodes[n]['start'] < start]
    if len(nodes) == 0:
        return None
    #sort by start
    nodes.sort(key=lambda n: G.nodes[n]['start'])
    return nodes[0]

#get the head of a given non-terminal that is not the root
def selectHead(G, n):
    father = list(G.in_edges(n))[0][0]
    while(True):
        cand = getCandidate(G, father, G.nodes[n]['start'])
        if cand != None:
            return cand
        father = list(G.in_edges(father))[0][0]

#get root dependency tree
def getRoot(G):
    nodes = list(G.nodes)
    nodes = [n for n in nodes if G.nodes[n]['is_terminal']]
    nodes.sort(key=lambda n: G.nodes[n]['start'])
    return nodes[0]

#get the tokens of the dependency tree
def getTokens(T, code):
    return [getToken(code, T.nodes[t]['start'], T.nodes[t]['end']) for t in sorted(list(T.nodes), 
                                                 key = lambda n: T.nodes[n]['start'])]


#preprocess code, obtain the ast and returns a network graph.
#it returns the graph of the ast and the preprocessed code
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
        return G, code

#it adds dependency labels between non-terminals to the previous obtained ast graph
def enrichAstWithDeps(G):
    root = getRoot(G)
    nodes = [n for n in list(G.nodes) if root!=n and G.nodes[n]['is_terminal']]
    for n in nodes:
        h = selectHead(G, n)
        G.add_edge(h, n, label = 'dependency')
        
#obtains the dependecency subgraph from the enriched one
def getDependencyTree(G):
    view = nx.subgraph_view(G, filter_node=lambda n: G.nodes[n]['is_terminal'],
                            filter_edge=lambda n1, n2: G[n1][n2].get("label", 'dependency'))
    return nx.DiGraph(view)

#obtains the distance matrix from the dependency graph
def getMatrixAndTokens(T, code):
    distance = nx.floyd_warshall_numpy(nx.Graph(T), sorted(list(T.nodes), 
                                                 key = lambda n: T.nodes[n]['start']))
    tokens = getTokens(T, code)
    return distance, tokens