from src.data.code2ast import get_id, get_root_ast
import networkx as nx
import numpy as np

def ast2binary(G):
    #fussion non-terminals with on non-terminal child
    def ast2binary_aux(current_node_G, G, new_G, parent_in_new_G):
        out_edges = list(G.out_edges(current_node_G))
        if len(out_edges) == 2:
            for _, m in out_edges:
                id_m_new = get_id(new_G)
                new_G.add_node(id_m_new, **G.nodes[m])
                new_G.add_edge(parent_in_new_G, id_m_new)
                ast2binary_aux(m, G, new_G, id_m_new)
        elif len(out_edges) == 1:
            m = out_edges[0][1]
            if not G.nodes[m]['is_terminal']:
                new_G.nodes[parent_in_new_G]['type'] = new_G.nodes[parent_in_new_G]['type'] + '-' + G.nodes[m]['type']
                ast2binary_aux(m, G, new_G, parent_in_new_G)
            else:
                #todo: check this, unary things
                new_G.nodes[parent_in_new_G]['is_terminal'] = True
        elif len(out_edges) > 2:
            out_nodes = [m for _, m in out_edges]
            out_nodes.sort(key=lambda m: G.nodes[m]['start'])
            id_m_new = get_id(new_G)
            new_G.add_node(id_m_new, **G.nodes[out_nodes[0]])
            new_G.add_edge(parent_in_new_G, id_m_new)
            ast2binary_aux(out_nodes[0], G, new_G, id_m_new)
            new_empty_id = get_id(new_G)
            new_G.add_node(new_empty_id, type='<empty>')
            new_G.add_edge(parent_in_new_G, new_empty_id)
            for j, out_node in enumerate(out_nodes[1:]):
                if len(list(new_G.out_edges(new_empty_id))) == 1 and len(out_nodes[1:]) - j > 1:
                    new_empty_id_new = get_id(new_G)
                    new_G.add_node(new_empty_id_new, type='<empty>')
                    new_G.add_edge(new_empty_id, new_empty_id_new)
                    new_empty_id = new_empty_id_new
                id_m_new = get_id(new_G)
                new_G.add_node(id_m_new, **G.nodes[out_node])
                new_G.add_edge(new_empty_id, id_m_new)
                ast2binary_aux(out_node, G, new_G, id_m_new)
    new_G = nx.DiGraph()
    root_G = get_root_ast(G)
    new_G.add_node(0, **G.nodes[root_G])
    ast2binary_aux(root_G, G, new_G, 0)
    return new_G

def tree_to_distance(tree, node):
    if tree.out_degree(node) == 0:
        d = []
        c = []
        h = 0
    else:
        #todo: ->sort left to right
        left_child = list(tree.out_edges(node))[0][1]
        right_child = list(tree.out_edges(node))[1][1]
        d_l, c_l, h_l = tree_to_distance(tree, left_child)
        d_r, c_r, h_r = tree_to_distance(tree, right_child)
        h = max(h_r, h_l) + 1
        d = d_l + [h] + d_r
        c = c_l + [tree.nodes[node]['type']] + c_r
    return d, c, h

def distance_to_tree(d, c):
    def distance_to_tree_aux(G, d, c, father):
        if d == []:
            new_id = get_id(G)
            G.add_node(new_id)
            G.add_edge(father, new_id)
        else:
            i = np.argmax(d)
            new_id = get_id(G)
            G.add_node(new_id, type=c[i])
            if father != None:
                G.add_edge(father, new_id)
            distance_to_tree_aux(G, d[0:i], c[0:i], new_id)
            distance_to_tree_aux(G, d[i+1:], c[i+1:], new_id)
    G = nx.DiGraph()
    distance_to_tree_aux(G, d, c, None)
    return G
