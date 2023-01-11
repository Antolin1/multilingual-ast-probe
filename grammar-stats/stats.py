import argparse
import os

import json

import pandas as pd

GRAMMARS_INFO = {
    'tree-sitter-c': 'C',
    'tree-sitter-c-sharp': 'C#',
    'tree-sitter-go': 'Go',
    'tree-sitter-java': 'Java',
    'tree-sitter-javascript': 'Javascript',
    'tree-sitter-php': 'PHP',
    'tree-sitter-python': 'Python',
    'tree-sitter-ruby': 'Ruby'
}

GRAMMARS = GRAMMARS_INFO.keys()


def extract_tokens(grammar_nodes) -> list:
    return [n['type'] for n in grammar_nodes if not n['named']]


def read_grammar_nodes(grammar_nodes_file: object) -> list:
    with open(grammar_nodes_file, "r") as f:
        grammar_nodes = json.load(f)
        return extract_tokens(grammar_nodes)


def main(args):
    nodes = {}
    for g in args.grammars:
        node_types_file = os.path.join(args.folder, g, 'src', 'node-types.json')
        nodes[g] = set(read_grammar_nodes(node_types_file))

    rows = []
    for g1 in args.grammars:
        g1_data = {}
        for g2 in args.grammars:
            common = nodes[g1] & nodes[g2]
            # metric = 100.0 * len(common) / len(nodes[g1])
            metric = 100.0 * len(common) / len(nodes[g1] | nodes[g2])
            g1_data[g2] = metric
        rows.append(g1_data)

    df = pd.DataFrame(rows, columns=args.grammars, index=args.grammars)
    df = df.rename(GRAMMARS_INFO).rename(GRAMMARS_INFO, axis='columns')

    print(df.round(2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for analysing grammars')
    parser.add_argument('--folder', help='grammar folder.', default='grammars')
    parser.add_argument('--grammars', help='grammar names.', nargs='+', default=GRAMMARS)

    parsed_args = parser.parse_args()
    main(parsed_args)
