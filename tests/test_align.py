import unittest
from transformers import BertTokenizer, RobertaTokenizer

from src.data.code2ast import code2ast, get_tokens_ast
from src.data import PARSER_OBJECT_BY_NAME
from src.data.utils import match_tokenized_to_untokenized_bert, match_tokenized_to_untokenized_roberta

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

code = """'''Compute the maximum'''
def max(a,b):
    #compare a and b
    if a > b:
        return asdfrsdfsdf
    return b
"""
parser = PARSER_OBJECT_BY_NAME['python']
G, pre_code = code2ast(code, parser)
code_tokens = get_tokens_ast(G, pre_code)


class TestAlign(unittest.TestCase):
    def test_align_wordpiece(self):
        print(code_tokens)
        t, mapping = match_tokenized_to_untokenized_bert(untokenized_sent=code_tokens, tokenizer=bert_tokenizer)
        print(t)
        print(mapping)
        self.assertEqual(True, True)  # todo

    def test_align_blbpe(self):
        print(code_tokens)
        t, mapping = match_tokenized_to_untokenized_roberta(untokenized_sent=code_tokens, tokenizer=roberta_tokenizer)
        print(t)
        print(mapping)
        self.assertEqual(True, True)  # todo


if __name__ == '__main__':
    unittest.main()
