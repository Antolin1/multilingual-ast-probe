import unittest
from transformers import BertTokenizer

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class TestAlign(unittest.TestCase):
    def test_align_wordpiece(self):
        self.assertEqual(True, True)  # todo

    def test_align_blbpe(self):
        self.assertEqual(True, True)  # todo


if __name__ == '__main__':
    unittest.main()
