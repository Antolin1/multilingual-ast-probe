#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 16:05:32 2022

@author: Jose Antonio
"""

from tree_sitter import Language

Language.build_library(
  # Store the library in the `build` directory
  'grammars/languages.so',

  # Include one or more languages
  [
    'grammars/tree-sitter-python'
  ]
)
