#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spelling Corrector for GloVe Applications (Python).

Contains functions to check and match GloVe spellings for:
  - a word,
  - a sentence,
  - a list (array) of words or sentences.

Implements an adapted version of Peter Norvig's spelling corrector
(http://norvig.com/spell-correct.html) to accomodate for most words
in GloVe 400K (https://nlp.stanford.edu/projects/glove).
It includes the option to "delete" the word if not present in the GloVe based
dictionaty. Excludes any words (strings) that contain non-alphanumeric 
characters.

Files used:
  `spelling_dict.csv`: contains an English dictionary with "frequencies" 
  assigned using a Wikipedia Corpus and The Adventures of Sherlock Holmes.

If `spelling_dict.csv` is not in the directory, a dictionary will be built using:
  `glove.6B.50d.txt` from:
  GloVe: Global Vectors for Word Representation
  https://nlp.stanford.edu/projects/glove
and,
  `Doyle_Sherlock_H.txt` from:
  The Adventures of Sherlock Holmes by Arthur Conan Doyle
  http://www.gutenberg.org/ebooks/1661.txt.utf-8, or https://norvig.com/big.txt

Author: Reynaldo Vazquez
This version April 3 2018, first Version March 25 2018,
"""
import numpy as np
import pandas as pd
import os
import os.path
import re
from collections import Counter

def create_spelling_dict():
  """
  Creates a dictionary to be used by the spelling corrector
  """
  # Extract GloVe dictionary
  glove_file_path = 'glove.6B.50d.txt'
  with open(glove_file_path, 'r') as f:
    glove_words = []
    for line in f:
      line = line.strip().split()
      curr_word = line[0]
      glove_words.append(curr_word)

  # Eliminate words that contain non-alnum characters.
  pattern = r'[\W_]'
  indices = [i for i, x in enumerate(glove_words) if re.search(pattern, x)]
  mask = np.ones(len(glove_words),dtype=bool)
  mask[indices] = False
  new_dict = np.array(glove_words)[mask]
  WORDS_GloVe = Counter(new_dict)

  # Original Peter Norvig's Word dictionary
  text = open('Doyle_Sherlock_H.txt').read()
  def words(text): return re.findall(r'\w+', text.lower())
  Sherlock_words = Counter(words(text))
  # Assign frequencies + 1 to GloVe dictionary (do not add new words)
  for word, value in Sherlock_words.items():
    if WORDS_GloVe[word] != 0:
      WORDS_GloVe[word] = value + 1
  WORDS = WORDS_GloVe
  return WORDS

def load_spelling_dictionary():
  """
  Loads an existing dictionary if it exisits, 
  creates a new one otherwise
  """
  spelling_fname = 'spelling_dict.csv'
  if os.path.isfile(spelling_fname):
    words_dict = pd.read_csv(spelling_fname, sep=',')
    WORDS_dict = dict(zip(words_dict["word"], words_dict["freq"]))
    WORDS = WORDS_dict
  else:
    WORDS = create_spelling_dict()
  return WORDS

WORDS = load_spelling_dictionary()
WORDS = Counter(WORDS)

# Modified spelling corrector
def P(word):
  """
  Probability of `word`.
  """
  N = sum(WORDS.values())
  return WORDS[word] / N

def correction(word, delete = True):
  """
  Most probable spelling correction for word."
  """
  original_word = word
  word = re.sub("[\W_]", '', word).lower()
  suggestion = candidates(word)
  if suggestion != None:
    suggestion = max(suggestion, key=P)
  elif delete == True:
    suggestion = ""
  else:
    suggestion = original_word
  return suggestion

def candidates(word):
  """
  Generate possible spelling corrections for word.
  """
  return (known([word]) or known(edits1(word)) or known(edits2(word)) or None)

def known(words):
  """
  The subset of `words` that appear in the dictionary of WORDS.
  """
  return set(w for w in words if w in WORDS)

def edits1(word):
  """
  All edits that are one edit away from `word`.
  """
  letters    = 'abcdefghijklmnopqrstuvwxyz'
  splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
  deletes    = [L + R[1:]               for L, R in splits if R]
  transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
  replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
  inserts    = [L + c + R               for L, R in splits for c in letters]
  return set(deletes + transposes + replaces + inserts)

def edits2(word):
  """
  All edits that are two edits away from `word`.
  """
  return (e2 for e1 in edits1(word) for e2 in edits1(e1))

def correct_phrase(phrase):
  """
  Corrects misspellings in a phrase. (deletes words not in dictionary by
  correction()'s default)
  """
  corrected_words = [correction(word) for word in phrase.strip().split()]
  corrected_phrase = ' '.join(map(str, corrected_words)).strip()
  corrected_phrase = re.sub(re.compile(r'\s+'), ' ', corrected_phrase)
  return corrected_phrase

def correct_list(list_of_phrases):
  """
  Corrects misspellings in a list (or array) of phrases. (deletes words not 
  in dictionary by correction()'s default)
  """
  return [correct_phrase(phrase) for phrase in list_of_phrases]