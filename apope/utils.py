import re
import time
import itertools
import numpy as np



# find difference b/t 2 strings
def levenshtein_explicit(str1, str2, verbose=False):
    prev_num_edits = range(len(str1) + 1)
    for j in xrange(1, len(str2) + 1):
        num_edits = [prev_num_edits[0] + 1]
        for i in xrange(1, len(str1) + 1):
            # 1 if letters differ (substitution is free if the letters are the same)
            substitution = 0 if str1[i - 1] == str2[j - 1] else 1
            result = min([num_edits[i - 1] + 1,
                          prev_num_edits[i] + 1,
                          prev_num_edits[i - 1] + substitution
            ])
            num_edits.append(result)
        if verbose:
            print prev_num_edits
        prev_num_edits = num_edits
    if verbose:
        print prev_num_edits
    return prev_num_edits[len(str1)]



# Word processing functions
def flatten(list_of_lists):
    """Flatten a list-of-lists into a single list."""
    return list(itertools.chain.from_iterable(list_of_lists))

def canonicalize_digits(word):
    if any([c.isalpha() for c in word]): return word
    word = re.sub("\d", "DG", word)
    if word.startswith("DG"):
        word = word.replace(",", "") # remove thousands separator
    return word

def canonicalize_word(word, wordset=None, digits=True):
    word = word.lower()
    if digits:
        if (wordset != None) and (word in wordset): return word
        word = canonicalize_digits(word) # try to canonicalize numbers
    if (wordset == None) or (word in wordset): return word
    else: return "<unk>" # unknown token

def canonicalize_words(words, **kw):
    return [canonicalize_word(word, **kw) for word in words]

