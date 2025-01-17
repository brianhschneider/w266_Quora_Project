import re
import time
import itertools
import numpy as np

# For pretty-printing
import pandas as pd
from IPython.display import display, HTML
import jinja2



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
            print (prev_num_edits)
        prev_num_edits = num_edits
    if verbose:
        print (prev_num_edits)
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




# for pretty printing
def flatten(list_of_lists):
    """Flatten a list-of-lists into a single list."""
    return list(itertools.chain.from_iterable(list_of_lists))

HIGHLIGHT_BUTTON_TMPL = jinja2.Template("""
<script>
colors_on = true;
function color_cells() {
  var ffunc = function(i,e) {return e.innerText {{ filter_cond }}; }
  var cells = $('table.dataframe').children('tbody')
                                  .children('tr')
                                  .children('td')
                                  .filter(ffunc);
  if (colors_on) {
    cells.css('background', 'white');
  } else {
    cells.css('background', '{{ highlight_color }}');
  }
  colors_on = !colors_on;
}
$( document ).ready(color_cells);
</script>
<form action="javascript:color_cells()">
<input type="submit" value="Toggle highlighting (val {{ filter_cond }})"></form>
""")

RESIZE_CELLS_TMPL = jinja2.Template("""
<script>
var df = $('table.dataframe');
var cells = df.children('tbody').children('tr')
                                .children('td');
cells.css("width", "{{ w }}px").css("height", "{{ h }}px");
</script>
""")

def render_matrix(M, rows=None, cols=None, dtype=float,
                        min_size=30, highlight=""):
    html = [pd.DataFrame(M, index=rows, columns=cols,
                         dtype=dtype)._repr_html_()]
    if min_size > 0:
        html.append(RESIZE_CELLS_TMPL.render(w=min_size, h=min_size))

    if highlight:
        html.append(HIGHLIGHT_BUTTON_TMPL.render(filter_cond=highlight,
                                             highlight_color="yellow"))

    return "\n".join(html)
    
def pretty_print_matrix(*args, **kwargs):
    """Pretty-print a matrix using Pandas.

    Optionally supports a highlight button, which is a very, very experimental
    piece of messy JavaScript. It seems to work for demonstration purposes.

    Args:
      M : 2D numpy array
      rows : list of row labels
      cols : list of column labels
      dtype : data type (float or int)
      min_size : minimum cell size, in pixels
      highlight (string): if non-empty, interpreted as a predicate on cell
      values, and will render a "Toggle highlighting" button.
    """
    html = render_matrix(*args, **kwargs)
    display(HTML(html))


def pretty_timedelta(fmt="%d:%02d:%02d", since=None, until=None):
    """Pretty-print a timedelta, using the given format string."""
    since = since or time.time()
    until = until or time.time()
    delta_s = until - since
    hours, remainder = divmod(delta_s, 3600)
    minutes, seconds = divmod(remainder, 60)
    return fmt % (hours, minutes, seconds)


##
# Word processing functions
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


##
# Data loading functions
import nltk
from shared_lib import vocabulary


#!!!!!!!!!!!!!!!!!!!!!!!!
## Once Working will go through and delete functions not being used

#!!!!!!!!!!!!!!!!!!!!!!

def get_corpus(name="brown"):
    return nltk.corpus.__getattr__(name)

def sents_to_tokens(sents, vocab):
    """Returns an flattened list of the words in the sentences, with normal padding."""
    padded_sentences = (["<s>"] + s + ["</s>"] for s in sents)
    # This will canonicalize words, and replace anything not in vocab with <unk>
    return np.array([canonicalize_word(w, wordset=vocab.wordset)
                     for w in flatten(padded_sentences)], dtype=object)

#def build_vocab(corpus, V=10000):
#    token_feed = (canonicalize_word(w) for w in corpus.words())
#    vocab = vocabulary.Vocabulary(token_feed, size=V)
#    return vocab

def build_vocab(questions, V=10000):
    #token_feed = (canonicalize_word(w) for w in sentence.split() for sentence in questions)
    #token_feed = (canonicalize_word(w) for sentence in questions for w in sentence.split())
    #print (token_feed)
    token_feed = []
    for sentence in questions:
        for w in sentence.split():
            token_feed.append(canonicalize_word(w))
    token_feed = set(token_feed)
    
    vocab = vocabulary.Vocabulary(token_feed, size=V)
    
    return vocab
    

def get_train_test_sents(corpus, split=0.8, shuffle=True):
    """Get train and test sentences.

    Args:
      corpus: nltk.corpus that supports sents() function
      split (double): fraction to use as training set
      shuffle (int or bool): seed for shuffle of input data, or False to just
      take the training data as the first xx% contiguously.

    Returns:
      train_sentences, test_sentences ( list(list(string)) ): the train and test
      splits
    """
    sentences = np.array(corpus.sents(), dtype=object)
    fmt = (len(sentences), sum(map(len, sentences)))
    print ("Loaded %d sentences (%g tokens)" % fmt)

    if shuffle:
        rng = np.random.RandomState(shuffle)
        rng.shuffle(sentences)  # in-place
    train_frac = 0.8
    split_idx = int(train_frac * len(sentences))
    train_sentences = sentences[:split_idx]
    test_sentences = sentences[split_idx:]

    fmt = (len(train_sentences), sum(map(len, train_sentences)))
    print ("Training set: %d sentences (%d tokens)" % fmt)
    fmt = (len(test_sentences), sum(map(len, test_sentences)))
    print ("Test set: %d sentences (%d tokens)" % fmt)

    return train_sentences, test_sentences

def preprocess_sentences(sentences, vocab):
    """Preprocess sentences by canonicalizing and mapping to ids.

    Args:
      sentences ( list(list(string)) ): input sentences
      vocab: Vocabulary object, already initialized

    Returns:
      ids ( array(int) ): flattened array of sentences, including boundary <s>
      tokens.
    """
    # Add sentence boundaries, canonicalize, and handle unknowns
    #words = flatten(["<s>"] + s + ["</s>"] for s in sentences)
    words = ["<s>" + s + "</s>" for s in sentences]
    # print (words[0:5])
    words = [canonicalize_word(w, wordset=vocab.word_to_id) for w in words]

     
    return np.array(vocab.words_to_ids(words))

##
# Use this function
def load_corpus(name, split=0.8, V=10000, shuffle=0):
    """Load a named corpus and split train/test along sentences."""
    corpus = get_corpus(name)
    vocab = build_vocab(corpus, V)
    train_sentences, test_sentences = get_train_test_sents(corpus, split, shuffle)
    train_ids = preprocess_sentences(train_sentences, vocab)
    test_ids = preprocess_sentences(test_sentences, vocab)
    return vocab, train_ids, test_ids

##
def batch_generator(q1_sequences, q2_sequences, batch_size, max_time):
    """Convert ids to data-matrix form."""
    # Clip to multiple of max_time for convenience
    # print ("len(q1_sequences): ", len(q1_sequences))
    clip_len_q1 = int(((len(q1_sequences)-1) / batch_size) * batch_size)
    # print ("clip_len_q1: ", clip_len_q1)
    input_w_q1 = q1_sequences[:clip_len_q1]     # current word
    
    clip_len_q2 = int(((len(q2_sequences)-1) / batch_size) * batch_size)
    input_w_q2 = q2_sequences[:clip_len_q2]     # current word
    
    
    # what is the proper size...should it come togeth
    target_y = q1_sequences[1:clip_len_q1+1] + q2_sequences[1:clip_len_q2+1]  # next word
    
    # Reshape so we can select columns
    input_w_q1 = input_w_q1.reshape([batch_size,-1])
    input_w_q2 = input_w_q2.reshape([batch_size,-1])
    target_y = target_y.reshape([batch_size,-1])

    # Yield batches
    for i in xrange(0, input_w_q1.shape[1], max_time):
        yield input_w_q1[:,i:i+max_time], input_w_q2[:,i:i+max_time], target_y[:,i:i+max_time]
