from __future__ import absolute_import, division, print_function

import os
import pickle

import tflearn
from tflearn.data_utils import *

import sys

path = sys.argv[1] # give a path to the input text document
char_idx_file = 'char_idx.pickle'

maxlen = 25
char_idx = None


if os.path.isfile(char_idx_file):
  print('Loading previous char_idx')
  char_idx = pickle.load(open(char_idx_file, 'rb'))

X, Y, char_idx = \
    textfile_to_semi_redundant_sequences(path, seq_maxlen=maxlen, redun_step=10)

pickle.dump(char_idx, open(char_idx_file,'wb'))

g = tflearn.input_data([None, maxlen, len(char_idx)])
g = tflearn.lstm(g, 128, return_seq=True)
g = tflearn.dropout(g, 0.5)
g = tflearn.lstm(g, 128, return_seq=True)
g = tflearn.dropout(g, 0.5)
g = tflearn.lstm(g, 128)
g = tflearn.dropout(g, 0.5)
g = tflearn.fully_connected(g, len(char_idx), activation='softmax') 
g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy',
                       learning_rate=0.01) # try some other learning rate

m = tflearn.SequenceGenerator(g, dictionary=char_idx,
                              seq_maxlen=maxlen,
                              clip_gradients=5.0,
                              checkpoint_path='model_word')

seed = random_sequence_from_textfile(path, maxlen)
m.fit(X, Y, validation_set=0.1, batch_size=528, # try different batch sizes
      n_epoch=5, run_id='word')

# Create fifteen sentences.
the_sentences_file = open('generated.txt', 'w') # the sentences will be saved in this file

for i in range(15):

  generate = m.generate(1000, temperature=1.0, seq_seed=seed) #some random sentences are generated here
  the_sentences_file.write("\r%s\n" % generate)
  print('new sentence: ' + str(i))
