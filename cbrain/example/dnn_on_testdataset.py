
import nltk
import os
import csv
import itertools
import numpy as np
import matplotlib.pyplot as plt

from ..network.neuron import Layer, Network
from ..dataset.dataset import DataGenerator
from ..function import Logsig, Tansig, Softmax


def DNN_copy():
    """see https://au.mathworks.com/help/nnet/ug/
            how-dynamic-neural-networks-work.html
    """

    class ZeroOneDataGenerator(DataGenerator):
        def iget(self):
            while True:
                yield np.array([np.random.randint(2)])

    def example3(data):
        l3 = Layer(1, B=np.zeros((1,)), activate_function=Logsig())
        net2 = Network((l3, ))
        net2.load(1, 1, W=np.array([[1]]), D=0)
        net2.connect(1, 1, W=np.array([[0.5]]), D=1)

        return net2.forward(data)

    l1 = Layer(1, Logsig())
    net = Network((l1,))
    net.connect(1, 1, D=1)

    dg = ZeroOneDataGenerator()

    data = []
    tdata = []
    for _ in range(10000):
        _data = dg.get(0, 5)
        _tdata = example3(_data)
        data.append(_data)
        tdata.append(_tdata)

    net.train(data, tdata)

    eval_data = dg.get(0, 10)
    eval_ret = net.forward(eval_data)
    eval_tdata = example3(eval_data)

    plt.plot(range(len(eval_data)), eval_tdata, "b")
    plt.plot(range(len(eval_data)), eval_ret, "r")
    plt.show()


def DNN_lm():
    """
        http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial
        -part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/
    """

    vocabulary_size = 26
    unknown_token = "UNKNOWN_TOKEN"
    sentence_start_token = "SENTENCE_START"
    sentence_end_token = "SENTENCE_END"
    path = os.path.dirname(os.path.realpath(__file__))

    #print("Reading CSV file...")
    with open(path + '/data/reddit-comments-2015-08.csv', 'r') as f:
        reader = csv.reader(f, skipinitialspace=True)
        next(reader)
        # Split full comments into sentences
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower())
                                      for x in reader])
        # Append SENTENCE_START and SENTENCE_END
        sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token)
                     for x in sentences]
    #print("Parsed %d sentences." % (len(sentences)))

    # Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    #print("Found %d unique words tokens." % len(word_freq.items()))

    # Get the most common words and build index_to_word
    # and word_to_index vectors
    vocab = word_freq.most_common(vocabulary_size-1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    #print("Using vocabulary size %d." % vocabulary_size)
    #print("The least frequent word in our vocabulary is '%s' \
    #      and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token
                                  for w in sent]

    print("\nExample sentence: '%s'" % sentences[0])
    print("\nExample sentence after Pre-processing: '%s'"
          % tokenized_sentences[0])

    # Create the training data
    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]]
                          for sent in tokenized_sentences])
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]]
                          for sent in tokenized_sentences])

    #print(X_train)
    #print(y_train)

    def gen_input(x):
        sentence = x
        d = np.zeros((len(sentence), vocabulary_size))
        d[np.arange(len(sentence)), sentence] = 1
        return d

    l1 = Layer(100, Tansig())
    l2 = Layer(vocabulary_size, Softmax())
    #l2 = Layer(vocabulary_size, Logsig())

    net = Network((l1, l2))
    net.connect(1,2, D=0)
    net.connect(1,1, D=1)

    net.load(vocabulary_size, 1, D=0)

    sentence10 = gen_input(X_train[10])
    o = net.forward(sentence10)
    print(o)

    training_input = [gen_input(x) for x in X_train[:2]]
    training_output = [gen_input(x) for x in y_train[:2]]

    net.train(training_input, training_output)
    o = net.forward(sentence10)
    print(o)
