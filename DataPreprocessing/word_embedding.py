# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 23:40:46 2020

@author: vamsi
"""
import warnings
warnings.filterwarnings('ignore')
class switch(object):
    value = None
    def __new__(class_, value):
        class_.value = value
        return True

def case(*args):
    return any((arg == switch.value for arg in args))

def word_embedding(pswitch, corpus):
    while switch(pswitch):
        if case('CountVectorizer'):
            from sklearn.feature_extraction.text import CountVectorizer
            vectorizer = CountVectorizer()
            X = vectorizer.fit_transform(corpus)
            X.toarray()
            return X
            break
        if case('word2vec'):
            from gensim.models import word2vec
            # we need to pass splitted sentences to the model
            tokenized_sentences = [sentence.split() for sentence in corpus]
            model = word2vec.Word2Vec(tokenized_sentences, min_count=1)
            return model
            break
        if case('GloVe'):
            import itertools
            from gensim.models.word2vec import Text8Corpus
            from glove import Corpus, Glove
            # sentences and corpus from standard library
            sentences = list(itertools.islice(Text8Corpus('text8'),None))
            corpus = Corpus()
            # fitting the corpus with sentences and creating Glove object
            corpus.fit(sentences, window=10)
            glove = Glove(no_components=100, learning_rate=0.05)
            # fitting to the corpus and adding standard dictionary to the object
            glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
            glove.add_dictionary(corpus.dictionary)
            return glove
            break
        

if __name__ == '__main__':
    corpus = [
          'Text of first document.',
          'Text of the second document made longer.',
          'Number three.',
          'This is number four.',]
    d = word_embedding('GloVe',corpus)
    print(d)
    