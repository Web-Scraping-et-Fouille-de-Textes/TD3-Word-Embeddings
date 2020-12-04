import re
import string
import gensim.models

def preprocess(s):
	s_without_punct = s
	for char in string.punctuation:
		s_without_punct = s_without_punct.replace(char, ' ') # on replace les ponctuations par des espaces
	tokens = s_without_punct.split()
	return tokens


def get_corpus(corpus_path):
	# assume there's one sentence per line, tokens separated by whitespace
	# simple_preprocess: "Convert a document into a list of lowercase tokens, 
	# ignoring tokens that are too short or too long.""
	with open(corpus_path, mode='r', encoding='utf-8') as inFile:
		return [preprocess(line) for line in inFile.readlines()]


class Embeddings(object):
	"""Programs to learn, save and load embeddings."""
	def __init__(self, corpus_path, model_path):
		self._corpus_path = corpus_path
		self._model_path = model_path

	def learn(self):
		sentences = get_corpus(self._corpus_path) # List of sentences (a sentence = list of words)
		self._model = gensim.models.Word2Vec(sentences=sentences)
		self._model.save(self._model_path)

	def learn_restrictive(self, min_freq):
		sentences = get_corpus(self._corpus_path) # List of sentences (a sentence = list of words)
		self._model = gensim.models.Word2Vec(sentences=sentences, size=100, window=20, min_count=min_freq, workers=4)
		self._model.save(self._model_path)

	def load(self):
		self._model = gensim.models.Word2Vec.load(self._model_path)

	def most_similar(self, word, topN=10):
		return self._model.wv.most_similar(positive=[word], topn=topN)

	def most_similar_analogy(self, pos, neg, topN=10):
		return self._model.wv.most_similar(positive=pos, negative=neg, topn=topN)

	def similarity(self, w1, w2):
		if w1 in self._model.wv and w2 in self._model.wv:
			return self._model.wv.similarity(w1, w2)
		else:
			return -1

	def get_vocab(self):
		return self._model.wv.vocab

	def get_vector(self, word):
		return self._model.wv.get_vector(word)

	def get_model(self):
		return self._model
		