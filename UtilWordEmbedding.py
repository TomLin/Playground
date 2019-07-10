"""
Reference for the following codes:
1. Multi-Class Text Classification with Doc2Vec & Logistic Regression
(https://towardsdatascience.com/multi-class-text-classification-with-doc2vec-logistic-regression-9da9947b43f4)

2. Gensim Doc2Vec Tutorial on the IMDB Sentiment Dataset
(https://github.com/RaRe-Technologies/gensim/blob/3c3506d51a2caf6b890de3b1b32a8b85f7566ca5/docs/notebooks/doc2vec-IMDB.ipynb)

3. smart-open
(https://www.open-open.com/lib/view/open1422349535814.html)

4. What r'string' or u'string' stands for?
(https://stackoverflow.com/questions/2081640/what-exactly-do-u-and-r-string-flags-do-and-what-are-raw-string-literals)

5. The regular expression of \1
(https://stackoverflow.com/questions/20802056/python-regular-expression-1)

6. Class of MeanEmbeddingVectorizer and TF-IDF-Weighted-MeanEmbedding
(http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/)


"""




import numpy as np
import re
import glob
from smart_open import smart_open
import os
import gensim
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn import utils
from collections import namedtuple, defaultdict
import logging
from sklearn.feature_extraction.text import TfidfVectorizer


# Convert text to lower-case and strip punctuation/symbols from words
def normalize_text(text):
	"""
	Ref: https://stackoverflow.com/questions/20802056/python-regular-expression-1

	:param text: string
	:return:
		clean string
	"""
	norm_text = text.lower()
	# Replace breaks with spaces
	norm_text = norm_text.replace('<br />', ' ')
	# Pad punctuation with spaces on both sides
	norm_text = re.sub(r"([\.\",\(\)!\?;:])", r" \1 ", norm_text)
	return norm_text


def concat_files(dirname, folders):
	"""
	Concatenate text from files to one file, and return a file list.

	:param dirname: string of directory
	:param folders: list of folder names
	:return
		files: list of file paths
	"""
	files = []

	for fol in folders:
		output = fol.replace('/', '-') + '.txt'
		txt_files = glob.glob(os.path.join(dirname, fol, '*.txt'))
		print('{} records in {}...'.format(len(txt_files), output))
		files.append(output)

		with smart_open(os.path.join(dirname, output), 'wb') as n:
			for i, txt in enumerate(txt_files):
				with smart_open(txt, 'rb') as f:
					one_text = f.read().decode('utf-8')  # from binary to string
					one_text = normalize_text(one_text)  # convert to lower-case and strip punctuations
					n.write(one_text.encode('utf-8') + b'\n')  # from string to binary + newline

	return files


def select_imdb(select_num, dirname, files, file_splits, file_sentiments):
	"""
	Subset and split IMDB dataset into train/test.

	:param select_num: num of rows to select
	:param dirname: directory of txt files
	:param files: list of string name of files
	:param file_splits: list of string on train/test split
	:param file_sentiments: list of string on pos/neg sentiment label
	:return:
		list of namedtuple
	"""

	sent_doc = namedtuple('sent_doc', ['words', 'tags', 'split', 'sentiment'])
	all_doc = []
	doc_id = 0
	for i, fi in enumerate(files[:-1]):
		s_ = file_splits[i]
		se_ = file_sentiments[i]

		with smart_open(os.path.join(dirname, fi), 'rb', encoding='utf-8') as texts:
			for line_no, line in enumerate(texts):
				if line_no < select_num:
					tokens = gensim.utils.to_unicode(line).split()
					words = tokens  # must be a list for doc2vec
					tags = [doc_id]  # must be a list for doc2vec
					doc_id += 1
					split = s_
					sentiment = se_
					all_doc.append(sent_doc(words, tags, split, sentiment))
				else:
					break

	return all_doc


class DocPreprocess(object):

	def __init__(self,
				 nlp,
				 stop_words,
				 docs,
				 labels,
				 build_bi=False,
				 min_count=5,
				 threshold=10,
				 allowed_postags=['ADV', 'VERB', 'ADJ', 'NOUN', 'PROPN', 'NUM']):

		self.nlp = nlp  # spacy nlp object
		self.stop_words = stop_words  # spacy.lang.en.stop_words.STOP_WORDS
		self.docs = docs  # docs must be either list or numpy array or series of docs
		self.labels = labels # labels must be list or or numpy array or series of labels
		self.doc_ids = np.arange(len(docs))
		self.simple_doc_tokens = [gensim.utils.simple_preprocess(doc, deacc=True) for doc in self.docs]

		if build_bi:
			self.bi_detector = self.build_bi_detect(self.simple_doc_tokens, min_count=min_count, threshold=threshold)
			self.new_docs = self.make_bigram_doc(self.bi_detector, self.simple_doc_tokens)
		else:
			self.new_docs = self.make_simple_doc(self.simple_doc_tokens)
		self.doc_words = [self.lemmatize(doc, allowed_postags=allowed_postags) for doc in self.new_docs]
		self.tagdocs = [TaggedDocument(words=words, tags=[tag]) for words, tag in zip(self.doc_words, self.doc_ids)]


	def build_bi_detect(self, simple_doc_tokens, min_count, threshold):
		bi_ = gensim.models.phrases.Phrases(simple_doc_tokens, min_count=min_count, threshold=threshold)
		bi_detector = gensim.models.phrases.Phraser(bi_)  # wrapper enhance efficiency
		return bi_detector


	def make_bigram_doc(self, bi_detector, simple_doc_tokens):
		bi_doc_tokens = [bi_detector[doc_tokens] for doc_tokens in simple_doc_tokens]
		bi_docs = []
		for bi_tokens in bi_doc_tokens:
			bi_doc = " ".join(bi_tokens)  # concatenate back to a sentence
			bi_docs.append(bi_doc)
		return bi_docs


	def make_simple_doc(self, simple_doc_tokens):
		simple_docs = []
		for doc_tokens in simple_doc_tokens:
			simple = " ".join(doc_tokens)  # concatenate back to a sentence
			simple_docs.append(simple)
		return simple_docs


	def lemmatize(self, doc, allowed_postags):
		"""
		Lemmatize words and remove stop_words.

		:param doc: text
		:param allowed_postags: list of pos tags
		:return:
			list of tokens
		"""
		doc = self.nlp(doc)
		tokens = [token.lemma_ for token in doc if (
				token.pos_ in allowed_postags) and (token.text not in self.stop_words)]
		return tokens



class DocModel(object):

	def __init__(self, docs, **kwargs):
		"""

		:param docs: list of TaggedDocument
		:param kwargs: dictionary of (key,value) for Doc2Vec arguments
		"""
		self.model = Doc2Vec(**kwargs)
		self.docs = docs
		self.model.build_vocab([x for x in self.docs])

	def custom_train(self, fixed_lr=False, fixed_lr_epochs=None):
		"""
		Train Doc2Vec with two options, without fixed learning rate(recommended) or with fixed learning rate.
		Fixed learning rate also includes implementation of shuffling training dataset.


		:param fixed_lr: boolean
		:param fixed_lr_epochs: num of epochs for fixed lr training
		"""
		if not fixed_lr:
			self.model.train([x for x in self.docs],
							 total_examples=len(self.docs),
							 epochs=self.model.epochs)
		else:
			for _ in range(fixed_lr_epochs):
				self.model.train(utils.shuffle([x for x in self.docs]),
								 total_examples=len(self.docs),
								 epochs=1)
				self.model.alpha -= 0.002
				self.model.min_alpha = self.model.alpha  # fixed learning rate


	def test_orig_doc_infer(self):
		"""
		Use the original doc as input for model's vector inference,
		and then compare using most_similar()
		to see if model finds the original doc id be the most similar doc to the input.
		"""
		idx = np.random.randint(len(self.docs))
		print('idx: ' + str(idx))
		doc = [doc for doc in self.docs if doc.tags[0] == idx]
		inferred_vec = self.model.infer_vector(doc[0].words)
		print(self.model.docvecs.most_similar([inferred_vec]))  # wrap vec in a list


class MeanEmbeddingVectorizer(object):


	def __init__(self, word_model):
		self.word_model = word_model
		self.vector_size = word_model.wv.vector_size

	def fit(self):  # comply with scikit-learn transformer requirement
		return self

	def transform(self, docs):  # comply with scikit-learn transformer requirement
		doc_word_vector = self.word_average_list(docs)
		return doc_word_vector

	def word_average(self, sent):
		"""
		Compute average word vector for a single doc/sentence.


		:param sent: list of sentence tokens
		:return:
			mean: float of averaging word vectors
		"""
		mean = []
		for word in sent:
			if word in self.word_model.wv.vocab:
				mean.append(self.word_model.wv.get_vector(word))

		if not mean:  # empty words
			# If a text is empty, return a vector of zeros.
			logging.warning("cannot compute average owing to no vector for {}".format(sent))
			return np.zeros(self.vector_size)
		else:
			mean = np.array(mean).mean(axis=0)
			return mean


	def word_average_list(self, docs):
		"""
		Compute average word vector for multiple docs, where docs had been tokenized.

		:param docs: list of sentence in list of separated tokens
		:return:
			array of average word vector in shape (len(docs),)
		"""
		return np.vstack([self.word_average(sent) for sent in docs])


class TfidfEmbeddingVectorizer(object):

	def __init__(self, word_model):

		self.word_model = word_model
		self.word_idf_weight = None
		self.vector_size = word_model.wv.vector_size

	def fit(self, docs):  # comply with scikit-learn transformer requirement
		"""
		Fit in a list of docs, which had been preprocessed and tokenized,
		such as word bi-grammed, stop-words removed, lemmatized, part of speech filtered.

		Then build up a tfidf model to compute each word's idf as its weight.
		Noted that tf weight is already involved when constructing average word vectors, and thus omitted.

		:param
			pre_processed_docs: list of docs, which are tokenized
		:return:
			self
		"""

		text_docs = []
		for doc in docs:
			text_docs.append(" ".join(doc))

		tfidf = TfidfVectorizer()
		tfidf.fit(text_docs)  # must be list of text string

		# if a word was never seen - it must be at least as infrequent
		# as any of the known words - so the default idf is the max of
		# known idf's
		max_idf = max(tfidf.idf_)  # used as default value for defaultdict
		self.word_idf_weight = defaultdict(lambda: max_idf,
										   [(word, tfidf.idf_[i]) for word, i in tfidf.vocabulary_.items()])
		return self


	def transform(self, docs):  # comply with scikit-learn transformer requirement
		doc_word_vector = self.word_average_list(docs)
		return doc_word_vector


	def word_average(self, sent):
		"""
		Compute average word vector for a single doc/sentence.


		:param sent: list of sentence tokens
		:return:
			mean: float of averaging word vectors
		"""

		mean = []
		for word in sent:
			if word in self.word_model.wv.vocab:
				mean.append(self.word_model.wv.get_vector(word) * self.word_idf_weight[word])  # idf weighted

		if not mean:  # empty words
			# If a text is empty, return a vector of zeros.
			logging.warning("cannot compute average owing to no vector for {}".format(sent))
			return np.zeros(self.vector_size)
		else:
			mean = np.array(mean).mean(axis=0)
			return mean


	def word_average_list(self, docs):
		"""
		Compute average word vector for multiple docs, where docs had been tokenized.

		:param docs: list of sentence in list of separated tokens
		:return:
			array of average word vector in shape (len(docs),)
		"""
		return np.vstack([self.word_average(sent) for sent in docs])


















