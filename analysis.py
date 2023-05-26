# !pip install rake-nltk
import math
from typing import List, Set, Dict, Tuple, Union, Counter, Iterable, Iterator, Callable, NamedTuple
from collections import Counter
import itertools
from enum import Enum

import numpy as np
import os
from io import StringIO

from tqdm import tqdm
from language import Language, English
from vocabulary import Vocabulary
from corpus import Corpus, CorpusLevel, CorpusItem
from corpus_content_filter import ContentType, content_type_filter

import json
import numpy
from scipy.stats import kendalltau

from gensim.corpora import Dictionary
from gensim.similarities.docsim import MatrixSimilarity
from gensim.models import FastText, Word2Vec, TfidfModel
from gensim.interfaces import SimilarityABC
from gensim.similarities import WordEmbeddingSimilarityIndex, SparseTermSimilarityMatrix, SoftCosineSimilarity

import nltk
from nltk.corpus import stopwords


class FrequencyMode(Enum):
    """Defines the mode of accounting for term frequency during document similarity calculation:
       BOW - bag-of-words, direct term count, or TFIDF - convert BOW term counts using TFIDF model."""
    bow = 0,
    tfidf=1


CORPUS_INPUT_DIR = 'corpus_line_sentence'
OUT_DIR = 'analysis'

os.makedirs(OUT_DIR, exist_ok=True)


def get_exclude_words(language_name = English.name) -> Set[str]:
    # nltk.download('stopwords')
    exclude_words = set(stopwords.words(language_name))
    exclude_words |= {"n't", 't', 's', 'us','one','two','also','like','said','told','would','could', 'http'}
    return exclude_words


def corpus_items_list(corpus:Corpus, level:CorpusLevel, out_dir=OUT_DIR):
    titles = list(map(lambda corpus_item: corpus_item.title(), corpus.corpus_item_iterator(level=level, load_content=False)))
    corpus_items_file_name = level.name + '.json'
    corpus_items_file_path = os.path.join(out_dir, corpus_items_file_name)
    with open(corpus_items_file_path, mode='wt', encoding='utf-8') as f:
        json.dump(titles, f, indent=4)
    return titles


def rake_keyphrases(
    corpus:Corpus,
    level:CorpusLevel = CorpusLevel.book,
    keywords_count = 16,
    vocabulary:Iterable[str]= None,
    out_dir = OUT_DIR,
    keywords_base_file_name ='keyphrases'
    )->Dict[str, List[Tuple[str, float]]]:
    """Extract from the corpus texts the key-phrases using Rapid Automatic Keywords Extraction algorithm."""
    from rake_nltk import Rake

    item_count = sum([1 for corpus_item in corpus.corpus_item_iterator(level=level, load_content=False)])
    action_descr = 'Extracting {0}s keywords...'.format(level.name)
    progress = tqdm(range(item_count), total=item_count, desc=action_descr)
    corpus_items_keywords = dict() # title to keywords
    for corpus_item in corpus.corpus_item_iterator(level=level, load_content=True):
        # convert Iterable[List[str] - sentences as lists of tokens - to sentences as strings, with tokens delimited with space
        sentences = list(map(lambda tokens: ' '.join(tokens), corpus.sentences_of_tokens_iterator(corpus_item)))
        rake: Rake = Rake(language=English.name, include_repeated_phrases=False)
        rake.extract_keywords_from_sentences(sentences)
        scores_keyphrases:List[Tuple[float, str]] = rake.get_ranked_phrases_with_scores()[:keywords_count]
        keyphrases_scores:List[Tuple[str, float]] = list()
        # optionally convert from lower (as is output of rake-nltk)
        # to sentence/upper case using supplied vocabulary
        if vocabulary:
            for i in range(len(scores_keyphrases)):
                score, keyphrase = scores_keyphrases[i]
                keywords:List[str] = keyphrase.split(' ')
                keywords = Vocabulary.tokens_filter(keywords, vocabulary)
                keyphrase = ' '.join(keywords)
                keyphrases_scores.append((keyphrase, score))

        progress.update(1)
        corpus_items_keywords[corpus_item.title()] = keyphrases_scores

    keywords_file_name = '{0}_{1}.json'.format(keywords_base_file_name, level.name)
    file_path = os.path.join(out_dir, keywords_file_name)
    with open(file_path, 'wt', encoding='utf-8') as f:
        json.dump(corpus_items_keywords, f, indent=4)

    return corpus_items_keywords

def word2vec_model_train(
        corpus: Corpus,
        out_dir = OUT_DIR,
        model_file_name: str = 'word2vec.model',
        epochs=10) -> Word2Vec:

    sentences_of_tokens:List[List[str]] = list(corpus.sentences_of_tokens_iterator())
    sentences_count = len(sentences_of_tokens)
    print(f"sentences_count={sentences_count}")
    model = Word2Vec(vector_size=64, window=5, min_count=2)
    print('Building vocabulary...')
    model.build_vocab(sentences_of_tokens)
    print('Training Word2Vec model...')
    model.train(corpus_iterable=sentences_of_tokens, total_examples=sentences_count, epochs=epochs)  # train

    os.makedirs(out_dir, exist_ok=True)
    model_file_path = os.path.join(out_dir, model_file_name)
    print(f"Saving model to {model_file_path} ...")
    model.save(model_file_path)
    return model

def fasttext_model_train(
        corpus:Corpus,
        out_dir = OUT_DIR,
        model_file_name = 'fasttext.model'
        )->FastText:

    sentences_of_tokens = list(corpus.sentences_of_tokens_iterator())
    sentences_count = len(sentences_of_tokens)
    print(f"sentences_count={sentences_count}")
    model = FastText(vector_size=64, window=5, min_count=2)
    print('Building vocabulary...')
    model.build_vocab(corpus_iterable=sentences_of_tokens)
    print('Training FastText model...')
    model.train(corpus_iterable=sentences_of_tokens, total_examples=sentences_count, epochs=10)  # train

    os.makedirs(out_dir, exist_ok=True)
    model_file_path = os.path.join(out_dir, model_file_name)
    print(f"Saving model to {model_file_path} ...")
    model.save(model_file_path)
    return model

def corpus_dictionary(corpus:Corpus, level:CorpusLevel)->Dictionary:
    """Builds gensim's Dictionary from the texts of corpus in embedded line-sentence format."""
    item_count = sum([1 for corpus_item in corpus.corpus_item_iterator(level, load_content=False)])
    action_descr = 'Building gensim Dictionary over {0}s texts...'.format(level.name)
    progress = tqdm(range(item_count), total=item_count, desc = action_descr)
    dictionary = Dictionary()
    for corpus_item in corpus.corpus_item_iterator(level, load_content=True):
        sentences_of_tokens_iterator = corpus.sentences_of_tokens_iterator(corpus_item)
        # make flat list of tokens from all sentences
        corpus_item_tokens:List[str] = list(itertools.chain.from_iterable(sentences_of_tokens_iterator))
        dictionary.add_documents([corpus_item_tokens])
        progress.update(1)

    print(f"corpus dictionary: num_docs={dictionary.num_docs}, num_tokens={len(dictionary.token2id)}, num_pos={dictionary.num_pos}, num_nnz={dictionary.num_nnz}")
    return dictionary

def corpus_item_to_bow(
        corpus:Corpus,
        corpus_item:CorpusItem,
        dictionary:Dictionary
        )->List[Tuple[int, int]]:
    """Using the supplied gensim Dictionary,
       convert the united text(s) of a corpus item (book|section) to a bag-of-words form."""
    # Iterable[List[str]] - iterator over sentences as lists of tokens
    sentences_of_tokens_iterator = corpus.sentences_of_tokens_iterator(corpus_item)
    # make flat list of tokens from all sentences
    corpus_item_tokens = list(itertools.chain.from_iterable(sentences_of_tokens_iterator))
    corpus_item_bow: List[Tuple[int, int]] = dictionary.doc2bow(corpus_item_tokens)
    return corpus_item_bow

def text_top_terms(
        text_term_freq:List[Tuple[int, Union[int, float]]],
        dictionary:Dictionary,
        top_n:int,
        exclude_words:Iterable[str]=None
        ) -> Dict[str, Union[int, float]]:
    """Given text token statistics
        as a bag-of-words - list of (token_id, token_count) tuples
        or as tfidf vector - list of (token_id, token_tfidf) tuples,
    and the tokens ditionary, extracts and return the dictionary mapping
    token->(token_count|token_tfidf)
    of the top_n top terms with highest token_count|token_tfidf
    (excluding exclude_words, if any)."""
    text_term_freq_top = text_term_freq
    if exclude_words:
        text_term_freq_top = list(filter(
            lambda id_and_score: dictionary.get(id_and_score[0], '').lower() not in exclude_words,
            text_term_freq_top))
    text_term_freq_top = sorted(
        text_term_freq_top,
        key=lambda id_and_score: id_and_score[1], reverse=True)[:top_n]
    top_terms: Dict[str, Union[int, float]] = \
        {dictionary.get(id, ''): score for id, score in text_term_freq_top}
    if '' in top_terms: del top_terms['']
    return top_terms


def corpus_items_top_terms(
        corpus:Corpus,
        dictionary:Dictionary = None,
        freq_mode: FrequencyMode = FrequencyMode.bow,
        level: CorpusLevel = CorpusLevel.book,
        exclude_words:Iterable[str]=None,
        subset_filters:Dict[str, Callable[[CorpusItem], bool]] = None,
        out_dir=OUT_DIR,
        out_base_file_name='top_terms',
        top_n=16
        ) -> Dict[str, Dict[str, float]]:
    """Extracts top term, accortding to tfidf model, for each corpus item text."""

    dictionary = corpus_dictionary(corpus, level) if dictionary is None else dictionary
    tfidf_model = TfidfModel(dictionary=dictionary) if freq_mode == FrequencyMode.tfidf else None
    term_freq_transform = (lambda doc_bow: tfidf_model[doc_bow] if freq_mode == FrequencyMode.tfidf else doc_bow)
    item_count = sum([1 for corpus_item in corpus.corpus_item_iterator(level=level, load_content=False)])

    action_descr = 'Extracting from {0}s top terms with {1} model...'.format(level.name, freq_mode.name)
    progress = tqdm(range(item_count), total=item_count, desc=action_descr)
    title_to_topterms:Dict[str, Dict[str, float]] = dict()
    subset_filters = {} if subset_filters is None else subset_filters
    subset_word_counters = {subset_name:Counter() for subset_name in subset_filters.keys()}
    for corpus_item in corpus.corpus_item_iterator(level=level):
        corpus_item_bow:List[Tuple[int, int]] = corpus_item_to_bow(corpus, corpus_item, dictionary)
        corpus_item_term_freq:List[Tuple[int, Union[int,float]]] = term_freq_transform(corpus_item_bow)
        corpus_item_top_terms:Dict[str, Union[int, float]] = text_top_terms(
            corpus_item_term_freq, dictionary, top_n, exclude_words)
        title_to_topterms[corpus_item.title()] = corpus_item_top_terms

        # fill word counters for aggregated - subsets - items, as defined by the given filters
        for subset_name, filter in subset_filters.items():
            if filter(corpus_item):
                subset_word_counters[subset_name].update(dict(corpus_item_bow))

        progress.update(1)

    for subset_name, word_counter in subset_word_counters.items():
        subset_term_freq:List[Tuple[int, Union[int,float]]] = term_freq_transform(list(word_counter.items()))
        subset_top_terms:Dict[str, Union[int, float]] = text_top_terms(
            subset_term_freq, dictionary, top_n, exclude_words)
        title_to_topterms[subset_name] = subset_top_terms


    out_file_name = '{0}_{1}_{2}_{3}.json'.format(out_base_file_name, level.name, freq_mode.name, top_n)
    out_file_path = os.path.join(out_dir, out_file_name)
    with open(out_file_path, 'wt', encoding='utf-8') as f:
        json.dump(title_to_topterms, f, indent=4)

    return title_to_topterms


def corpus_top_terms(
    freq_modes: Iterable[FrequencyMode] = [FrequencyMode.bow, FrequencyMode.tfidf],
    levels: Iterable[CorpusLevel] = [CorpusLevel.book, CorpusLevel.section],
    top_n:int = 16):
    subset_filters = {
        '<TARGET>': content_type_filter(ContentType.target),
        '<SOURCE>': content_type_filter(ContentType.source)
        }

    for level in levels:
        dictionary:Dicionary = corpus_dictionary(corpus, level)
        for freq_mode in freq_modes:
            corpus_items_top_terms(
                corpus,
                dictionary=dictionary,
                freq_mode=freq_mode,
                level=level,
                exclude_words=get_exclude_words(),
                subset_filters=subset_filters,
                top_n=top_n)

def get_wv_model_type_name(wv_model:Word2Vec)->str:
    if isinstance(wv_model, FastText):
        return 'fasttext'
    elif isinstance(wv_model, Word2Vec):
        return 'word2vec'
    elif wv_model is None:
        return 'none'
    else:
        raise ValueError('Unsupported wv_model type={0}'.format(type(wv_model)))

def corpus_items_similarity(corpus: Corpus,
                            dictionary: Dictionary = None,
                            freq_mode: FrequencyMode = FrequencyMode.bow,
                            wv_model:Word2Vec = None,
                            level: CorpusLevel = CorpusLevel.book,
                            corpus_item_filter_target: Callable[[CorpusItem], bool] = None,
                            corpus_item_filter_source: Callable[[CorpusItem], bool] = None,
                            out_dir=OUT_DIR,
                            sim_base_file_name='sim',
                            top_n_docs=16,
                            top_n_terms:int=None
                            )  -> Dict[str, Dict[str, float]]:
    """Calculates pair-wise BoW- or TFIDF- based similarity of the texts in the corpus.
    if model_wv is specified, calculates SoftCosineSimilarity of the texts,
    using SparseTermSimilarityMatrix as the term similarity matrix;
    otherwise, if model_wv is None, calculates MatrixSimilarity of the texts,
    (which returns cosine similarity of the BOW or TFIDF vectors, representing the texts)
    corpus_item_filter_target, corpus_item_filter_source, if specified, define subsets of corpus items to compare:
    each corpus item from target subset vs each one from source subset.
    """

    dictionary = corpus_dictionary(corpus, level) if dictionary is None else dictionary
    tfidf_model = TfidfModel(dictionary=dictionary) if freq_mode == FrequencyMode.tfidf else None
    term_freq_transform = (lambda doc_bow: tfidf_model[doc_bow] if freq_mode == FrequencyMode.tfidf else doc_bow)

    corpus_item_count = sum([1 for corpus_item in corpus.corpus_item_iterator(level=level, load_content=False)])

    action_descr = 'Converting {0}s to BoW...'.format(level.name)
    progress = tqdm(range(corpus_item_count), total=corpus_item_count, desc=action_descr)
    # convert text of each corpus_item
    #    to gensim BoW   - list of (token_id, token_count) tuples
    # or to gensim tfidf - list of (token_id, token_tfidf) tuples
    cmp_item_titles:List[str] = []
    cmp_items_bow:List[List[Tuple[int, int]]] = []
    # indices of corpus items, which fall under source|target filter for comparison
    indices_target = []
    indices_source = []
    word_counter_target = Counter()
    word_counter_source = Counter()
    for i_corpus_item, corpus_item in enumerate(corpus.corpus_item_iterator(level=level, load_content=True)):
        cmp_item_titles.append(corpus_item.title())
        corpus_item_bow:List[Tuple[int, int]] = corpus_item_to_bow(corpus, corpus_item, dictionary)
        cmp_items_bow.append(corpus_item_bow)
        if corpus_item_filter_target is None or corpus_item_filter_target(corpus_item):
            indices_target.append(i_corpus_item)
            word_counter_target.update(dict(corpus_item_bow))
        if corpus_item_filter_source is None or corpus_item_filter_source(corpus_item):
            indices_source.append(i_corpus_item)
            word_counter_source.update(dict(corpus_item_bow))
        progress.update(1)

    # include special items to compare, representing all target|source texts united
    bow_target, bow_source = map(lambda wc: list(wc.items()), [word_counter_target, word_counter_source])
    cmp_items_bow.extend([bow_target, bow_source])
    cmp_item_titles.extend(['<TARGET>','<SOURCE>'])
    indices_target.append(corpus_item_count)
    indices_source.append(corpus_item_count+1)
    cmp_items_term_freq: List[List[Tuple[int, float]]] = list(map(term_freq_transform, cmp_items_bow))

    cmp_item_top_terms = None
    if top_n_terms is not None:
        cmp_item_top_terms = \
            [text_top_terms(term_freq, dictionary, top_n_terms, get_exclude_words()).keys() for term_freq in cmp_items_term_freq]

    doc_sim_mtx_index: SimilarityABC = None
    if wv_model is not None: # model_wv is None: compare documents, accounting for term similarity as per word vectors model
        term_sim_mtx:SparseTermSimilarityMatrix = get_term_sim_matrix(
            out_dir, freq_mode, level, wv_model, dictionary, tfidf_model)
        print('Building corpus index - SoftCosineSimilarity...')
        assert term_sim_mtx is not None
        doc_sim_mtx_index = SoftCosineSimilarity(cmp_items_term_freq, term_sim_mtx)
    else: # model_wv is None: compare documents simply by raw or tfidf term frequency
        print('Building corpus index - MatrixSimilarity...')
        doc_sim_mtx_index = MatrixSimilarity(cmp_items_term_freq, num_features=len(dictionary.token2id))

    cmp_item_count = len(cmp_items_term_freq) # corpus_item_count+2
    doc_sim_mtx = np.zeros((cmp_item_count, cmp_item_count))
    progress = tqdm(range(cmp_item_count), total=cmp_item_count, desc='Calculating {0}s similarity...'.format(level.name))
    for i, sim_row in enumerate(doc_sim_mtx_index):
        doc_sim_mtx[i] = sim_row
        progress.update(1)

    # mtx_sim_file_name = '{0}_{1}_{2}_mtx.npy'.format(sim_base_file_name, freq_mode.name, level.name)
    # file_path = os.path.join(out_dir, mtx_sim_file_name)
    # np.save(file_path, doc_sim_mtx)

    # with top_n_terms option to report the intersections of top terms, produce 2 files: without top terms and with them
    top_term_options = [None, cmp_item_top_terms] if cmp_item_top_terms is not None else [None]
    sim_closest: Dict[str, Dict[str, float]] = None
    wv_model_type = get_wv_model_type_name(wv_model)
    for cmp_item_top_terms in top_term_options:
        kw = '_kw' if cmp_item_top_terms is not None else ''
        sim_closest_file_name = '{0}_{1}_{2}_{3}_closest{4}.json'.format(sim_base_file_name, wv_model_type, level.name, freq_mode.name, kw)
        sim_closest_file_path = os.path.join(out_dir, sim_closest_file_name)
        ret_closest = corpus_items_similarity_closest(
            level,
            cmp_item_titles,
            cmp_item_top_terms,
            doc_sim_mtx,
            indices_target,
            indices_source,
            top_n_docs,
            sim_closest_file_path)
        if cmp_item_top_terms is None: sim_closest=ret_closest # return result without top terms intersections

    return sim_closest

def get_term_sim_matrix(
        matrix_dir:str,
        freq_mode:FrequencyMode,
        level:CorpusLevel,
        model_wv:Word2Vec,
        dictionary:Dictionary,
        tfidf_model:TfidfModel,
        ) -> SparseTermSimilarityMatrix:
    """Creates/loads created earlier returns a SparseTermSimilarityMatrix object,
    used for comparing documents, while accounting for terms similarity."""
    assert model_wv is not None
    assert isinstance(model_wv, Word2Vec)
    wv_model_type_name = get_wv_model_type_name(model_wv)

    term_sim_mtx_file_name = 'SparseTermSimilarityMatrix_{0}_{1}_{2}'.format(wv_model_type_name, level.name, freq_mode.name)
    term_sim_mtx_file_path = os.path.join(matrix_dir, term_sim_mtx_file_name)
    term_sim_mtx:SparseTermSimilarityMatrix = None
    if os.path.exists(term_sim_mtx_file_path):
        try:
            term_sim_mtx = SparseTermSimilarityMatrix.load(term_sim_mtx_file_path)
        except BaseException as exc:
            print(f"Failed loading SparseTermSimilarityMatrix from {term_sim_mtx_file_path}, {exc}")
    if term_sim_mtx is None:
        print('Building WordEmbeddingSimilarityIndex...')
        words = [word for word, count in dictionary.most_common()]
        word_vectors = model_wv.wv.vectors_for_all(words, allow_inference=False)  # produce vectors for words in corpus
        termsim_index = WordEmbeddingSimilarityIndex(word_vectors)
        print('Building SparseTermSimilarityMatrix...')
        term_sim_mtx = SparseTermSimilarityMatrix(termsim_index, dictionary, tfidf=tfidf_model)  # compute word similarities
        term_sim_mtx.save(term_sim_mtx_file_path)

    return term_sim_mtx


def corpus_items_similarity_closest(
        level:CorpusLevel,
        cmp_item_titles:List[str],
        cmp_item_top_terms:List[str],
        cmp_sim_mtx:np.ndarray,
        indices_target:List[int],
        indices_source:List[int],
        top_n:int,
        sim_closest_file_path:str
        ) -> Dict[str, Dict[str, float]]:
    """Analyzes the matrix of similarity between corpus items (books|sections),
    for each target item finds closest source items,
    writing results to the output .json file.
    Parameters:
        cmp_item_titles - titles of compared items;
            2 last elements represent aggregators - unions of texts of all target and all source items being compared.
        cmp_sim_mtx: 2d-matrix of comparing each each-vs-each item from the list cmp_item_titles;
        indices_target: define the indices (in cmp_item_titles, and cmp_sim_mtx) of items to compare;
            last index is expected to represent the aggregator - union of all [normal] target items.
        indices_source: define the indices (in cmp_item_titles, and cmp_sim_mtx) of items to compare against;
            last index is expected to represent the aggregator - union of all [normal] source items.
        top_n: number of top similar items to report;
        sim_closest_file_path: path to a .json file to write results into.
    """
    item_count = len(cmp_item_titles)
    assert cmp_sim_mtx.shape == (item_count, item_count)

    sim_closest: Dict[str, List[Tuple[int, float, str]]] = dict()
    index_source_all = indices_source[-1]    # all-source-items aggregator
    indices_source_raw = indices_source[:-1] # source item indices, excluding all-source-items aggregator
    for i in indices_target: # for each target item find top_n closest to it source items
        closest_indices = list(filter(lambda j: j in indices_source, np.argsort(-cmp_sim_mtx[i])))[:top_n]
        # report result of comparing vs all-source-items, even if it is not in top_n
        if index_source_all not in closest_indices: closest_indices.append(index_source_all)
        closest_to_i:Dict[str,Union[float,Dict]] = dict() # maps target item title to per-source-item-cmp-result
        for j in closest_indices:
            score_i_j = cmp_sim_mtx[i,j]
            common_terms = set(cmp_item_top_terms[i]) & set(cmp_item_top_terms[j]) if cmp_item_top_terms else None
            common_terms = [key for key in common_terms] if common_terms is not None else None
            closest_to_i[cmp_item_titles[j]] = {'score':score_i_j, 'common_terms':common_terms} if (common_terms is not None) else score_i_j
        # report average-per-source-item result
        sim_avg_to_i:float = cmp_sim_mtx[i][indices_source_raw].sum() / len(indices_source_raw)
        closest_to_i['SOURCE_MICRO'] = {'score': sim_avg_to_i, 'common_terms':None} if cmp_item_top_terms is not None else sim_avg_to_i
        sort_value_fn = (lambda value: value.get('score', math.nan) if isinstance(value, dict) else value )
        closest_to_i = dict(sorted(list(closest_to_i.items()), key=(lambda key_value: sort_value_fn(key_value[1])), reverse=True))

        title_i = cmp_item_titles[i]
        sim_closest[title_i] = closest_to_i

    if sim_closest_file_path is not None:
        with open(sim_closest_file_path, mode='wt', encoding='utf-8') as f:
            json.dump(sim_closest, f, indent=4)

    return sim_closest

def corpus_items_similarity_target_source(
        corpus:Corpus,
        wv_models:Iterable[Word2Vec],
        freq_modes:Iterable[FrequencyMode] = [FrequencyMode.bow, FrequencyMode.tfidf],
        levels:Iterable[CorpusLevel] = [CorpusLevel.book, CorpusLevel.section],
        sim_base_file_name='sim_ts',
        top_n_docs:int=64,
        top_n_terms:int=None):

    for level in levels:
        dictionary: Dictionary = corpus_dictionary(corpus, level)
        for wv_model in wv_models:
            wv_model_type = 'none' if wv_model is None else str(type(wv_model))
            for freq_mode in freq_modes:
                print(
                    f"Launching corpus_items_similarity(model_type={wv_model_type}, freq_mode={freq_mode.name}, level={level.name})")
                corpus_items_similarity(corpus,
                                        dictionary=dictionary,
                                        freq_mode=freq_mode,
                                        wv_model=wv_model,
                                        level=level,
                                        corpus_item_filter_target=content_type_filter(ContentType.target),
                                        corpus_item_filter_source=content_type_filter(ContentType.source),
                                        sim_base_file_name=sim_base_file_name,
                                        top_n_docs=top_n_docs,
                                        top_n_terms=top_n_terms)



def merge_similar_fragments(
        text_fragments:List[List[List[str]]],
        dictionary:Dictionary,
        term_sim_mtx:SparseTermSimilarityMatrix,
        tfidf_model:TfidfModel = None,
        verbosity=0
        )-> List[List[List[str]]]:
    """Merges together fragments with similar content into larger fragments of text.
    Parameters:
        text_fragments - list of text fragments; a text fragment is a list of sentences, a sentence is a list of string tokens.
        dictionary - corpus dictionary, for converting fargments to BOW format.
        term_sim_mtx - term similarity matrix, used for calculating similarity between fragments of text.
        tfidf_model - optional, TFIDF model to account for tokens frequences.
        Return: list of new fragments, obtained via merging together similar adjoining fragments recursively.
    """
    import math
    fragment_count = len(text_fragments)
    if fragment_count <= 1 : return text_fragments
    sentences:List[List[str]] = list(itertools.chain.from_iterable(text_fragments)) # flatten list of all sentences
    sentence_count = len(sentences)
    sentence_bow:List[List[Tuple[int, int]]] = [dictionary.doc2bow(sentence) for sentence in sentences]
    sentence_term_freq = tfidf_model[sentence_bow] if tfidf_model is not None else sentence_bow
    sentence_sim_index:SoftCosineSimilarity = SoftCosineSimilarity(sentence_term_freq, term_sim_mtx)

    sentence_sim_mtx = np.zeros((sentence_count, sentence_count))
    for i, sim_row in enumerate(sentence_sim_index):
        sentence_sim_mtx[i] = sim_row

    def sentence_sim(i:int, j:int):
        """Sentence-to-sentence similarity"""
        return sentence_sim_mtx[i, j]

    fragments_index = [] # text fragments as lists of indices of contained sentences
    i_sentence = 0
    for text_fragment in text_fragments:
        fragment_len = len(text_fragment)
        fragment_index = list(range(i_sentence, i_sentence+fragment_len))
        fragments_index.append(fragment_index)
        i_sentence += fragment_len
    if fragments_index:
        assert fragments_index[-1][-1] == sentence_count-1

    class FragmentInternalSim(NamedTuple):
        """Holds internal similarity of a fragment."""
        min_adj_sim: float
        avg_intrn_sim: float
        adj_sims: List[float]

    def fragment_internal_sim(fragment_index:List[int])->FragmentInternalSim:
        """Returns several statistics of simlarities between sentences of a text fragment."""
        adj_sims = []
        for i in range(len(fragment_index)-1):
            adj_sims.append(sentence_sim(fragment_index[i], fragment_index[i+1]))
        adj_sent_filtered = list(filter(lambda s: s != 0.0 and not math.isnan(s), adj_sims))
        min_adj_sim = min(adj_sent_filtered) if adj_sent_filtered else math.nan

        sim_sum = 0.0
        sim_count = 0
        for i in range(1, len(fragment_index)):
            for j in range(0, i):
                sim_sum += sentence_sim(fragment_index[i], fragment_index[j])
                sim_count += 1
        avg_internal_sim = sim_sum/sim_count if sim_count!=0 else math.nan

        return FragmentInternalSim(min_adj_sim, avg_internal_sim, adj_sims)

    def fragments_sim(fragment_index1:List[int], fragment_index2:List[int]) -> float:
        sim_sum = 0.0
        assert len(fragment_index1) != 0
        assert len(fragment_index2) != 0
        for i in fragment_index1:
            for j in fragment_index2:
                sim_sum += sentence_sim(i, j)
        return sim_sum / ( len(fragment_index1)*len(fragment_index2) )

    class Fragment(NamedTuple):
        """Holds indices of contained sentences and internal similarity statistics."""
        index: List[int]
        internal: FragmentInternalSim

    class MergePoint(NamedTuple):
        """Holds score of candidate merge point of two adjacent fragments."""
        score:float
        fragments_sim:float

    def fragments_merge_point(fragment1:Fragment, fragment2:Fragment) -> Tuple[float, float]:
        avg_intrn_sims =  [fragment1.internal.avg_intrn_sim, fragment2.internal.avg_intrn_sim]
        avg_intrn_sims = list(filter(lambda v: not math.isnan(v), avg_intrn_sims))
        min_intrn_sim = min(avg_intrn_sims) if len(avg_intrn_sims) != 0 else 0.0
        fragments_sim_value = fragments_sim(fragment1.index, fragment2.index)
        # decisive score: how greater is the between-fragments similarity
        # compared to min within-fragment similarity
        score = fragments_sim_value - min_intrn_sim
        return MergePoint(score, fragments_sim_value)

    # main cycle: find the most similar adjacent fragments to merge and merge them
    fragments:List[Fragment] = list(map(
        lambda index: Fragment(index, fragment_internal_sim(index)), fragments_index))

    # with n fragments we have (n-1) candidate points of merge
    merge_points: List[MergePoint] = list(map(
        lambda i : fragments_merge_point(fragments[i], fragments[i+1]), range(len(fragments)-1) ))

    if verbosity >= 2:
        for i in range(len(fragments)-1):
            print(f"{fragments[i]}\n{fragments[i+1]}\n{merge_points[i]}\n")

    while len(fragments) > 1:
        merge_score_values = list(map(lambda merge_score: merge_score.score, merge_points))
        # between-fragments similarity less than within-fragments similarity for all merge points
        max_score = max(merge_score_values)
        if max_score <= 0.0:
            break
        i_merge = merge_score_values.index(max_score)
        # merge adjacent fragments at indices i_merge, i_merge+1
        if verbosity >= 2:
            print('Merge:')
            print(f"{fragments[i_merge]}")
            print(f"{fragments[i_merge+1]}")
            print(f"{merge_points[i_merge]}\n")
        index_merged:List[int] = fragments[i_merge].index + fragments[i_merge+1].index
        fragment_merged:Fragment = Fragment(index_merged, fragment_internal_sim(index_merged))
        fragments[i_merge:(i_merge+2)] = [fragment_merged]
        merge_points[i_merge:(i_merge+1)] = []
        # update merge scores of the new merged fragment with its adjacent fragments
        if i_merge > 0:
            merge_points[i_merge-1] = fragments_merge_point(fragments[i_merge-1], fragments[i_merge])
        if i_merge < len(fragments)-1:
            merge_points[i_merge] = fragments_merge_point(fragments[i_merge], fragments[i_merge+1])

    if verbosity >= 2:
        print('Final fragments:')
        for fragment in fragments:
            print(fragment)

    class FragmentsStat(NamedTuple):
        fragment_count:int
        sentence_count:float

    def fragments_stat(text_fragments: List[List[List[str]]])->FragmentsStat:
        fragment_count = len(text_fragments)
        sentence_count = sum([len(text_fragment) for text_fragment in text_fragments])
        return FragmentsStat(fragment_count, sentence_count)

    text_fragments_new = list(map(
        lambda fragment: [sentences[i] for i in fragment.index],
        fragments))

    if verbosity >= 1:
        fragments_old = fragments_stat(text_fragments)
        fragments_new = fragments_stat(text_fragments_new)
        print(f"sentence_count:{fragments_old.sentence_count}; fragments_count: {fragments_old.fragment_count}->{fragments_new.fragment_count}")

    return text_fragments_new

def merge_similar_paragraphs(
        text: str,
        dictionary: Dictionary,
        term_sim_mtx: SparseTermSimilarityMatrix,
        tfidf_model: TfidfModel = None,
        delim_line: str = '\n',
        verbosity=0) -> str:
    """In the input text, given in the line-sentence format
    (one sentence per line, paragraphs delimited with an empty line),
    merge together the similar paragraphs into bigger paragraphs;
    output is in the same line-sentence format."""

    # split to paragraphs/sentences/tokens
    paragraphs:List[List:List[str]] = Corpus.line_sentence_text_to_paragraphs(text)

    paragraphs_merged:List[List[List[str]]] = merge_similar_fragments(
        paragraphs, dictionary, term_sim_mtx, tfidf_model, verbosity)

    text_out = StringIO()
    for i_paragraph, paragraph in enumerate(paragraphs_merged):
        for sentence in paragraph:
            text_out.write(' '.join(sentence) + '\n')
        if i_paragraph != (len(paragraphs_merged)-1):
            text_out.write(delim_line) # end-of-paragraph marker

    return text_out.getvalue()

def corpus_merge_similar_paragraphs(
        corpus:Corpus,
        wv_models:Iterable[Word2Vec],
        freq_modes:Iterable[FrequencyMode]=[FrequencyMode.bow, FrequencyMode.tfidf],
        corpus_out_dir:str=None,
        delim_line='\n',
        verbosity=1):

    dictionary = corpus_dictionary(corpus, CorpusLevel.section)

    for wv_model in wv_models:
        for freq_mode in freq_modes:
            tfidf_model = TfidfModel(dictionary=dictionary) if freq_mode==FrequencyMode.tfidf else None

            wv_model_type = get_wv_model_type_name(wv_model)
            corpus_out_dir = '{0}_pg_merged_{1}_{2}'.format(corpus.corpus_dir, wv_model_type, freq_mode.name)
            print(f"merge_similar_paragraphs(wv_model={wv_model_type}, freq_mode={freq_mode}) -> {corpus_out_dir}")

            term_sim_mtx: SparseTermSimilarityMatrix = get_term_sim_matrix(
                OUT_DIR, freq_mode, CorpusLevel.section, wv_model, dictionary, tfidf_model)

            text_transform = (lambda text:
                              merge_similar_paragraphs(
                                  text, dictionary, term_sim_mtx, tfidf_model, delim_line, verbosity))
            corpus.corpus_text_transform(corpus_out_dir, text_transform, 'Merging similar paragraphs...')

class ParagraphMatch(NamedTuple):
    """Holds information about matched paragraphs."""
    similarity:float
    title_target:str
    title_source:str
    text_target:str
    text_source:str

def corpus_match_paragraphs(
        corpus: Corpus,
        corpus_orig_text: Corpus,
        wv_model: Word2Vec,
        freq_mode: FrequencyMode,
        level: CorpusLevel,
        corpus_item_filter_target: Callable[[CorpusItem], bool],
        corpus_item_filter_source: Callable[[CorpusItem], bool],
        out_dir=OUT_DIR,
        out_base_file_name='match_pg',
        min_paragraph_len=1,
        top_n_match=64,
        top_n_match_target: int = None,
        omit_target_text: bool = False,
        omit_source_text: bool = False
        ) -> List[ParagraphMatch]:
    """Locates in the source texts the paragraphs, most similar to paragraphs in target texts."""
    def get_paragraph_title(corpus_item_title_path:List[str], paragraph_index:int) -> str:
        pg_index = '{0:03}'.format(paragraph_index)
        paragraph_title_path = corpus_item_title_path[:]
        paragraph_title_path.append(pg_index)
        return CorpusItem.title_from_path(paragraph_title_path)

    def get_paragraph_bow(paragraph:List[List[str]]) -> List[Tuple[int, Union[int,int]]]:
        paragraph_tokens = list(itertools.chain.from_iterable(paragraph))
        paragraph_bow = dictionary.doc2bow(paragraph_tokens)
        return paragraph_bow

    def get_paragraph_term_freq(paragraph:List[List[str]], term_freq_transform) -> List[Tuple[int, Union[int,float]]]:
        return term_freq_transform(get_paragraph_bow(paragraph))

    def get_paragraph_text(paragraph_title:str, paragraph:List[List[str]]) -> str:
        """Find and return original text of the tokenized paragraph's sentences."""
        text_default = ' '.join([' '.join(sentence) for sentence in paragraph])
        if corpus_orig_text is None:
            return text_default
        text_title_path = CorpusItem.title_to_path(paragraph_title)[:-1] # skip paragraph index, the last path item
        tokens_filter = (lambda tokens: Vocabulary.tokens_filter(tokens, dictionary.token2id.keys()))
        sentence_texts:List[str] = corpus_orig_text.find_tokenized_sentences_text(
            text_title_path, paragraph, tokens_filter)
        if sentence_texts is None: return text_default
        return ' '.join(sentence_texts)

    print(f"corpus_match_paragraphs: corpus_dir={corpus.corpus_dir}, wv_model={get_wv_model_type_name(wv_model)}, freq_mode={freq_mode}, min_paragraph_len={min_paragraph_len}, top_n_match={top_n_match}")

    dictionary:Dictionary = corpus_dictionary(corpus, level)
    tfidf_model = TfidfModel(dictionary=dictionary) if freq_mode == FrequencyMode.tfidf else None
    term_freq_transform = (lambda doc_bow: tfidf_model[doc_bow] if freq_mode == FrequencyMode.tfidf else doc_bow)

    target_paragraphs:List[List[List[str]]]= []
    target_paragraphs_title:List[str] = []
    target_paragraphs_term_freq: List[List[Tuple[int, Union[int, float]]]] = []

    target_text_count = sum([1 for corpus_item_target in corpus.corpus_item_leaf_iterator(None, corpus_item_filter_target)])
    target_paragraphs_word_counter: Counter[str] = Counter() # bow/tfidf for united target texts
    progress_msg = "Converting target paragraphs' texts to {0}".format(freq_mode.name)
    progress = tqdm(corpus.corpus_item_leaf_iterator(None, corpus_item_filter_target),
                    total=target_text_count,
                    desc=progress_msg)
    for corpus_item_target in progress:
        paragraphs:List[List[List[str]]] = Corpus.line_sentence_text_to_paragraphs(corpus_item_target.text)
        for i_paragraph, paragraph in enumerate(paragraphs):
            target_paragraph_bow:List[Tuple[int,int]] = get_paragraph_bow(paragraph)
            target_paragraphs_word_counter.update(dict(target_paragraph_bow))
            if len(paragraph) >= min_paragraph_len:
                target_paragraphs.append(paragraph)
                paragraph_title = get_paragraph_title(corpus_item_target.title_path, i_paragraph)
                target_paragraphs_title.append(paragraph_title)
                target_paragraphs_term_freq.append(term_freq_transform(target_paragraph_bow))
    # add a special target paragraph, representing all target texts united
    TARGET_NAME = '<TARGET>'
    if len(target_paragraphs) > 1:
        target_paragraphs_title.append(TARGET_NAME)
        target_paragraphs_bow = list(target_paragraphs_word_counter.items())
        target_paragraphs_term_freq.append(term_freq_transform(target_paragraphs_bow))
        target_paragraphs.append(None) # do not store content of the united target texts since we do not output it anyway
    elif len(target_paragraphs) == 1:
        target_paragraphs_title[0] = TARGET_NAME
        target_paragraphs[0] = None
    assert len(target_paragraphs) == len(target_paragraphs_title) == len(target_paragraphs_term_freq)

    target_paragraph_count = len(target_paragraphs_title)
    print(f"target_paragraph_count={target_paragraph_count}")

    term_sim_mtx:SparseTermSimilarityMatrix = get_term_sim_matrix(
        OUT_DIR, freq_mode, level, wv_model, dictionary, tfidf_model)

    assert term_sim_mtx is not None
    # create the index of target paragraphs, to match source paragraphs against them, taking into account terms similarity
    target_index = SoftCosineSimilarity(target_paragraphs_term_freq, term_sim_mtx)
    paragraph_matches:List[ParagraphMatch] = []

    model_type_name = get_wv_model_type_name(wv_model)
    out_file_name = '{0}_{1}_{2}.json'.format(out_base_file_name, model_type_name, freq_mode.name)
    out_file_path = os.path.join(out_dir, out_file_name)

    # gather statistics:
    #   average source-paragraph-to-target-paragraph similarity
    #   avreage source-paragraph-to-united-target-paragraphs similarity
    sim_sum:float = 0.0
    sim_sum_target:float=0.0
    sources_paragraph_count = 0 # number of paragraphs in all processed sources

    # iterate over source books, match texts of theit paragraphs vs target paragraph texts
    source_book_count = sum([1 for book_item in corpus.books_iterator(False, corpus_item_filter_source)])
    progress = tqdm(
            corpus.books_iterator(True, corpus_item_filter_source),
            desc='Matching source to target paragraphs',
            total=source_book_count)
    for book_item_source in progress:
        source_paragraphs_title:List[str] = []
        source_paragraphs:List[List[List[str]]] = []
        for corpus_item_source in corpus.corpus_item_leaf_iterator(book_item_source, corpus_item_filter_source):
            paragraphs:List[List[List[str]]] = Corpus.line_sentence_text_to_paragraphs(corpus_item_source.text)
            for i_paragraph, paragraph in enumerate(paragraphs):
                if len(paragraph) >= min_paragraph_len:
                    paragraph_title = get_paragraph_title(corpus_item_source.title_path, i_paragraph)
                    source_paragraphs_title.append(paragraph_title)
                    source_paragraphs.append(paragraph)

        source_paragraphs_term_freq:List[List[Tuple[int, Union[int, float]]]] = list(map(
            lambda paragraph: get_paragraph_term_freq(paragraph, term_freq_transform), source_paragraphs ))

        source_paragraph_count = len(source_paragraphs)
        sources_paragraph_count += source_paragraph_count
        progress_msg = '{0}, {1} paragraphs'.format(book_item_source.title(), source_paragraph_count)
        progress.set_description(progress_msg)
        src_trg_sim_mtx:np.ndarray = target_index[source_paragraphs_term_freq]
        assert src_trg_sim_mtx.shape == (source_paragraph_count, target_paragraph_count)

        # take top_n_match matched pairs in order of decreasing similarity
        top_n = min(src_trg_sim_mtx.size-1, top_n_match)
        top_flat_indices:np.ndarray = (-src_trg_sim_mtx).argpartition(top_n, axis=None)
        assert top_flat_indices.shape == (source_paragraph_count*target_paragraph_count,)
        for i_flat_top in top_flat_indices[:top_n]:
            i_target_paragraph = (i_flat_top % target_paragraph_count)
            i_source_paragraph = int(i_flat_top/target_paragraph_count)
            similarity:float = src_trg_sim_mtx[i_source_paragraph, i_target_paragraph]
            title_target = target_paragraphs_title[i_target_paragraph]
            title_source = source_paragraphs_title[i_source_paragraph]
            text_target = target_paragraphs[i_target_paragraph] if not omit_target_text else None
            text_source = source_paragraphs[i_source_paragraph] if not omit_source_text else None
            paragraph_match = ParagraphMatch(float(similarity), title_target, title_source, text_target, text_source)
            paragraph_matches.append(paragraph_match)
        # leave top_n_match best matches
        paragraph_matches.sort(key = (lambda pm: pm.similarity), reverse=True)
        if len(paragraph_matches) > top_n_match:
            paragraph_matches[top_n_match:] = []
        # optimization: since retrieval of original paragraph text is a costly operation
        # do it only for the paragraphs which got onto top (so far, after current book processed)
        for i_paragraph_match in range(len(paragraph_matches)):
            paragraph_match:ParagraphMatch = paragraph_matches[i_paragraph_match]
            # replace paragraph's sentences-as-lists-of-tokens with (almost) original string content
            text_source, text_target = None, None
            if paragraph_match.text_source is not None and isinstance(paragraph_match.text_source, list):
                text_source = get_paragraph_text(paragraph_match.title_source, paragraph_match.text_source)
            if paragraph_match.text_target is not None and isinstance(paragraph_match.text_target, list):
                text_target = get_paragraph_text(paragraph_match.title_target, paragraph_match.text_target)
            if text_source is not None or text_target is not None:
                paragraph_match_dict = paragraph_match._asdict()
                if text_source is not None: paragraph_match_dict.update({'text_source':text_source})
                if text_target is not None: paragraph_match_dict.update({'text_target':text_target})
                paragraph_matches[i_paragraph_match] = ParagraphMatch(*paragraph_match_dict.values())
        # calculate and fill statistics
        single_target = (target_paragraphs_title == [TARGET_NAME])
        target_paragraph_count_raw = (target_paragraph_count-1 if not single_target else 1)
        sim_sum += src_trg_sim_mtx[:,:target_paragraph_count_raw].sum() # sum of source-paragraph-to-target-paragraph similarity
        sim_sum_target += src_trg_sim_mtx[:,-1].sum() # sum of source-paragraph-to-united-target-paragraphs similarity
        sim_micro_avg = sim_sum/(sources_paragraph_count*target_paragraph_count_raw)
        sim_target_micro_avg = sim_sum_target/sources_paragraph_count
        paragraph_match_target_micro_avg = ParagraphMatch(sim_target_micro_avg, '<TARGET>', '<SOURCE_MICRO>', None, None)
        paragraph_match_micro_avg = ParagraphMatch(sim_micro_avg, '<TARGET_MICRO>', '<SOURCE_MICRO>', None, None)
        paragraph_matches.extend([paragraph_match_target_micro_avg,paragraph_match_micro_avg])

        with open(out_file_path, mode='wt', encoding='utf-8') as fout:
            pm_list:List[Dict] = list(map(lambda pm: pm._asdict(), paragraph_matches))
            json.dump(pm_list, fout, indent=4)
    # end source iteration
    print(f"sources_paragraph_count={sources_paragraph_count}")

def corpus_match_paragraphs_target_source(
        wv_models:Iterable[Word2Vec],
        freq_modes:Iterable[FrequencyMode] = [FrequencyMode.bow, FrequencyMode.tfidf],
        out_dir=OUT_DIR,
        min_paragraph_len=1,
        top_n_match=64,
        ):
    corpus_orig_text = Corpus('corpus_dehyphen', '*.txt')
    #target_filter = (lambda corpus_item: corpus_item.title().find('12 points') >= 0)
    #subsetsuffix = '_12points'
    target_filter = content_type_filter(ContentType.target)
    subsetsuffix = ''
    source_filter = content_type_filter(ContentType.source)
    for wv_model in wv_models:
        for freq_mode in freq_modes:
            wv_model_type: str = get_wv_model_type_name(wv_model)
            input_dir = '{0}_pg_merged_{1}_{2}'.format(CORPUS_INPUT_DIR, wv_model_type, freq_mode.name)
            corpus = Corpus(input_dir, '*.txt')
            corpus_match_paragraphs(corpus, corpus_orig_text, wv_model, freq_mode, CorpusLevel.section,
                                    target_filter, source_filter,
                                    out_dir=OUT_DIR, out_base_file_name='match_pg_merged'+subsetsuffix,
                                    min_paragraph_len=min_paragraph_len, top_n_match=top_n_match,
                                    omit_target_text=False, omit_source_text=False)


def top_terms_neigbours(top_terms_file_name:str, title:str):
    top_terms = []
    with open(os.path.join(OUT_DIR, top_terms_file_name), mode='rt', encoding='utf-8') as f:
        title_to_top_terms:Dict[str, Dict[str, float]] = json.load(f)
        top_terms_with_scores = title_to_top_terms.get(title, {})
        top_terms = list(top_terms_with_scores.keys())

    for term in top_terms:
        words_close = dict(model_word2vec.wv.most_similar(positive=term, topn=4))
        print(f"{term}: {words_close}")

def cmp_ranking_pair_kendall(ranking1:List[str], ranking2:List[str])->Dict[str, float]:
    """Calculates Kendall-tau correlation of two rankings, given as lists of string items
    (where items presumably follow in the ranking's order)."""
    intersection:Set[str] = set(ranking1) & set(ranking2)
    r1_filtered = list(filter(lambda s: s in intersection, ranking1))
    r2_filtered = list(filter(lambda s: s in intersection, ranking2))
    assert len(r1_filtered) == len(r2_filtered)
    count = len(r1_filtered)
    r1 = list(range(0, count))
    r2 = [r2_filtered.index(s) for s in r1_filtered]
    result = kendalltau(r1, r2)._asdict()
    result.update({'common_elem_count':count})
    return result

def cmp_file_rankings_kendall(level:CorpusLevel, base_file_name:str='sim_ts'):
    """Compute average (per-target-text) of the Kendall-tau correlations of similarity rankings,
       produced by each pair of models, matching texts on book or section level."""
    file_rankings:List[Dict[str, Dict[str, float]]] = []
    model_names:List[str] = []
    for w2v_model_type in ['none','word2vec','fasttext']:
        for freq_mode_name in ['bow','tfidf']:
            filename = '{0}_{1}_{2}_{3}_closest.json'.format(base_file_name, w2v_model_type, level.name, freq_mode_name)
            filepath = os.path.join(OUT_DIR, filename)
            if os.path.exists(filepath):
                with open(filepath, mode='r', encoding='utf-8') as f:
                    file_ranking = json.load(f)
                    file_rankings.append(file_ranking)
                    model_name = '({0},{1})'.format(w2v_model_type, freq_mode_name)
                    model_names.append(model_name)

    print(f"{level.name}: model_names={model_names}")

    for i in range(1, len(file_rankings)):
        assert file_rankings[i].keys() == file_rankings[0].keys()

    model_count = len(file_rankings)
    title_count = len(file_rankings[0].keys())

    arr_clr = np.zeros((model_count, model_count, title_count))
    arr_pval = np.zeros((model_count, model_count, title_count))
    arr_cmn_el_cnt = np.zeros((model_count, model_count, title_count))

    for k, target_title in enumerate(file_rankings[0].keys()):
        title_rankings:List[List[str]] = \
            [list(file_rankings[i][target_title].keys()) for i in range(len(file_rankings))]

        for i in range(0, model_count):
            for j in range(i, model_count):
                ranking_i = title_rankings[i]
                ranking_j = title_rankings[j]
                kendall_i_j:Dict[str, float] = cmp_ranking_pair_kendall(ranking_i, ranking_j)
                arr_clr[i,j,k] = arr_clr[j,i,k] = kendall_i_j.get('correlation', math.nan)
                arr_pval[i,j,k] = arr_pval[j,i,k] = kendall_i_j.get('pvalue', math.nan)
                arr_cmn_el_cnt[i,j,k] = arr_cmn_el_cnt[j,i,k] = kendall_i_j.get('common_elem_count', math.nan)

    kendall_cmp_dict:Dict[str, Dict[str, float]] = dict()
    for i in range(0, model_count-1):
        for j in range(i+1, model_count):
            cmp_i_j:Dict[str, float] = dict()
            cmp_i_j['correlation_avg'] = arr_clr[i,j].mean()
            cmp_i_j['correlation_min'] = arr_clr[i,j].min()
            cmp_i_j['correlation_max'] = arr_clr[i,j].max()
            cmp_i_j['pvalue_geom_avg'] = np.exp( np.log(arr_pval[i,j]).mean() )
            cmp_i_j['pvalue_max'] = arr_pval[i,j].max()
            cmp_i_j['pvalue_min'] = arr_pval[i,j].min()
            cmp_i_j['commmon_elem_count_avg'] = arr_cmn_el_cnt[i,j].mean()
            cmp_i_j['commmon_elem_count_min'] = arr_cmn_el_cnt[i,j].min()
            cmp_i_j['commmon_elem_count_max'] = arr_cmn_el_cnt[i,j].max()
            cmp_pair_name = '{0}<->{1}'.format(model_names[i], model_names[j])
            kendall_cmp_dict[cmp_pair_name] = cmp_i_j

    filename = '{0}_kendall_{1}.json'.format(base_file_name, level.name)
    filepath = os.path.join(OUT_DIR, filename)
    with open(filepath, mode='wt', encoding='utf-8') as fout:
        json.dump(kendall_cmp_dict, fout, indent=4)
    return kendall_cmp_dict




corpus = Corpus(CORPUS_INPUT_DIR, '*.txt')

# corpus_items_list(corpus, level=CorpusLevel.book)
# corpus_items_list(corpus, level=CorpusLevel.section)

# corpus_word_counter = Vocabulary.load_word_counter(os.path.join(corpus.corpus_dir, 'corpus_word_counter.json'))
# rake_keyphrases(corpus, level = CorpusLevel.book, vocabulary=corpus_word_counter.keys())
# rake_keyphrases(corpus, level = CorpusLevel.section, vocabulary=corpus_word_counter.keys())

# corpus_top_terms()
# top_terms_neigbours('top_terms_tfidf_section_16.json', '<TARGET>')

word2vec_model_train(corpus, out_dir=OUT_DIR, model_file_name='word2vec.model')
fasttext_model_train(corpus, out_dir=OUT_DIR, model_file_name='fasttext.model')
print('Loading Word2Vec model')
word2vec_model = Word2Vec.load(os.path.join(OUT_DIR, 'word2vec.model'))
print('Loading FastText model')
fasttext_model = FastText.load(os.path.join(OUT_DIR, 'fasttext.model'))

corpus_items_similarity_target_source(
    corpus, [None, word2vec_model, fasttext_model], [FrequencyMode.bow, FrequencyMode.tfidf], [CorpusLevel.book, CorpusLevel.section], sim_base_file_name='sim_ts', top_n_docs=64, top_n_terms=256)

corpus_merge_similar_paragraphs(
    corpus, [word2vec_model, fasttext_model], [FrequencyMode.bow, FrequencyMode.tfidf], None, delim_line='\n', verbosity=1)

corpus_match_paragraphs_target_source(
        wv_models=[word2vec_model, fasttext_model],
        freq_modes = [FrequencyMode.bow, FrequencyMode.tfidf],
        out_dir=OUT_DIR,
        min_paragraph_len=8,
        top_n_match=64)


cmp_file_rankings_kendall(CorpusLevel.book)
cmp_file_rankings_kendall(CorpusLevel.section)







