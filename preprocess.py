from typing import List, Set, Dict, Iterable, Callable, Counter
from collections import Counter
from enum import Enum
import os
from tqdm import tqdm
from language import Language, English
from vocabulary import Vocabulary
from dehyphenator import Dehyphenator
from corpus import Corpus, CorpusLevel, CorpusItem
from corpus_content_filter import ContentType, content_type_filter

from nltk.stem import WordNetLemmatizer


def get_corpus_word_counter(
        corpus_dir:str,
        corpus_file_pattern: str = "*.txt",
        content_type: ContentType = None,
        corpus_word_counter_file_name: str = None,
        min_word_len: int = 1,
        merge_case_forms:bool = True,
        is_word: Callable[[str], bool] = English.can_be_word,
        min_word_count=1,
        language: str = English.name) -> Counter[str]:

    corpus_word_counter = None

    corpus_word_counter_file_path = None
    if corpus_word_counter_file_name is not None:
        corpus_word_counter_file_path = os.path.join(corpus_dir, corpus_word_counter_file_name)
        if os.path.exists(corpus_word_counter_file_path):
            corpus_word_counter = Vocabulary.load_word_counter(corpus_word_counter_file_path)

    if corpus_word_counter is None:
        corpus_word_counter:Counter[str] = Counter()
        corpus = Corpus(corpus_dir, corpus_file_pattern)
        corpus_item_filter = content_type_filter(content_type)
        books_count = len(list(corpus.books_iterator(load_content=False, corpus_item_filter=corpus_item_filter)))
        books_iterator = corpus.books_iterator(load_content=True, corpus_item_filter=corpus_item_filter)
        for book_item in tqdm(books_iterator, desc='Building corpus word counter', total = books_count):
            book_word_counter= Vocabulary.text_word_counter(book_item.text, min_word_len=min_word_len, language=language)
            corpus_word_counter.update(book_word_counter)

        if merge_case_forms or is_word is not None or min_word_count > 1:
            corpus_word_counter = Vocabulary.filter_word_counter(
                corpus_word_counter, merge_case_forms, is_word, min_word_count)

        if corpus_word_counter_file_path is not None:
            corpus_word_counter = Vocabulary.sort_word_counter(corpus_word_counter)
            Vocabulary.save_word_counter(corpus_word_counter, corpus_word_counter_file_path)

    return corpus_word_counter

def preprocess():
    corpus_raw = Corpus('corpus_raw', '*.txt')
    Corpus.build_section_structure('corpus_raw', 'corpus_section_structure.json')

    # for later analysis, where we want to analyze the main texts of the books,
    # remove sections with notes, references, bibliography...
    corpus_filter_refs_dir = 'corpus_filter_refs'
    corpus_raw.corpus_filter_references(corpus_out_dir=corpus_filter_refs_dir)
    Corpus.build_section_structure(corpus_filter_refs_dir, 'corpus_section_structure.json')

    corpus_filter_refs = Corpus(corpus_filter_refs_dir, '*.txt')
    text_transform = (lambda text: Corpus.filter_end_of_paragraph_references(text))
    corpus_filter_refs.corpus_text_transform(corpus_filter_refs_dir, text_transform, "Removing end-of-paragraph references")
    text_transform = (lambda text: Corpus.filter_slanting_quotes(text))
    corpus_filter_refs.corpus_text_transform(corpus_filter_refs_dir, text_transform, "Removing slanting quotation marks")

    # for dehyphenation to work correctly, build an extensive vocabulary, counting as existing any token which occured at least 2 times
    corpus_word_counter = get_corpus_word_counter(
        corpus_filter_refs_dir, '*.txt', None, 'corpus_word_counter.json', merge_case_forms=False, min_word_count=2)

    corpus_dehyphen_dir = 'corpus_dehyphen'

    # restore the words, split between lines with a hyphen, into their original form
    Dehyphenator.dehyphenate_corpus(
                corpus_word_counter = corpus_word_counter,
                corpus_in_dir = corpus_filter_refs_dir,
                corpus_out_dir = corpus_dehyphen_dir,
                dehyphen_log_dir=corpus_dehyphen_dir + '_log',
                corpus_file_pattern='*.txt',
                file_suffixes_to_remove = ['.pdf', '.djvu', '.epub', '.mobi'])

    Corpus.build_section_structure(corpus_dehyphen_dir, 'corpus_section_structure.json')

    # make separate vocabulary for subcorpuses of source and target documents
    wc_source = get_corpus_word_counter(corpus_dehyphen_dir, '*.txt', ContentType.source, 'corpus_word_counter_source.json', min_word_count=2)
    wc_target = get_corpus_word_counter(corpus_dehyphen_dir, '*.txt', ContentType.target, 'corpus_word_counter_target.json', min_word_count=2)

    # united vocabulary for source and target subcorpuses
    corpus_word_counter = Counter()
    corpus_word_counter.update(wc_source)
    corpus_word_counter.update(wc_target)

    # merge the different case forms of a token, replacing with the most frequent one
    corpus_word_counter = Vocabulary.filter_word_counter(
        corpus_word_counter, merge_case_forms=True, is_word = English.can_be_word, min_word_count=2)
    corpus_word_counter = Vocabulary.sort_word_counter(corpus_word_counter)
    Vocabulary.save_word_counter(corpus_word_counter, os.path.join(corpus_dehyphen_dir, 'corpus_word_counter.json'))

    corpus_word_counter = get_corpus_word_counter(corpus_dehyphen_dir, '*.txt', None, 'corpus_word_counter.json')
    # print(f"len(corpus_word_counter)={len(corpus_word_counter)}")

    # convert the corpus to embedded line-sentence format, handy for feeding to gensim library models:
    corpus_dehyphen = Corpus(corpus_dehyphen_dir, "*.txt")

    # not lemmatized
    corpus_line_sentence_dir = 'corpus_line_sentence'
    tokens_filter = (lambda tokens: Vocabulary.tokens_filter(tokens, corpus_word_counter.keys()))
    corpus_dehyphen.corpus_to_line_sentence_format(corpus_line_sentence_dir, tokens_filter)
    Vocabulary.save_word_counter(
        corpus_word_counter,
        os.path.join(corpus_line_sentence_dir, 'corpus_word_counter.json'))

    # lemmatized
    if False:
        corpus_line_sentence_lemmatized_dir = 'corpus_line_sentence_lemmatized'
        wnl = WordNetLemmatizer()
        lemmatizer = (lambda token : wnl.lemmatize(token))
        lemmatized_corpus_word_counter = Vocabulary.lemmatize(corpus_word_counter, lemmatizer)
        tokens_filter = (lambda tokens: Vocabulary.tokens_filter(
            tokens, lemmatized_corpus_word_counter.keys(), lemmatizer=lemmatizer))
        corpus_dehyphen.corpus_to_line_sentence_format(corpus_line_sentence_lemmatized_dir, tokens_filter)
        Vocabulary.save_word_counter(
            lemmatized_corpus_word_counter,
            os.path.join(corpus_line_sentence_lemmatized_dir, 'corpus_word_counter.json'))





preprocess()

