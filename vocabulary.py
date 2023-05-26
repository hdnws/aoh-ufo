from typing import List, Set, Dict, Tuple, Iterable, Callable, Counter
from collections import Counter
import re
from glob import glob
import os
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
import json

from language import Language, English, CaseConversionMode, case_forms

class Vocabulary:

    @staticmethod
    def load_vocab(vocab_file_path:str, separator:str= ' ') -> List[str]:
        """Loads from file the list of known words for a specific language and returns the list"""
        with open(vocab_file_path, 'rt', encoding='utf-8') as fwords:
            content = fwords.read()

        return content.split(separator)

    @staticmethod
    def sort_word_counter(word_counter:Counter[str]):
        """Sorts the items in the word counter so that tokens appear in order of decrease of frequency,
           those with equal frequency ordered alphabetically."""

        key_function = (lambda token_and_count: (-token_and_count[1], token_and_count[0]))
        token_counts_sorted = sorted(word_counter.most_common(), key=key_function)
        return Counter(dict(token_counts_sorted))

    @staticmethod
    def text_word_counter(
            text:str,
            min_word_len:int=1,
            language:str=English.name
            ) -> Counter[str]:
        """Counts words in the given text."""
        word_counter = Counter()
        for sentence in sent_tokenize(text, language=language):
            for word in word_tokenize(sentence, language=language):
                if len(word) >= min_word_len:
                    word_counter.update([word])

        return word_counter

    @staticmethod
    def file_word_counter(
            text_file_path:str,
            min_word_len:int=1,
            language:str=English.name
            ) -> Counter[str]:
        """Counts words in the given text file."""
        with open(text_file_path, 'rt', encoding='utf-8') as ftext:
            text = ftext.read()

        return Vocabulary.text_word_counter(text, min_word_len=min_word_len, language=language)

    @staticmethod
    def files_word_counter(
            file_path_pattern:str,
            min_word_len:int=1,
            language:str=English.name)->Counter[str]:
        """Counts (lowercased) words in a collection of text files."""

        files_word_counter = Counter()
        filepaths = glob(file_path_pattern)
        for filepath in tqdm(filepaths, desc='Vocabulary.files_word_counter'):
            file_word_counter = Vocabulary.file_word_counter(
                text_file_path=filepath, min_word_len=min_word_len, language=language)

            files_word_counter = files_word_counter + file_word_counter

        return files_word_counter

    @staticmethod
    def merge_case_forms(word_counter:Counter[str])->Counter[str]:
        # if a word is present in several case forms, replace with the most frequent one all other forms
        word_counter_merged = Counter()
        processed_case_forms: Set[str] = set()
        for token in word_counter.keys():
            if token not in processed_case_forms:
                token_case_forms = list(set(case_forms(token, CaseConversionMode.any)))
                token_case_forms_counts = list(map(lambda t: word_counter[t], token_case_forms))
                i_max = token_case_forms_counts.index(max(token_case_forms_counts))
                token_cf_max = token_case_forms[i_max]
                sum_counts = sum(token_case_forms_counts)
                word_counter_merged[token_cf_max] = sum_counts
                processed_case_forms |= set(token_case_forms)

        return word_counter_merged

    @staticmethod
    def lemmatize(word_counter:Counter[str], lemmatizer:Callable[[str], str])->Counter[str]:
        """Replace counts of different morphologic forms of a word
           with summed-up count and its signle lemmatized form."""
        if lemmatizer is None:
            return word_counter

        word_counter_lemma = Counter()
        for token in word_counter.keys():
            token_lemma = lemmatizer(token)
            word_counter_lemma[token_lemma] += word_counter[token]

        return word_counter_lemma

    @staticmethod
    def filter_word_counter(word_counter:Counter[str],
                            merge_case_forms:bool=True,
                            is_word:Callable[[str], bool] = None,
                            min_word_count:int=1)->Counter[str]:
        if merge_case_forms:
            word_counter = Vocabulary.merge_case_forms(word_counter)

        filter_fn = (lambda word : is_word is not None and not is_word(word) or word_counter[word] < min_word_count)
        tokens_to_remove:List[str] = list(filter(filter_fn, word_counter.keys()))
        for token in tokens_to_remove:
            del word_counter[token]

        return word_counter

    @staticmethod
    def tokens_filter(
            tokens:Iterable[str],
            vocabulary:Iterable[str]=None,
            case_mode=CaseConversionMode.any,
            split_hyphen=True,
            lemmatizer:Callable[[str],str]=None)->List[str]:
        """Accepts list of tokens, returns a filtered list of tokens (possibly with transformed case and|or lemmatized).
           vocabulary - collection of tokens which can be used on output.
           If an input token is not found in vocabulary, but a different case form of it is, the latter is used for output.
           A not-in-vocabulary input token containig a hyphen '-' is split by hyphen
           and replaced by a sequence of composing it subtokens, found in vocabulary."""
        if lemmatizer is None:
            lemmatizer = (lambda token: token)

        tokens_filtered = []
        for token in tokens:
            found = False
            for token_transformed in map(lemmatizer, case_forms(token, case_mode)):
                if token_transformed in vocabulary:
                    tokens_filtered.append(token_transformed)
                    found = True
                    break

            if not found and '-' in token and split_hyphen:
                tokens_filtered.extend(
                    Vocabulary.tokens_filter(token.split('-'), vocabulary, case_mode, False, lemmatizer))

        return tokens_filtered

    @staticmethod
    def save_word_counter(word_counter:Counter[str], file_path):
        word_counter = dict(word_counter.most_common())
        with open(file_path, mode='wt', encoding='utf-8') as f:
            json.dump(word_counter, f)

    @staticmethod
    def load_word_counter(file_path)->Counter[str]:
        try:
            with open(file_path, mode='rt', encoding='utf-8') as f:
                word_counts:Dict[str, int] = json.load(f)
            return Counter(word_counts)
        except BaseException as exc:
            print(f"load_word_counter('{file_path}') failed: {exc})")
            return None


    @staticmethod
    def vocab_from_word_counter(word_counter:Counter[str], min_word_count=1)->List[str]:
        word_count_tuples:List[Tuple[str,int]] = word_counter.most_common()
        return [word for word, count in word_count_tuples if count >= min_word_count]

    @staticmethod
    def analyse_vocab_case(word_counter_file_path:str):
        word_counter = Vocabulary.load_word_counter(word_counter_file_path)
        print(f"len(word_counter)={len(word_counter)}")
        lc_and_not_lc = []
        not_lc_only = []
        uc_only = []
        lc_set = set()
        for word in word_counter.keys():
            lc_set |= {word.lower()}
            if word.lower() != word:
                if word.lower() in word_counter:
                    lc_and_not_lc.append(word)
                else:
                    not_lc_only.append(word)
                    if word == word.upper() and word.title() not in word_counter:
                        uc_only.append(word)

        print(f"len(lc_set)={len(lc_set)}")
        print(f"len(lc_and_not_lc)={len(lc_and_not_lc)}")
        print(f"len(not_lc_only)={len(not_lc_only)}")
        print(f"len(uc_only)={len(uc_only)}")
        print(f"lc_and_not_lc:\n{lc_and_not_lc}")
        print(f"not_lc_only:\n{not_lc_only}")
        print(f"uc_only:\n{uc_only}")

def test_vocab_from_word_counter():
    word_counter = Vocabulary.files_word_counter(file_path_pattern='corpus_raw\\*.djvu.txt',min_word_len=12)
    print(f"word_counter.most_common(50):\r\n{word_counter.most_common(50)}")
    vocab = Vocabulary.vocab_from_word_counter(word_counter, min_word_count=80)
    print("vocabulary:\r\n", vocab)

def test_merge_case_forms():
    text = "The the the NASA TOP Top Top top F-16"
    tokens = text.split(' ')
    word_counter = Counter(tokens)
    print(f"word_counter: {word_counter}")
    word_counter = Vocabulary.merge_case_forms(word_counter)
    print(f"merge_case_forms(word_counter): {word_counter}")

def test_filter_word_counter(word_counter_file_path:str, min_word_counts:List[int]):
    word_counter = Vocabulary.load_word_counter(word_counter_file_path)
    print(f"len(word_counter)={len(word_counter)}")
    for min_word_count in min_word_counts:
        word_counter_filtered = Vocabulary.filter_word_counter(word_counter, merge_case_forms=True, is_word=None, min_word_count=min_word_count)
        print(f"min_word_count:{min_word_count} -> len(word_counter_filtered)={len(word_counter_filtered)}")

def test_lemmatize_text(file_path_pattern, print_conversions:bool=False):
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.tag import pos_tag

    wnl = WordNetLemmatizer()
    tokens = set()
    lemmas = set()
    conv = dict()

    file_paths = glob(file_path_pattern)
    for file_path in tqdm(file_paths):
        text = ''
        with open(file_path, mode='rt', encoding='utf-8') as f:
            text = f.read()
        for sentence in sent_tokenize(text):
            for word in word_tokenize(sentence):
                if English.can_be_word(word):
                    lc_word = word.lower()
                    lemma = wnl.lemmatize(lc_word)
                    tokens |= {lc_word}
                    lemmas |= {lemma}
                    if lemma != lc_word:
                        conv[lc_word] = lemma
    print(f"len(lc_tokens)={len(tokens)}, len(lemmas)={len(lemmas)}")
    if print_conversions:
        print(conv)

def lemmatize_with_pos():
    from nltk.tag import pos_tag
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    wnl = WordNetLemmatizer()

    pos_map = {'NN':'n', 'VB':'v', 'VBP':'v', 'VBD':'v', 'VBN':'v','VBZ':'v', 'VBG':'v', 'JJ':'a', 'RB':'r' }


    sentence = """cat dog man 
                  who whose whom which when where why 
                  A the this these that those there I you he she it they my your yours his her it's their theirs mine 
                  be is are was were been have has had go goes went seek seeks sought teach taught teaching see seen saw look looked swim swimmed swimming keep keeps kept keeping 
                  beautiful dark light rude cold warm eager apt fast slow bitter tough 
                  ugly bitterly coldly heartily slowly eagerly aptly often during 
                  though as if or and
                  of with without on upon up down"""

    word_tags_lemma = []
    for word_and_tag in pos_tag(word_tokenize(sentence)):
        word = word_and_tag[0]
        pos = word_and_tag[1]
        pos_short = pos_map.get(pos, None)
        lemma = wnl.lemmatize(word, pos_short) if pos_short is not None else word
        word_tags_lemma.append((word, pos, pos_short, lemma))

    for tpl in word_tags_lemma:
        print(tpl)

# test_vocab_from_word_counter()
# test_merge_case_forms()
# Vocabulary.analyse_vocab_case('corpus_filter_refs\\corpus_word_counter.json')
# test_filter_word_counter('corpus_filter_refs\\corpus_word_counter.json', [2, 4, 8, 16, 32, 64, 128])

# test_lemmatize_text('corpus_raw\\*The Threat*.txt',print_conversions=True)
# word_counter = Vocabulary.text_word_counter(text)
# word_counter = Vocabulary.file_word_counter('corpus_raw\\Good T. Above Top Secret, 1988.txt')


# import nltk
# nltk.download('averaged_perceptron_tagger')

# lemmatize_with_pos(sentence)

