import dataclasses
import json
from typing import List, Dict, Set, Iterator, Callable, Iterable, Iterator
from collections import namedtuple, Counter
import itertools
from enum import Enum
from dataclasses import dataclass
from glob import glob
import os
import re
from io import StringIO
from language import Language, English, CaseConversionMode, case_forms
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm

class CorpusLevel(Enum):
    """Defines the level of location of a text fragment in the corpus."""
    book = 0
    section = 1
    subsection = 2

# Delimiter chars on the level of book (divides to sections), section (divides to subsections).
# A line filled with 16+ of delim chars splits the text into sections/subsections.
# The next line after delimiter contains the (sub)section title.
TEXT_DELIM_CHARS = ['=','-']

@dataclass(init=True, repr=True)
class CorpusItem:
    """Represents the text content from corpus on a particular level - book/section/subsection."""
    level : CorpusLevel
    file_path: str = None
    title_path : List[str] = None # list of item titles in text hierarchy: [book_title[, section_title[, subsection_title,...]]]
    text: str = None  # text content of a book/section/subsection, not including the title/subtitle

    @staticmethod
    def title_from_path(title_path:List[str]) -> str:
        return '\\'.join(title_path)

    @staticmethod
    def title_to_path(title:str) -> List[str]:
        return title.split('\\')

    def title(self)->str:
        return CorpusItem.title_from_path(self.title_path)


class Corpus:
    """Represents a corpus of book texts as text files,
    internally separated to sections/subsections;
    supports different-level iterations over the content of the corpus texts."""

    def __init__(self,
                 corpus_dir:str,
                 file_pattern:str='*.txt',
                 language_name:str=English.name,
                 encoding='utf-8'):
        self.corpus_dir = corpus_dir
        self.file_pattern = file_pattern
        self.language_name = language_name
        self.encoding = encoding

    @staticmethod
    def remove_file_suffixes(file_path:str, suffixes:List[str]):
        while any(map(lambda suffix: file_path.endswith(suffix), suffixes)):
            for suffix in suffixes:
                file_path = file_path.removesuffix(suffix)
        return file_path

    @staticmethod
    def delim_line(corpus_level:CorpusLevel) -> str:
        assert CorpusLevel.book.value <= corpus_level.value and corpus_level.value <= CorpusLevel.section.value
        return TEXT_DELIM_CHARS[corpus_level.value] * 64 + '\n'


    @staticmethod
    def split_to_sections(corpus_item:CorpusItem)->List[CorpusItem]:
        """Splits the text of a given CorpusItem (a book|section) into (sub)sections,
           delimited with a line, matching the given pattern (pattern includes ending newline char).
           The text of the line following the delimiter line is taken as section title.
           Returns the list of CorpusItem elements, containing title and content(includes title) of each [sub]section.
           If delimiter line is not found, returns None."""
        assert CorpusLevel.book.value <= corpus_item.level.value and corpus_item.level.value <= CorpusLevel.subsection.value
        if corpus_item.level == CorpusLevel.subsection:
            return None # no more level than subsections supported
        delim_char:str = TEXT_DELIM_CHARS[corpus_item.level.value]
        re_delim_line:str = '^' + delim_char + '{16,}\n'
        text = corpus_item.text
        if text is None or len(text) == 0 or re.search(re_delim_line, text, flags=re.MULTILINE) is None:
            return None

        section_items:List[CorpusItem] = []
        sections_content:List[str] = re.split(re_delim_line, text, flags=re.MULTILINE)
        for i_section, section_content in enumerate(sections_content):
            title = ''
            if i_section > 0:
                title = None
                title_end = section_content.find('\n')
                if title_end>0:
                    title = section_content[:title_end].strip()
                    section_content = section_content[(title_end+1):]
            if title is not None and len(section_content.strip()) != 0:
                level = CorpusLevel(corpus_item.level.value + 1)
                title_path = corpus_item.title_path + [title]
                section_item = dataclasses.replace(
                    corpus_item,
                    level=level,
                    title_path=title_path,
                    text=section_content)

                section_items.append(section_item)

        return section_items

    @staticmethod
    def split_to_sentences(text:str, language:str=English.name) -> List[str]:
        return sent_tokenize(text, language=language)

    @staticmethod
    def split_to_paragraphs_of_sentences(text:str, language:str=English.name) -> List[List[str]]:
        """Splits texts to paragraphs, constisting of sentences.
        Each element of the returned list, List[str], represents a paragraph,
        its string elements are paragraph sentences (newlines removed from them, if any)."""
        sentences:List[str] = Corpus.split_to_sentences(text, language=language)

        paragraphs:List[List[str]] = [] # list of paragraphs
        paragraph = []  # list of sentence strings in the current paragraph
        text_pos = 0
        for i_sentence, sentence in enumerate(sentences):
            sentence_start = text.find(sentence, text_pos)
            assert sentence_start != -1
            if text[text_pos:sentence_start].find('\n') != -1:
                # newline char(s) between previous and this sentence
                # thus this sentence starts a new paragraph
                if paragraph:
                    paragraphs.append(paragraph)
                    paragraph = []

            sentence_cleaned = re.sub(' *\n *', ' ', sentence, flags=re.MULTILINE)
            paragraph.append(sentence_cleaned)
            text_pos = sentence_start + len(sentence)
        if len(paragraph) != 0:
            paragraphs.append(paragraph)
        return paragraphs

    def read_book(self, book_file_path:str):
        with open(book_file_path, mode='rt', encoding=self.encoding) as f:
            return f.read()

    def books_iterator(self,
                       load_content:bool=False,
                       corpus_item_filter:Callable[[CorpusItem], bool]=None
                       )->Iterator[CorpusItem]:
        corpus_file_path_pattern = os.path.join(self.corpus_dir, self.file_pattern)
        for file_path in glob(corpus_file_path_pattern):
            file_name:str = os.path.split(file_path)[1]
            book_title = os.path.splitext(file_name)[0]
            content = self.read_book(file_path) if load_content else None

            book_item = CorpusItem(
                level=CorpusLevel.book,
                file_path=file_path,
                title_path=[book_title],
                text=content)

            if corpus_item_filter is not None and not corpus_item_filter(book_item):
                continue

            yield book_item

    def find_book(self, title_substring:str, load_content:bool=False):
        for book_item in self.books_iterator(load_content=False):
            if book_item.title().lower().find(title_substring.lower()) != -1:
                if load_content:
                    book_item.text = self.read_book(book_item.file_path)
                return book_item

        return None

    def sections_iterator(
            self,
            corpus_item_filter:Callable[[CorpusItem], bool]=None
            )->Iterator[CorpusItem]:
        for book_item in self.books_iterator(load_content=True, corpus_item_filter=corpus_item_filter):
            section_items: List[CorpusItem]  = Corpus.split_to_sections(book_item)
            if section_items is not None:
                for section_item in section_items:
                    if corpus_item_filter is None or corpus_item_filter(section_item):
                        yield section_item

    def subsections_iterator(
            self,
            corpus_item_filter: Callable[[CorpusItem], bool] = None
            )->Iterator[CorpusItem]:
        for section_item in self.sections_iterator(corpus_item_filter=corpus_item_filter):
            subsection_items: List[CorpusItem]  = Corpus.split_to_sections(section_item)
            if subsection_items is not None:
                for subsection_item in subsection_items:
                    if corpus_item_filter is None or corpus_item_filter(subsection_item):
                        yield subsection_item

    def corpus_item_iterator(
            self,
            level:CorpusLevel=CorpusLevel.book,
            corpus_item_filter: Callable[[CorpusItem], bool] = None,
            load_content:bool=True)->Iterator[CorpusItem]:
        if level == CorpusLevel.book:
            return self.books_iterator(load_content=load_content, corpus_item_filter=corpus_item_filter)
        elif level == CorpusLevel.section:
            return self.sections_iterator(corpus_item_filter=corpus_item_filter)
        elif level == CorpusLevel.subsection:
            return self.subsections_iterator(corpus_item_filter=corpus_item_filter)
        else:
            return None

    def find_item(self, title_path_substrings:List[str], load_content:bool=True) -> CorpusItem:
        assert 1 <= len(title_path_substrings)
        assert len(title_path_substrings) <= (CorpusLevel.subsection.value+1)
        load_content |= (len(title_path_substrings) > 1)
        corpus_item = self.find_book(title_path_substrings[0], load_content)
        while corpus_item is not None and len(corpus_item.title_path) < len(title_path_substrings):
            child_items:List[CorpusItem] = Corpus.split_to_sections(corpus_item)
            if child_items is None: return None
            corpus_item = next(iter(filter(
                lambda ci: title_path_substrings[ci.level.value].lower() in ci.title_path[ci.level.value].lower(),
                child_items)))
        return corpus_item



    def build_corpus_section_structure(
            self,
            corpus_section_structure_file_path:str=None
            )->Dict[str, Dict[str, Dict[str, str]]]:
        """Returns a mapping from the book title
           to the mapping of section title to either None or
           to the mapping of subsection title to None"""
        corpus_section_structure:Dict[str, List[str]] = dict()
        book_count = sum([1 for book_item in self.books_iterator(load_content=False)])
        for book_item in tqdm(self.books_iterator(load_content=True), total=book_count, desc='Loading corpus sections...'):
            section_items:List[CorpusItem] = Corpus.split_to_sections(book_item)
            section_titles = None
            if section_items is not None:
                section_titles = dict(map(lambda section_item: (section_item.title_path[-1], None), section_items))
                for section_item in section_items:
                    subsection_items:List[SectionItem] = Corpus.split_to_sections(section_item)
                    if subsection_items is not None:
                        subsection_titles = dict(map(lambda subsection_item: (subsection_item.title_path[-1], None), subsection_items))
                        section_titles[section_item.title_path[-1]] = subsection_titles

            corpus_section_structure[book_item.title_path[0]] = section_titles

        if corpus_section_structure_file_path is not None:
            with open(corpus_section_structure_file_path, mode='wt', encoding=self.encoding) as f:
                json.dump(corpus_section_structure, f, indent=4)

        return corpus_section_structure

    @staticmethod
    def build_section_structure(
            corpus_dir:str,
            corpus_section_structure_file_name:str='corpus_section_structure.json'):

        corpus = Corpus(corpus_dir=corpus_dir, file_pattern='*.txt')
        corpus_section_structure_file_path = None
        if corpus_section_structure_file_name is not None:
            corpus_section_structure_file_path = os.path.join(corpus.corpus_dir, corpus_section_structure_file_name)
        return corpus.build_corpus_section_structure(corpus_section_structure_file_path)

    @staticmethod
    def book_filter_sections(book_item:CorpusItem, title_filter:Dict[str, bool], language:str=English.name) -> str:
        """Removes from the book text sections/subsections according to filter, applied to (sub)section title.
        title_filter maps the title keyword, with which to exclude (sub)sections,
        into the match-whole-title mode; case is ignored."""

        def filter_match(title:str, title_filter:Dict[str, bool], language:str=English.name):
            match = False
            for keyword, match_whole in title_filter.items():
                if match_whole:
                    match |= (keyword.lower() == title.lower())
                else:
                    match |= (keyword.lower() in word_tokenize(title.lower(), language=language))
            return match

        text_out = StringIO()
        section_items: List[CorpusItem] = Corpus.split_to_sections(book_item)
        if section_items is None:
            text_out.write(book_item.text)
        else:
            for section_item in section_items:
                if not filter_match(section_item.title_path[-1], title_filter, language):
                    if section_item.title_path[-1] != '':
                        text_out.write(Corpus.delim_line(book_item.level))
                        text_out.write(section_item.title_path[-1] + '\n')
                    subsection_items: List[CorpusItem] = Corpus.split_to_sections(section_item)
                    if subsection_items is None:
                        text_out.write(section_item.text)
                        text_out.write('\n')
                    else:
                        for subsection_item in subsection_items:
                            if not filter_match(subsection_item.title_path[-1], title_filter, language):
                                if subsection_item.title_path[-1] != '':
                                    text_out.write(Corpus.delim_line(section_item.level))
                                    text_out.write(subsection_item.title_path[-1] + '\n')
                                text_out.write(subsection_item.text)
                                text_out.write('\n')

        return text_out.getvalue()

    def corpus_filter_references(self,
                                 corpus_out_dir:str):
        # keyword to whole-title-match-mode
        title_filter = {
            'index':False,
            'bibliography':False,
            'reading list':True,
            'notes':False,
            'endnotes': True,
            'footnotes':True,
            'references':True,
            'sources':True,
            'acknowledgements':True,
            'acknowledgments':True,
            'about the author':True
        }
        if not os.path.exists(corpus_out_dir):
            os.mkdir(corpus_out_dir)

        book_count = sum([1 for book_item in self.books_iterator(load_content=False)])
        for book_item in tqdm(self.books_iterator(load_content=True), total=book_count, desc='Filtering out references'):
            book_text_filtered = Corpus.book_filter_sections(book_item, title_filter, self.language_name)
            file_name = os.path.split(book_item.file_path)[1]
            out_file_path = os.path.join(corpus_out_dir, file_name)
            with open(out_file_path, mode='wt', encoding=self.encoding) as f:
                f.write(book_text_filtered)

    @staticmethod
    def book_text_transform(
            book_item: CorpusItem,
            out_file_path:str,
            text_transform: Callable[[str],str],
            encoding='utf-8')->str:
        """Converts text of each leaf corpus item (which has no child sections/subsections)
        using given transform function.
        To transform function is passed the section/subsection text, not including the section/subsection title line;
        the section/subsection delimiter and the section/subsection title line
        are written to output before the text_transform output.
        So the book text remains delimited into sections/subsections with '^=+\n' and '^-+\n' lines,
        followed by (sub)section title lines."""
        if book_item.text is None:
            raise ValueError('Book text is None')

        text_out = StringIO()
        section_items: List[CorpusItem] = Corpus.split_to_sections(book_item)
        if section_items is None:
            text_converted = text_transform(book_item.text)
            text_out.write(text_converted)
        else:
            for section_item in section_items:
                if section_item.title_path[-1] !='':
                    text_out.write(Corpus.delim_line(book_item.level))

                text_out.write(section_item.title_path[-1] + '\n')
                subsection_items:List[CorpusItem] = Corpus.split_to_sections(section_item)
                if subsection_items is None:
                    text_converted = text_transform(section_item.text)
                    text_out.write(text_converted)
                    text_out.write('\n')
                else:
                    for subsection_item in subsection_items:
                        if subsection_item.title_path[-1] != '':
                            text_out.write(Corpus.delim_line(section_item.level))
                        text_out.write(subsection_item.title_path[-1] + '\n')
                        text_converted = text_transform(subsection_item.text)
                        text_out.write(text_converted)
                        text_out.write('\n')
        with open(out_file_path, mode='wt', encoding=encoding) as fout:
            fout.write(text_out.getvalue())

    def corpus_text_transform(
            self,
            corpus_out_dir:str,
            text_transform: Callable[[str],str],
            progress_title:str
            )->str:
        """Transforms the texts of all corpus leaf items (having no child [sub]sections)
           using given text_transform."""
        if not os.path.exists(corpus_out_dir):
            os.mkdir(corpus_out_dir)
        book_count = sum([1 for book_item in self.books_iterator(load_content=False)])
        for book_item in tqdm(self.books_iterator(load_content=True), desc=progress_title, total=book_count):
            file_name = os.path.split(book_item.file_path)[1]
            out_file_path = os.path.join(corpus_out_dir, file_name)
            Corpus.book_text_transform(book_item, out_file_path, text_transform, self.encoding)

    @staticmethod
    def filter_end_of_paragraph_references(text:str) -> str:
        """Removes the numerical references at the paragraph end, such as ". 6" or "? [6]"
        If left, these references affect tokenization to sentences and paragraphs,
        as the reference number gets adjoined to the first sentence of the next paragraph
        and the paragraphs do not get split."""
        end_paragraph_ref_pattern1 = "([\.!\?’”\'\"]+) *\[\d+\] *(( *\n *)+)"
        end_paragraph_ref_pattern2 = "([\.!\?’”\'\"]+) *\d+ *(( *\n *)+)"
        re_pattern1 = re.compile(end_paragraph_ref_pattern1)
        re_pattern2 = re.compile(end_paragraph_ref_pattern2)
        text = re_pattern1.sub(r"\1\2", text)
        text = re_pattern2.sub(r"\1\2", text)
        return text

    @staticmethod
    def filter_slanting_quotes(text:str) -> str:
        """Filter out the slope quotation marks, as it negatively affects NLTK tokenization to sentences."""

        # remove double slanting quotation marks
        re_pattern = re.compile("[“”]")
        text = re_pattern.sub('', text)

        # replace slanting single quotation mark, opening or closing, "‘’" with a simple vertical one "'"
        re_pattern = re.compile("[‘’]")
        text = re_pattern.sub("'", text)

        return text

    @staticmethod
    def text_to_line_sentence_format(
            text: str,
            tokens_filter: Callable[[Iterable[str]],Iterable[str]] = None,
            language: str = English.name) -> str:
        """Converts the original raw text to gensim's line-sentence format:
        text is split to paragraphs/sentences/tokens using NLTK sent_tokenize, word_tokenize;
        token_filter is applied to sequence of sentence tokens, returning filtered sequence of tokens;
        paragraphs are delimited from each other by '\n\n', sentences by '\n', tokens by ' ', punctuation is removed."""
        text_out = StringIO()
        if text is None: return None
        paragraphs:List[List[str]] = Corpus.split_to_paragraphs_of_sentences(text, language)
        for paragraph in paragraphs:
            sentence_filtered_count = 0
            for sentence in paragraph: # sentence : str
                tokens:List[str] = word_tokenize(sentence, language=language)
                tokens_filtered = tokens_filter(tokens) if tokens_filter is not None else tokens
                sentence_filtered = ' '.join(tokens_filtered)
                if len(sentence_filtered):
                    text_out.write(sentence_filtered + '\n')
                    sentence_filtered_count += 1
            if sentence_filtered_count != 0:
                text_out.write('\n')
        return text_out.getvalue()

    @staticmethod
    def line_sentence_text_to_paragraphs(text:str)->List[List[List[str]]]:
        """Convert the given text in the line-sentence format into a list of paragraphs;
        where a paragraph is a list of sentences
        and a sentence is a list of tokens."""
        # split to paragraphs/sentences/tokens
        paragraphs:List[List:List[str]] = []
        paragraph:List[List[str]] = []
        for line in (text+'\n').splitlines():
            line = line.strip()
            if len(line) != 0:
                tokens = line.split(' ')
                paragraph.append(tokens) # add a sentence
            elif len(paragraph) != 0:
                paragraphs.append(paragraph)
                paragraph = []
        return paragraphs

    @staticmethod
    def get_tokenized_sentences_text(
            text:str,
            sentences_tokens:List[List[str]],
            tokens_filter: Callable[[Iterable[str]], Iterable[str]],
            language:str=English.name)->List[str]:
        """Given the original text and some sentences from it as lists of tokens,
        returns almost-original texts of those tokenzed sentences as list of strings
        (the possible change is that newlines inside sentence strings are removed).
        text - original text, from which sentences as lists of tokens have been made.
        line_sentences - some sentences from the text in the form of lists of tokens.
        tokens_filter - filter which was used for convertion of original text to tokenized sentences.
        language - language name for NLTK tokenizers."""
        text_paragraphs:List[List[str]] = Corpus.split_to_paragraphs_of_sentences(text, language)
        text_sentences:List[str] = list(itertools.chain.from_iterable(text_paragraphs))
        joinedtokens_to_sentence:Dict[str, str] = dict()
        for sentence_text in text_sentences:
            sentence_tokenenized:str = Corpus.text_to_line_sentence_format(
                sentence_text, tokens_filter, language).strip(' \n')
            joinedtokens:str = ''.join(sentence_tokenenized.split(' '))
            joinedtokens_to_sentence[joinedtokens] = sentence_text

        sentences:List[str] = []
        for sentence_tokens in sentences_tokens:
            joinedtokens = ''.join(sentence_tokens)
            sentence_text_default = ' '.join(sentence_tokens)
            sentence_text = joinedtokens_to_sentence.get(joinedtokens, sentence_text_default)
            sentences.append(sentence_text)

        return sentences

    def find_tokenized_sentences_text(
            self,
            text_title_path:List[str],
            sentences_tokens:List[List[str]],
            tokens_filter: Callable[[Iterable[str]], Iterable[str]],
            language:str=English.name)->List[str]:
        corpus_item:CorpusItem = self.find_item(text_title_path)
        if corpus_item is None: return None
        return Corpus.get_tokenized_sentences_text(
            corpus_item.text, sentences_tokens, tokens_filter, self.language_name)

    @staticmethod
    def book_to_line_sentence_format(
            book_item: CorpusItem,
            out_file_path:str,
            tokens_filter: Callable[[Iterable[str]],Iterable[str]] = None,
            language: str = English.name,
            encoding='utf-8')->str:
        """Converts the book text to 'embedded' line-sentence format of gensim,
        for later efficient processing (because of sentence/word tokenization takes much time otherwise).
        The book text remains delimited into sections/subsections with '^=+\n' and '^-+\n' lines,
        followed by (sub)section title lines.
        The original text of (sub)section (including title line) is split to paragraphs/sentences/tokens
        by the method text_to_line_sentence_format."""
        if book_item.text is None:
            raise ValueError('Book text is None')

        text_transform = (lambda text: Corpus.text_to_line_sentence_format(text, tokens_filter, language))
        Corpus.book_text_transform(book_item, out_file_path, text_transform, encoding)

    def corpus_to_line_sentence_format(
            self,
            corpus_out_dir:str,
            tokens_filter: Callable[[Iterable[str]], Iterable[str]] = None):
        """Converts all the corpus books to embedded line-sentence format."""
        text_transform = (lambda text: Corpus.text_to_line_sentence_format(text, tokens_filter, self.language_name))
        self.corpus_text_transform(corpus_out_dir, text_transform, 'Converting to line-sentence format')

    def corpus_item_leaf_iterator(
            self,
            root_corpus_item:CorpusItem=None,
            corpus_item_filter: Callable[[CorpusItem], bool] = None
            )->Iterable[CorpusItem]:
        """Iterates over corpus items under root_corpus_item, which are leaf nodes (has no child sections/subsections)"""
        corpus_item_iterator = None
        if root_corpus_item is not None:
            corpus_item_iterator = iter([root_corpus_item])
        else:
            corpus_item_iterator = self.books_iterator(load_content=True, corpus_item_filter=corpus_item_filter)

        for corpus_item in corpus_item_iterator:
            child_items:List[CorpusItem] = Corpus.split_to_sections(corpus_item)
            if child_items is None:
                if corpus_item_filter is None or corpus_item_filter(corpus_item):
                    yield corpus_item
            else:
                for child_item in child_items:
                    for rec_res_item in self.corpus_item_leaf_iterator(child_item):
                        if corpus_item_filter is None or corpus_item_filter(rec_res_item):
                            yield rec_res_item


    def sentences_of_tokens_iterator(
            self,
            root_corpus_item:CorpusItem=None,
            corpus_item_filter:Callable[[CorpusItem], bool] = None
            )->Iterable[List[str]]:
        """For the corpus, stored in the embedded sentence-line format:
        iterates over sentences of the text(s) of a corpus_item (from entire corpus, if corpus_item is None)
        returning an iterable of lists of tokens, accepted by gensim's models word2vec, fasttext.
        Each list represents tokens of a single sentence."""
        for corpus_leaf_item in self.corpus_item_leaf_iterator(root_corpus_item, corpus_item_filter):
            if not corpus_leaf_item.text: continue
            lines = corpus_leaf_item.text.splitlines()
            for line in lines:
                line = line.strip()
                if len(line) != 0:
                    sentence_tokens = line.split(' ')
                    yield sentence_tokens

    # end class Corpus

def test_books_iterator(title_substring:str):
    corpus_item_filter = (lambda corpus_item: title_substring.lower() in corpus_item.title_path[0].lower())
    corpus = Corpus(corpus_dir='corpus_raw', file_pattern='*.txt')
    for book_item in corpus.books_iterator(load_content=False, corpus_item_filter=corpus_item_filter):
        print(book_item)

def test_sections_iterator(section_title_substring:str,
        corpus_dir:str='corpus_raw'):
    corpus_item_filter = (lambda corpus_item:
                          corpus_item.level == CorpusLevel.book or section_title_substring.lower() in corpus_item.title_path[1].lower())
    corpus = Corpus(corpus_dir=corpus_dir, file_pattern='*.txt')
    n = 0
    for section_item in corpus.sections_iterator(corpus_item_filter=corpus_item_filter):
        print(section_item.title_path)
        n+=1
    print(f"section_count={n}")

def test_subsections_iterator(subsection_title_substring:str):
    corpus_item_filter = (lambda corpus_item:
                          (corpus_item.level.value < CorpusLevel.subsection.value or
                           subsection_title_substring.lower() in corpus_item.title_path[2].lower()) )
    corpus = Corpus(corpus_dir='corpus_raw', file_pattern='*.txt')
    n = 0
    for section_item in corpus.subsections_iterator(corpus_item_filter=corpus_item_filter):
        print(section_item.title_path)
        n+=1
    print(f"section_count={n}")

def test_find_item():
    corpus = Corpus(corpus_dir='corpus_raw', file_pattern='*.txt')
    corpus_item:CorpusItem = corpus.find_item(['need to know','serious business'], load_content=True)
    print(corpus_item)

def test_split_to_sections(to_subsections:bool=False):
    corpus = Corpus(corpus_dir='corpus_raw', file_pattern='*.txt')
    book_item:CorpusItem = corpus.find_book('need to know', load_content=True)
    print(f"book_item.title()='{book_item.title()}'")
    section_items:List[CorpusItem] = Corpus.split_to_sections(book_item)

    print(f"len(section_items)={len(section_items)}")
    for section_item in section_items:
        print(f"section_title: {section_item.title_path[-1]}")
        section_exerpt = section_item.text[:100] + '...'
        print(f"section_exerpt:\n{section_exerpt}", )

def test_build_corpus_section_structure(corpus_dir:str):
    corpus = Corpus(corpus_dir=corpus_dir, file_pattern='*.txt')
    corpus_section_structure_file_path = os.path.join(corpus.corpus_dir, 'corpus_section_structure.json')
    corpus.build_corpus_section_structure(
        corpus_section_structure_file_path=corpus_section_structure_file_path)

def test_split_to_sentences():
    corpus = Corpus(corpus_dir='corpus_raw', file_pattern='*.txt')
    book_item:CorpusItem = corpus.find_book('Catoe', load_content=True)
    text = book_item.text

    text_lines = text.split('\n')
    print(f"len(text_lines)={len(text_lines)}")

    text_subset = '\n'.join(text_lines[3000:3100])
    sentences = Corpus.split_to_sentences(text_subset)
    for i, sentence in enumerate(sentences):
        sentence = sentence.replace('\n',' ')
        print(f"{i}: {sentence}")

def test_split_to_paragraphs_of_sentences():
    corpus = Corpus(corpus_dir='corpus_dehyphen', file_pattern='*.txt')
    section_item:CorpusItem = corpus.find_item(['viruses','Chapter 2'])
    if section_item:
        paragraphs:List[List[str]] = Corpus.split_to_paragraphs_of_sentences(
            Corpus.filter_slanting_quotes(section_item.text))
        for i, pararaph in enumerate(paragraphs):
            print(f"{i}: {pararaph}")

def test_text_to_line_sentence_format(min_word_count:int=1,
                                      case_mode:CaseConversionMode=CaseConversionMode.any,
                                      lemmatize=False):
    from vocabulary import Vocabulary
    from nltk.stem import WordNetLemmatizer
    wnl = WordNetLemmatizer()
    lemmatizer =  (lambda token: wnl.lemmatize(token)) if lemmatize else None

    corpus = Corpus(corpus_dir='corpus_dehyphen', file_pattern='*.txt')
    section_item:CorpusItem = corpus.find_item(['above','17. above'], load_content=True)
    print(f"section_item.title()='{section_item.title()}'")

    text = section_item.text

    word_counter_file_path = os.path.join(corpus.corpus_dir, 'corpus_word_counter.json')
    word_counter = Vocabulary.load_word_counter(word_counter_file_path)
    word_counter = Vocabulary.lemmatize(word_counter, lemmatizer=lemmatizer)
    word_counter = Vocabulary.filter_word_counter(word_counter, None, min_word_count=min_word_count)
    tokens_filter = (lambda tokens: Vocabulary.tokens_filter(
        tokens, word_counter.keys(), case_mode=case_mode, lemmatizer=lemmatizer))
    text_sentence_line = Corpus.text_to_line_sentence_format(text, tokens_filter, English.name)
    print(text_sentence_line)

def test_book_to_line_sentence_format(
        min_word_count:int=1,
        case_mode:CaseConversionMode=CaseConversionMode.any,
        lemmatize=False):
    from vocabulary import Vocabulary
    from nltk.stem import WordNetLemmatizer
    wnl = WordNetLemmatizer()
    lemmatizer =  (lambda token: wnl.lemmatize(token)) if lemmatize else None

    corpus = Corpus(corpus_dir='corpus_dehyphen', file_pattern='*.txt')
    book_item:BookItem = corpus.find_book('above', load_content=True)
    print(f"book_item.title()='{book_item.title()}'")
    out_file_name = book_item.title() + '.SL' + ('.lemma' if lemmatize else '')
    out_file_path = os.path.join(corpus.corpus_dir, out_file_name)

    word_counter = Vocabulary.load_word_counter(os.path.join(corpus.corpus_dir, 'corpus_word_counter.json'))
    word_counter = Vocabulary.lemmatize(word_counter, lemmatizer=lemmatizer)
    word_counter = Vocabulary.filter_word_counter(
        word_counter, merge_case_forms=True, is_word=None, min_word_count=min_word_count)
    tokens_filter = (lambda tokens: Vocabulary.tokens_filter(
        tokens, word_counter.keys(), case_mode=case_mode, lemmatizer=lemmatizer))
    Corpus.book_to_line_sentence_format(book_item, out_file_path, tokens_filter, English.name)
    print(out_file_path)

def test_find_tokenized_sentences_text():
    from vocabulary import Vocabulary
    word_counter = Vocabulary.load_word_counter(os.path.join('corpus_dehyphen','corpus_word_counter.json'))
    corpus_dh = Corpus('corpus_dehyphen', '*.txt')
    corpus_ls = Corpus('corpus_line_sentence', '*.txt')
    title_path_substrings = ['above top','introduction']
    corpus_ls_item:CorpusItem = corpus_ls.find_item(title_path_substrings)
    if corpus_ls_item is None:
        raise ValueError('Item {0} not found'.format(CorpusItem.title_from_path(title_path_substrings)))
    corpus_ls_tokenized_sentences:List[List[str]] = list(corpus_ls.sentences_of_tokens_iterator(corpus_ls_item))
    tokens_filter = (lambda tokens: Vocabulary.tokens_filter(tokens, word_counter.keys()) )
    sentences:List[str] = corpus_dh.find_tokenized_sentences_text(
        corpus_ls_item.title_path, corpus_ls_tokenized_sentences, tokens_filter)
    assert len(sentences) == len(corpus_ls_tokenized_sentences)
    for i in range(len(sentences)):
        assert sentences[i] != ' '.join(corpus_ls_tokenized_sentences[i])
        print(f"{corpus_ls_tokenized_sentences[i]} -> {sentences[i]}")

def test_corpus_item_leaf_iterator(book_title_substring:str=None, section_title_substring:str=None):
    corpus = Corpus('corpus_raw', '*.txt')
    root_corpus_item = None
    if book_title_substring is not None:
        root_corpus_item = corpus.find_book(book_title_substring, load_content=True)
    if section_title_substring is not None:
        corpus_item_filter = (
            lambda corpus_item:
            corpus_item.level.value < CorpusLevel.section.value or section_title_substring.lower() in ' '.join(corpus_item.title_path[1:3]).lower())
    for corpus_leaf_item in corpus.corpus_item_leaf_iterator(root_corpus_item, corpus_item_filter):
        print(corpus_leaf_item.title_path)

def test_sentences_of_tokens_iterator(
        file_name_pattern:str='*.txt',
        sentence_start:int=None,
        sentence_end:int=None):
    corpus = Corpus('corpus_line_sentence', file_name_pattern)
    sentences_of_tokens = list(corpus.sentences_of_tokens_iterator())
    sentence_count = len(sentences_of_tokens)
    print(f"file_name_pattern={file_name_pattern}")
    print(f"sentence_count={sentence_count}")

    if sentence_start is not None and sentence_end is not None:
        for i_sentence in range(max(0,sentence_start), min(sentence_end, sentence_count)):
            print(f"{i_sentence}: {sentences_of_tokens[i_sentence]}")

def test_line_sentence_text_to_paragraphs(title_path_substrings:List[str]):
    corpus = Corpus('corpus_line_sentence')
    corpus_item:CorpusItem = corpus.find_item(title_path_substrings)
    if corpus_item:
        paragraphs:List[List[List[str]]] = Corpus.line_sentence_text_to_paragraphs(corpus_item.text)
        print(f"paragraph_count={len(paragraphs)}")
        sentence_count = sum([len(paragraph) for paragraph in paragraphs])
        print(f"sentence_count={sentence_count}")

        sentences:List[List[str]] = list(corpus.sentences_of_tokens_iterator(corpus_item))
        assert len(sentences) == sentence_count

        i_sentence = 0
        for i_paragraph, paragraph in enumerate(paragraphs):
            print(f"{i_paragraph}:")
            for sentence_tokens in paragraph:
                assert sentence_tokens == sentences[i_sentence]
                i_sentence += 1
                print(' '.join(sentence_tokens))
            print('')

def test_count_paragraphs(corpus_dir:str):
    corpus = Corpus(corpus_dir, "*.txt")
    paragraph_count = 0
    for corpus_item in corpus.corpus_item_leaf_iterator():
        paragraph_count += len(Corpus.line_sentence_text_to_paragraphs(corpus_item.text))
    print(f"{corpus_dir}: paragraph_count={paragraph_count}")

def test_remove_end_of_paragraph_refs(out_dir='corpus_filter_refs_pe'):
    corpus = Corpus('corpus_dehyphen', "*.txt")
    text_transform = (lambda text: Corpus.filter_end_of_paragraph_references(text))
    corpus.corpus_text_transform(out_dir, text_transform, "Removing end-of-paragraph references")

# test_books_iterator('science')
# test_sections_iterator('','corpus_dehyphen')
# test_subsections_iterator('notes')
# test_find_item()
# test_split_to_sections()
# test_split_to_sentences()
# test_split_to_paragraphs_of_sentences()
# test_text_to_line_sentence_format(min_word_count=4, case_mode=CaseConversionMode.lower, lemmatize=False)
# test_book_to_line_sentence_format(min_word_count=4, case_mode=CaseConversionMode.any, lemmatize=False)
# test_find_tokenized_sentences_text()
# corpus_structure = Corpus.build_section_structure('corpus_raw')
# test_corpus_item_leaf_iterator(None, 'notes')
# test_sentences_of_tokens_iterator("Wood*.txt", 3000, 3020)
# test_line_sentence_text_to_paragraphs(['Wood','roswell'])
# test_count_paragraphs('corpus_line_sentence')
# for corpus_dir_suffix in ['word2vec_bow', 'fasttext_bow', 'word2vec_tfidf', 'fasttext_tfidf']:
#     test_count_paragraphs('corpus_line_sentence_pg_merged_'+corpus_dir_suffix)
# test_remove_end_of_paragraph_refs()

