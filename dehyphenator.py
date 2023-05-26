# The module dehyphenate provides the method to join in the text the words, which have been split into parts over line end, with or without hyphen.

from typing import List, Set, Dict, Iterable, Callable, Counter
from collections import Counter
from io import StringIO
from language import Language, English
from corpus import Corpus
import os
from tqdm import tqdm
from glob import glob

class Dehyphenator:
    """The class to restore in the text the words, which were split over the new line."""

    def __init__(self,
                 corpus_word_counter:Counter[str]=[],
                 language:Language=English,
                 hyphen:str='-',
                 white_space_chars:str=" \t",
                 new_line_sep:str="\r\n",
                 word_rhs_adj_punct="\"'’”),:;.!?"):

        self.word_counter:Counter[str] = Counter(corpus_word_counter)

        self.language:Language = language
        self.hyphen = hyphen
        self.white_space_chars = white_space_chars
        self.new_line_sep = new_line_sep

        # punctuation characters, which, if following a hyphenated word,
        # should follow the restored (dehyphenated) word on its line,
        # and not be left on the next line.
        self.word_rhs_adj_punct = word_rhs_adj_punct


    def is_known_word(self, word:str)->bool:
        return (word in self.word_counter) or (word.lower() in self.word_counter)

    def get_word_count(self, word:str)->int:
        return self.word_counter[word] if word in self.word_counter else self.word_counter[word.lower()]

    def contains_vowel(self, word:str)->bool:
        return any(map(lambda c: c in self.language.lc_vowels, word.lower()))

    def is_split_word(self, token1:str, token2:str) -> bool:
        """
        The function is_split_word determines whether the tokens token1 and token2 together represent a known word, split into these tokens.

        token1, token2 - the supposed left and right parts of a split word, containing only alphabetical chars,
        token1 possibly ending with a hyphen '-'.

        is_known_word -  a function parameter, which for the given string returns
        whether it is a known general language word (general here means - not a proper name).

        contains_vowel - a function parameter, which for the given string returns if it contains any vowel sounds.

        The criteria is the following:
            IF
                token1 and token2 combined together give a known word,
                AND ( (token1 OR token2) is not a known word OR token1 ends with hyphen)
                AND token1 and token2 each contain at least one vowel
            THEN
                (token1, token2) is a word split.
        """

        token1nohyphen = (token1[:-1] if token1.endswith(self.hyphen) else token1)

        return self.is_known_word(token1 + token2)                          \
            and ( not(self.is_known_word(token1)) or not(self.is_known_word(token2)) or token1.endswith(self.hyphen) ) \
            and self.contains_vowel(token1) and self.contains_vowel(token2)

    def get_split_tokens(
        self,
        line1:str, # line of text, without newline chars
        line2:str, # the next to it line of text, without trailing newline chars
        ) -> tuple[str, str]:
        """
        Determines if the last token of line1 and the first token of the line2 constitute a known word split;
        if yes, returns the tuple of split tokens (token1, token2),  which together constitutes a known split word,
        otherwise returns an empty tuple ().

        The hyphen, if present at the end of line1, is included/excluded in token1
        depending on whether the known word contains the hyphen or not.
        """

        # locate the last token in the line1 which consists of word chars, possibly ending with a hyphen
        token1end = len(line1)

        while token1end > 0 and line1[token1end-1] in self.white_space_chars:
            token1end -= 1

        token1start = token1end

        if token1start > 0 and line1[token1start-1] == self.hyphen:
           token1start -= 1

        while token1start > 0 and line1[token1start-1].lower() in self.language.lc_word_chars:
            token1start -= 1

        token1 = line1[token1start:token1end]

        token1nohyphen = (token1.rstrip(self.hyphen) if token1.endswith(self.hyphen) else token1)

        # locate the first token in the line2 which consists of word chars, possibly including a hyphen
        token2start = 0

        while token2start < len(line2) and line2[token2start] in self.white_space_chars:
            token2start += 1

        token2end = token2start
        while token2end < len(line2) and line2[token2end].lower() in self.language.lc_word_chars:
            token2end += 1

        token2 = line2[token2start:token2end]

        if len(token1) == 0 or len(token1nohyphen) == 0 or len(token2)==0:
            return ()

        #explicit hyphen present: if tokens combined without hyphen give a known world,
        #interpret this as intentional hyphenation split, even if both tokens are known words themselves
        # (is_split_word returns False if both tokens are known, so we perform this check before calling is_split_word)
        if token1nohyphen != token1 and (self.is_known_word(token1nohyphen + token2) or self.is_known_word(token1 + token2)):
            count_with_hyphen = self.get_word_count(token1 + token2)
            count_without_hyphen = self.get_word_count(token1nohyphen + token2)
            if count_without_hyphen > 0 and count_without_hyphen >= count_with_hyphen:
                return (token1nohyphen, token2)
            elif count_with_hyphen > 0 and count_with_hyphen > count_without_hyphen:
                return (token1, token2)
        elif self.is_split_word(token1, token2):
            return (token1, token2)
        else:
            return ()

    def dehyphenate_file(
        self,
        in_file_path:str,
        out_file_path:str,
        log_file_path:str) :
        """
        In the given text file joins the words, which were split by new line and return transformed text.
        Writes into the log the tuples of joined tokens and correspondent line numbers

        infilepath - input text file path
        outfilepath - output text file path
        logfilepath - log file path, lists line numbers where word splits started and pairs of tokens joined
        """

        lines = list()
        with open(in_file_path, "rt", -1, 'utf-8') as f:
            for line in f:
                lines.append(line.rstrip(self.new_line_sep))

        log = StringIO()

        line_count = len(lines)
        iLine1 = 0
        for iLine1 in range(line_count-1):
            iLine2 = iLine1 + 1

            line1 = lines[iLine1]

            #skip empty lines while searching for the line where ends the split, started at line1
            while iLine2 < line_count and lines[iLine2].strip(self.white_space_chars) == '':
                iLine2 += 1

            if iLine2 == line_count:
                break;

            line2 = lines[iLine2]

            split_tokens = self.get_split_tokens(line1, line2)
            if split_tokens != (): # a word split between the line1 and line2, process it
                (t1, t2) = split_tokens
                log.write(str(iLine1+1) + ',' + str(split_tokens) + '\n')

                t1start = line1.rfind(t1)
                t2start = line2.find(t2) 
                t2end = line2.find(t2) + len(t2)
                
                t2_rhs_punct_start = t2_rhs_punct_end = t2end
                while t2_rhs_punct_end < len(line2) and line2[t2_rhs_punct_end] in self.word_rhs_adj_punct:
                    t2_rhs_punct_end += 1
                t2_rhs_adj_punct = line2[t2_rhs_punct_start:t2_rhs_punct_end]

                # put the split word joined again into the end of the 1st line, 
                # followed by adjoining word punctuation, if any
                lines[iLine1] = (line1[:t1start]) + t1 + t2 + t2_rhs_adj_punct
                #remove the 2nd part of split word from the beginning of the 2nd line
                lines[iLine2] = line2[t2_rhs_punct_end:]

        #write modified text into the output file
        with open(out_file_path, 'wt', -1, 'utf-8') as f:
            for line in lines:
                f.write(line + '\n')

        with open(log_file_path, 'wt', -1, 'utf-8') as f:
            f.write(log.getvalue())

    @staticmethod
    def dehyphenate_corpus(
            corpus_word_counter:Counter[str],
            corpus_in_dir='corpus_raw',
            corpus_out_dir='corpus_dehyphen',
            dehyphen_log_dir='corpus_dehyphen_log',
            corpus_file_pattern='*.txt',
            file_suffixes_to_remove=['.pdf', '.djvu', '.epub', '.mobi']):

        if not os.path.exists(corpus_out_dir):
            os.mkdir(corpus_out_dir)
        if not os.path.exists(dehyphen_log_dir):
            os.mkdir(dehyphen_log_dir)

        # dehyphenate corpus text files
        dehyphenator: Dehyphenator = Dehyphenator(corpus_word_counter)
        corpus_file_path_pattern = os.path.join(corpus_in_dir, corpus_file_pattern)
        for in_file_path in tqdm(glob(corpus_file_path_pattern), desc='dehyphenating corpus'):
            in_file_name = os.path.split(in_file_path)[1]
            in_file_ext = os.path.splitext(in_file_name)[1]
            base_file_name = in_file_name.removesuffix(in_file_ext)
            base_file_name = Corpus.remove_file_suffixes(base_file_name, file_suffixes_to_remove)
            out_file_path = os.path.join(corpus_out_dir, base_file_name + in_file_ext)
            log_file_path = os.path.join(dehyphen_log_dir, base_file_name + '.log')
            dehyphenator.dehyphenate_file(in_file_path, out_file_path, log_file_path)


def test_token_pairs(dehyphenator:Dehyphenator):
    token_pairs = [
        ('ener','gy'), ('ener-','gy'), ('Ener','gy'), ('ENER','GY'), ('Ener-','gy'), \
        ('Un','derstand'), ('un','derstand'), ('under','stand'), ('UNDER','STAND'), ('UNDER-','STAND'),\
        ('ho','st'), ('rea','d'), ('half-','moon'), ('half','moon'),\
        ('pro-','ductions'),('pro','ductions'),('kill-','ing'),('kill','ing'),('roll-','ing'),('roll','ing'),\
        ('Al-','Akhbar'),('Al','Akhbar')]
    print("\r\nis_split_word:")
    for p in token_pairs:
        print(p, '->', dehyphenator.is_split_word(p[0], p[1]))

def test_line_pairs():
    known_words = ['energy','understand','productions','killing','rolling','host','read','half','moon','half-moon','Al-Akhbar']
    dehyphenator = Dehyphenator(Counter(known_words))
    line_pairs = [('ener', 'gy'), ('ener ', ' gy'), ('ener- ', '  gy'), ('Ener ', '  gy'), ('Ener- ', ' gy'), \
                  ('Un', 'derstand'), ('Un-', 'derstand'), ('un', 'derstand'), ('un- ', 'derstand'), ('under', 'stand'),
                  ('under- ', '  stand'), ('UNDER- ', ' STAND'), \
                  ('pro-', 'ductions'), ('pro', 'ductions'), ('kill-', 'ing'), ('kill', 'ing'), ('roll-', 'ing'),
                  ('roll', 'ing'), \
                  ('ho', 'st'), ('rea', 'd'), ('half-', 'moon'), ('half', 'moon'), ('Al-', 'Akhbar'),
                  ('Al- ', ' Akhbar')]

    print("\r\nget_split_tokens:")
    for p in line_pairs:
        tokens = dehyphenator.get_split_tokens(p[0], p[1])
        print(f"{p} -> {tokens}")

def test_dehyphenate_file(text_file_path:str):
    from vocabulary import Vocabulary
    word_counter = Vocabulary.file_word_counter(text_file_path=text_file_path, min_word_len=1)
    dehyphenator:Dehyphenator = Dehyphenator(word_counter)
    file_dir, file_name = os.path.split(text_file_path)
    out_file_path = os.path.join(file_dir, file_name + '.dehyphen')
    log_file_path = os.path.join(file_dir, file_name + '.dehyphen.log')
    dehyphenator.dehyphenate_file(text_file_path, out_file_path, log_file_path)

#test_line_pairs()
# test_dehyphenate_file('corpus_raw\\Good T. Above Top Secret, 1988.djvu.txt')
# Dehyphenator.dehyphenate_corpus()
