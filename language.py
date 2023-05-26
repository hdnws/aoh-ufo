from typing import List
from enum import Enum
from abc import ABC, abstractmethod
import re

class Language(ABC):
    """Abstract base class to describes language related information:
       lowercased alphabet and vowels; used for building vocabulary and dehyphenation."""
    def __init__(self, name:str, lc_word_chars:str, lc_vowels:str):
        self.name = name
        self.lc_word_chars = lc_word_chars
        self.lc_vowels = lc_vowels

    @abstractmethod
    def can_be_word(self, token:str)->bool:
        """Determines, if the given token can in principle be accepted as a word of this language."""
        return true

class EnglishLanguage(Language):
    def __init__(self):
        super().__init__(name='english', # lowercase name 'english' as reconginized by NLTK tokenizers
                         lc_word_chars="abcdefghijklmnopqrstuvwxyz",
                         lc_vowels="aeiouy")
        self.term_pattern = "([a-zA-Z]+(-?[a-zA-Z\d]+)+)|([a-zA-Z]+'?[a-zA-Z]*)|(\d+)"

    def can_be_word(self, token:str):
        return re.fullmatch(self.term_pattern, token) is not None


English = EnglishLanguage()


class CaseConversionMode(Enum):
    """Defines text case conversion mode."""
    lower=0      # convert to lowercase if original token is not found in vocabulary but it's lowercase form is found
    any=1        # use the case form which is present in vocabulary in the order: original token, lowercase, title case, upper case

def case_forms(token:str, case_mode:CaseConversionMode)->List[str]:
    if case_mode == CaseConversionMode.any:
        return [token, token.lower(), token.title(), token.upper()]
    else:
        return [token, token.lower()]




def test_can_be_word():
    text = "I'm don't Lord Hill-Norton face-to-face F-16 B-21s 123 1-2"
    for token in text.split(' '):
        print(token, English.can_be_word(token))

# test_can_be_word()