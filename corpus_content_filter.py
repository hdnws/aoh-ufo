from enum import Enum
from typing import Callable
from corpus import Corpus, CorpusLevel, CorpusItem

class ContentType(Enum):
    """Defines two kinds of content of text in the corpus, which we compare against each other."""
    source = 0
    target = 1

def get_content_type(book_title:str)->ContentType:
    return (ContentType.target if book_title.lower().startswith('summers') else ContentType.source)

def content_type_filter(content_type:ContentType) -> Callable[[CorpusItem], bool]:
    """Returns the filter function, for passing into the Corpus iterator methods, like books_iterator(...)."""
    return (lambda corpus_item:
                (content_type is None or
                 get_content_type(corpus_item.title_path[0]) == content_type))

def test_content_type_filter(content_type:ContentType, level:CorpusLevel, corpus_dir:str='corpus_raw'):
    corpus_item_filter = content_type_filter(content_type)
    corpus_items = list(Corpus(corpus_dir, '*.txt').corpus_item_iterator(
        level = level, corpus_item_filter=corpus_item_filter))
    for corpus_item in corpus_items:
        print(corpus_item.title_path)
    print(f"{level.name} count: {len(corpus_items)}")

def test_count_tokens(content_type:ContentType, corpus_dir:str='corpus_line_sentence'):
    corpus = Corpus(corpus_dir, '*.txt')
    sentence_count = 0
    token_count = 0
    for sentence_tokens in corpus.sentences_of_tokens_iterator(None, content_type_filter(content_type)):
        sentence_count += 1
        token_count += len(sentence_tokens)
    print(f"ContentType: sentence_count={content_type}, token_count={token_count}")

# test_content_type_filter(ContentType.source, CorpusLevel.section, 'corpus_dehyphen')
# test_count_tokens(ContentType.source)