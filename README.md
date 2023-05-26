# aoh-ufo
Comparing Allies-of-Humanity message vs UFO research books.

Data preprocessing is performed via running preprocess.py.
This will preprocess the texts form the corpus_rawfolder and create a number of folders named "corpus_..." with preprocessed dataset, the last to perform analysis on being "corpus_line_sentence".
After that, the comparison of the sub-corpus of texts of Allies-of-Humanity message vs the sub-corpus of UFO research books can be launched via executing analysis.py.

The analysis involves the following similarity models for comparing texts:
- cosine similarity between documents represented as bag-of-words- or tfidf- vectors; 2 models.
- soft-cosine similarity between documents represented in the same 2 forms, with terms similarity supplied form the Word2Vec or FastText model, trained on the entire dataset corpus; 4 models.
The anaysis script performs the following steps:
- train Word2Vec and FastText models on the preprocessed texts;
- perform matching of the texts between the two sub-corpora on book and section level according to 6 different similarity models: cosine similarity and soft-cosine similirty;
- compare rankings, produced by comparing texts on book/section level by 6 models, using averaged Kendall-tau correlation coefficient;
- merge together similar adjoining paragraphs, according to 4 different soft-cosine similarity models;
- match the merged paragraphs between AoH and UFO subcorpora the same 4 soft-cosine similarity models.
The results of analysis are logged into the 'analysis' subfolder as .json files.
See paper/aoh-ufo.pdf for the final report, containing important background information and more technical details.
