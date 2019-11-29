from deepsegment import DeepSegment
segmenter = DeepSegment('en', tf_serving=True)
from deepcorrect import DeepCorrect
corrector = DeepCorrect('/Users/rajesh/Documents/speech-analytics/deepconnect_models/deeppunct_params_en',
                        '/Users/rajesh/Documents/speech-analytics/deepconnect_models/deeppunct_checkpoint_wikipedia')


def boundary_definer(corpus):
    corpus = segmenter.segment(corpus)
    return corpus


def words_corrector(text):
    text = corrector.correct(text)
    return text[0]['sequence']


def processor(corpus):
    corpus = boundary_definer(corpus)
    ana_corpus = []
    for sentence in corpus:
        text = sentence
        ana_corpus.append(words_corrector(text=text))
    return ana_corpus

# print(boundary_definer("FG ibkbbk mnk we are the one Who can live"))
# print(words_corrector("We live once"))