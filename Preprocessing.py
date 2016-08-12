import re
import nltk
from nltk.corpus import stopwords
from nltk.tag.perceptron import PerceptronTagger
from nltk.stem.porter import PorterStemmer
from collections import Counter
from tqdm import tqdm

negation_words = [ 'no', 'not', 'none', 'nobody', 'nothing', 'neither', 'nowhere', 'never']

# Default English stopwords list obtained from http://www.ranks.nl/stopwords
stopwords_list = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours   ourselves", "out", "over", "own", "same", "shall", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]

emo_repl = { '<3': 'good', ':d': 'good', ':dd': 'good', '8)': 'good', ':-)': 'good', ':)': 'good', ';)': 'good', '(-:': 'good', '(:': 'good', ':/': 'bad', ':>': 'sad', ":')": 'sad', ":-(": 'bad', ':(': 'bad', ':S': 'bad', ':-S': 'bad' }

contractions = { "ain't": "am not; are not; is not; has not; have not", "aren't": "are not; am not", "can't": "cannot", "can't've": "cannot have", "'cause": "because", "could've": "could have", "couldn't": "could not", "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", "he'd": "he had ; he would", "he'd've": "he would have", "he'll": "he shall ; he will", "he'll've": "he shall have ; he will have", "he's": "he has ; he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how has ; how is ; how does", "i'd": "i had ; i would", "i'd've": "i would have", "i'll": "i shall ; i will", "i'll've": "i shall have ; i will have", "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it had ; it would", "it'd've": "it would have", "it'll": "it shall ; it will", "it'll've": "it shall have ; it will have", "it's": "it has ; it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not", "mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she had ; she would", "she'd've": "she would have", "she'll": "she shall ; she will", "she'll've": "she shall have ; she will have", "she's": "she has ; she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have", "so's": "so as ; so is", "that'd": "that would ; that had", "that'd've": "that would have", "that's": "that has ; that is", "there'd": "there had ; there would", "there'd've": "there would have", "there's": "there has ; there is", "they'd": "they had ; they would", "they'd've": "they would have", "they'll": "they shall ; they will", "they'll've": "they shall have ; they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we had ; we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what shall ; what will", "what'll've": "what shall have ; what will have", "what're": "what are", "what's": "what has ; what is", "what've": "what have", "when's": "when has ; when is", "when've": "when have", "where'd": "where did", "where's": "where has ; where is", "where've": "where have", "who'll": "who shall ; who will", "who'll've": "who shall have ; who will have", "who's": "who has ; who is", "who've": "who have", "why's": "why has ; why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have", "you'd": "you had ; you would", "you'd've": "you would have", "you'll": "you shall ; you will", "you'll've": "you shall have ; you will have", "you're": "you are", "you've": "you have" }

def ReplaceEmoticons(sentence_vec):
    sentence = []

    for word in sentence_vec:
        if word in emo_repl.keys():
            sentence.append(emo_repl[word])
        else:
            sentence.append(word)

    return sentence

def ExpandContractions(sentence_vec):
    sentence = []

    for word in sentence_vec:
        if word in contractions.keys():
            sentence.extend(contractions[word].split(';')[0].strip().split())
        else:
            sentence.append(word)

    return sentence

def CleanSentence(string, PreprocessEmoticons, PreprocessContractions):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    # Remove non-ascii characters
    re.sub(r'[^\x00-\x7F]', '', string)
    string = re.sub(r"\.", " \. ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    string = string.split(' ')

    if PreprocessEmoticons:
        string = ReplaceEmoticons(string)

    if PreprocessContractions:
        string = ExpandContractions(string)

    string = " ".join(string)

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower().split(' ')

def StopwordRemoval(sentence_vec):
    """
    Function that filters a sentence of stopwords.

    input:
        sentence_vec A list of strings representing the sentence to be filtered
    output:
        Vector of words with the stopwords removed
    """
    ret = []
    for word in sentence_vec:
        data = word.split('_')
        if len(data) > 1:
            raw_word = data[1]
        else:
            raw_word = data[0]
        
        if raw_word not in stopwords.words('english'):
            ret.append(word)
    return ret

def StemSentence(sentence_vec):
    """

    """
    if not hasattr(StemSentence, "stemmer"):
        StemSentence.stemmer = PorterStemmer()

    return [StemSentence.stemmer.stem_word(word) for word in sentence_vec]

def POSTag(sentence_vec):
    """

    """
    if not hasattr(POSTag, "tagger"):
        # Speed up tagging
        POSTag.tagger = PerceptronTagger()

    return ['{0}_{1}'.format(pos[1], pos[0]) for pos in nltk.tag._pos_tag(sentence_vec, None, POSTag.tagger)]

def Preprocess(sentences, PreprocessPOS=False, PreprocessStem=False, PreprocessStopword=False, PreprocessContractions=False, PreprocessEmoticons=False, PreprocessNegation=False):
    if sentences == None:
        return []

    processed_sentences = []

    for sentence in tqdm(sentences):
        processed_sentence = sentence

        processed_sentence = CleanSentence(processed_sentence, PreprocessEmoticons, PreprocessContractions)

        if PreprocessStem:
            processed_sentence = StemSentence(processed_sentence)

        if PreprocessPOS:
            processed_sentence = POSTag(processed_sentence)

        if PreprocessStopword:
            processed_sentence = StopwordRemoval(processed_sentence)

        if PreprocessNegation:
            processed_sentence = NegationTagging(processed_sentence)

        processed_sentences.append(processed_sentence)

    return processed_sentences

def NegationTagging(sentence):
    ret = []
    negated = False

    for word in sentence:
        data = word.split('_')
        if len(data) > 1:
            raw_word = data[1]
            if raw_word in negation_words:
                negated = not negated
                ret.append(word)
            elif negated:
                ret.append("{}_{}_{}".format(data[0], 'NEG', data[1]))
            else:
                ret.append(word)
        else:
            raw_word = data[0]
            if raw_word in negation_words:
                negated = not negated
                ret.append(word)
            elif negated:
                ret.append("{}_{}".format('NEG', word))
            else:
                ret.append(word)


    return ret
