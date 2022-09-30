import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import string
import random
import time
import seaborn as sn
from typing import List


def mfcc_feature_extraction(audio_file, n_mfcc):
    """
    extract mfcc features from an audio file -> (n_mfcc)*3 features
    :param audio_file: audio file address
    :param n_mfcc: number of MFCC features
    :return:
    """
    signal, sr = librosa.load(audio_file)
    mfccs = librosa.feature.mfcc(y=signal, n_mfcc=n_mfcc, sr=sr)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    mfccs_features = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))
    return mfccs_features, sr, signal

def get_ngrams(n: int, tokens: list) -> list:
    """
    :param n: n-gram size
    :param tokens: tokenized sentence
    :return: list of ngrams
    ngrams of tuple form: ((previous wordS!), target word)
    """
    tokens = (n-1)*['<START>']+tokens
    l = [(tuple([tokens[i-p-1] for p in reversed(range(n-1))]), tokens[i]) for i in range(n-1, len(tokens))]
    return l
# ideally we would use some smart text tokenizer, but for simplicity use this one
def tokenize(text: str) -> List[str]:
    """
    :param text: Takes input sentence
    :return: tokenized sentence
    """
    for punct in string.punctuation:
        text = text.replace(punct, ' '+punct+' ')
    t = text.split()
    return t

class NgramModel(object):

    def __init__(self, n):
        self.n = n

        # dictionary that keeps list of candidate words given context
        self.context = {}

        # keeps track of how many times ngram has appeared in the text before Ngram_counter dictionary just counts
        # how many times we have seen a particular N-gram in our training set before.

        self.ngram_counter = {}

    def update(self, sentence: str, need_tokenize=False) -> None:
        """
        Updates Language Model
        :param sentence: input text
        """
        n = self.n
        if need_tokenize:
            ngrams = get_ngrams(n, tokenize(sentence))
        else:
            ngrams = get_ngrams(n, sentence)
        for ngram in ngrams:
            if ngram in self.ngram_counter:
                self.ngram_counter[ngram] += 1.0
            else:
                self.ngram_counter[ngram] = 1.0

            prev_words, target_word = ngram
            if prev_words in self.context:
                self.context[prev_words].append(target_word)
            else:
                self.context[prev_words] = [target_word]

    def prob(self, context, token):
        """
        Calculates probability of a candidate token to be generated given a context
        :return: conditional probability
        """
        try:
            count_of_token = self.ngram_counter[(context, token)]
            count_of_context = float(len(self.context[context]))
            result = count_of_token / count_of_context

        except KeyError:
            result = 0.0
        return result

    def map_to_probs(self, context):
        """
        Given a context what is the probability of words
        :param context:
        :return:
        """

        map_to_probs = {}
        token_of_interest = self.context[context]
        for token in token_of_interest:
            map_to_probs[token] = self.prob(context, token)
        return map_to_probs

    def random_token(self, context):
        """
        Given a context we "semi-randomly" select the next word to append in a sequence
        :param context:
        :return:
        """
        r = random.random()
        map_to_probs = {}
        token_of_interest = self.context[context]
        for token in token_of_interest:
            map_to_probs[token] = self.prob(context, token)

        summ = 0
        for token in sorted(map_to_probs):
            summ += map_to_probs[token]
            if summ > r:
                return token

    def generate_text(self, token_count: int):
        """
        :param token_count: number of words to be produced
        :return: generated text
        """
        n = self.n
        context_queue = (n - 1) * ['<START>']
        result = []
        for _ in range(token_count):
            obj = self.random_token(tuple(context_queue))
            result.append(obj)
            if n > 1:
                context_queue.pop(0)
                if obj == '.':
                    context_queue = (n - 1) * ['<START>']
                else:
                    context_queue.append(obj)
        return ' '.join(result)


def listToString(s):
    # initialize an empty string
    str1 = " "

    # return string
    return (str1.join(s))

def convert_phone_phnoe_id(input,key_covert):
    output={}
    for key,value in input.items():
        output[(key_covert[key])]=value
    return output


def matrix_show(matr, labels,title):
    fig, ax = plt.subplots()
    im = ax.imshow(matr)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(matr.shape[0]), labels=labels)
    ax.set_yticks(np.arange(matr.shape[1]), labels=labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(matr.shape[1]):
        for j in range(matr.shape[0]):
            text = ax.text(j, i, matr[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("Diffusion matrix pred "+ title)
    fig.tight_layout()
    plt.show()

def get_delay_btwn_phoneme(acustic_feature_DF):
    delay_model = np.zeros((len(acustic_feature_DF.phone_id.unique()), len(acustic_feature_DF.phone_id.unique()), 2))
    df_delay_model = acustic_feature_DF[acustic_feature_DF.phone_id >= 2].reset_index()
    # diagonal elememnts
    for phoneme_id in acustic_feature_DF.phone_id.unique():
        delay_model[phoneme_id, phoneme_id, 0] = acustic_feature_DF.phone_duration[
            acustic_feature_DF.phone_id == phoneme_id].mean()
        delay_model[phoneme_id, phoneme_id, 1] = acustic_feature_DF.phone_duration[
            acustic_feature_DF.phone_id == phoneme_id].std()

    for phoneme_id_t in df_delay_model.phone_id.unique():
        phn_t_index = df_delay_model.index[df_delay_model.phone_id == phoneme_id_t].to_numpy()
        phn_t1_index = phn_t_index - 1
        phn_t1_index[phn_t1_index < 0] = 0
        df_t1 = df_delay_model.iloc[phn_t1_index]
        for ii in df_t1.phone_id.unique():
            delay_model[ phoneme_id_t,ii, 0] = df_t1.diff_from_last_phone[
                df_t1.phone_id == ii].mean()
            delay_model[ phoneme_id_t,ii, 1] = df_t1.diff_from_last_phone[
                df_t1.phone_id == ii].std()

    # f, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True, sharex=True)
    # sn.heatmap(np.round(delay_model[:, :, 0]), ax=axes[0], annot=True)
    # sn.heatmap(np.round(delay_model[:, :, 1]), ax=axes[1], annot=True)
    return delay_model

