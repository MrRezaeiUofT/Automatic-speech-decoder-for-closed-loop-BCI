import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import string
import random
import pandas as pd
import time
import seaborn as sn
from typing import List

def convert_phone_phnoe_id(input, key_covert):
    output={}
    for key,value in input.items():
        output[(key_covert[key])] = value
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


def get_state_transition_p_bigram(phones_code_dic, Biogram):
    pwtwt1=np.zeros((len(phones_code_dic),len(phones_code_dic)))

    for key,value in phones_code_dic.items():
        # print(key)
        fu_dic=Biogram.map_to_probs((key,))
        for key_temp, value_temp in fu_dic.items():
            pwtwt1[value,phones_code_dic[key_temp]]=value_temp
    return pwtwt1

def get_language_components(total_data, N_gram):


    ''' Phoneme duration model'''
    phoneme_duration_df = pd.DataFrame()
    phoneme_duration_df['mean'] = total_data.groupby(by=["phoneme_id"], dropna=False).mean().phoneme_duration
    phoneme_duration_df['std'] = total_data.groupby(by=["phoneme_id"], dropna=False).std().phoneme_duration

    ''' re-assign the phoneme ids'''
    phones_code_dic = dict(zip(total_data.phoneme.unique(), np.arange(total_data.phoneme.nunique())))

    ''' phonemes N-gram model'''
    non_phoneme_onset = total_data[total_data.phoneme_onset == 0].index.to_numpy()
    total_data = total_data.drop(non_phoneme_onset, axis=0)
    phones_NgramModel = NgramModel(N_gram)
    phones_NgramModel.update(sentence=(total_data['phoneme'].to_list()), need_tokenize=False)
    # print(phones_NgramModel.prob(('HH',),'IY1'))
    # print(phones_NgramModel.map_to_probs(('HH',)))
    num_state = total_data['phoneme'].nunique()
    pwtwt1 = get_state_transition_p_bigram(phones_code_dic, phones_NgramModel)
    plt.figure()
    plt.imshow(np.log(pwtwt1))
    return pwtwt1, phoneme_duration_df, phones_NgramModel, phones_code_dic