from speech_utils import *
import pandas as pd
N_gram=2
phones_DF=pd.read_csv('./Datasets/DM1008/sub-DM1008_ses-intraop_task-lombard_annot-produced-phonemes.tsv',sep='\t')
sentences_DF=pd.read_csv('./Datasets/DM1008/sub-DM1008_ses-intraop_task-lombard_annot-produced-sentences.tsv',sep='\t')
words_DF=pd.read_csv('./Datasets/DM1008/sub-DM1008_ses-intraop_task-lombard_annot-produced-words.tsv',sep='\t')

# bigram_words=get_ngrams(N_gram,words_DF['word'].to_list() )
# sentences N-gram model
sentences_NgramModel=NgramModel(N_gram)
for i in range(sentences_DF.shape[0]):
    sentences_NgramModel.update(sentence=sentences_DF['sentence'][i], need_tokenize=True)

print(sentences_NgramModel.prob(('THE',),'SAME'))
print(sentences_NgramModel.map_to_probs(('THE',)))

# words N-gram model
words_NgramModel=NgramModel(N_gram)
words_NgramModel.update(sentence=listToString(words_DF['word'].to_list()), need_tokenize=True)

print(words_NgramModel.prob(('THE',),'SAME'))
print(words_NgramModel.map_to_probs(('THE',)))

# phones N-gram model
phones_NgramModel=NgramModel(N_gram)
phones_NgramModel.update(sentence=listToString(phones_DF['phoneme'].to_list()), need_tokenize=True)

print(phones_NgramModel.prob(('HH',),'IY1'))
print(phones_NgramModel.map_to_probs(('HH',)))


