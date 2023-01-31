from data_utilities import *
import json
import eng_to_ipa as ipa
import cmudict
sentences_path ='LM/20200708-Harvard-sentences/20200708-Harvard-sentences/Harvard-setences-to-phonemes/Harvard-Senteces-flaged-ipa.csv'
datasets_add = './Datasets/'



cmudict_dic = cmudict.dict()
dataset_df = pd.read_csv(datasets_add+sentences_path)
for sent_id in range(dataset_df.shape[0]):
    sentence_arpabet_list = []
    sentence_words_list = dataset_df.Sentence[sent_id].split()
    for word in sentence_words_list:
        sentence_arpabet_list.append(apply_stress_remove(cmudict_dic[word]))
    flat_sentence_arpabet_list = [item for sublist in sentence_arpabet_list for item in sublist]
    flat_sentence_arpabet_list = [item for sublist in flat_sentence_arpabet_list for item in sublist]
    if sent_id == 0:
        phonemes_df = pd.DataFrame(flat_sentence_arpabet_list, columns=['phoneme'])
        phonemes_df['trial_id'] = sent_id+1
    else:
        phonemes_df_temp = pd.DataFrame(flat_sentence_arpabet_list, columns=['phoneme'])
        phonemes_df_temp['trial_id'] = sent_id+1
        phonemes_df = pd.concat([phonemes_df,phonemes_df_temp])

phones_code_dic = dict(zip(phonemes_df.phoneme.unique(), np.arange(phonemes_df.phoneme.nunique())))
########## add nan to the phonemes  sequences
if 'NAN' in phones_code_dic:
    pass
else:
    phones_code_dic.update({'NAN': len(phonemes_df.phoneme.unique())})


##### add SP as space indicator in the phonemes dataset
if 'SP' in phones_code_dic:
    pass
else:
    phones_code_dic.update({'SP': len(phonemes_df.phoneme.unique())+1})

phonemes_df['phoneme_id'] = 0
phonemes_df['phoneme_id'] = phonemes_df['phoneme'].apply(lambda x: phones_code_dic[x])
# phonemes_df['ph_temp'] = 1

phonemes_df.to_csv(datasets_add+'LM/phonemes_df_harvard_dataset.csv')
phones_code_dic_df = pd.DataFrame(phones_code_dic.keys(), columns=['phonemes'])
phones_code_dic_df['ids'] = phones_code_dic.values()
phones_code_dic_df.to_csv(datasets_add+'LM/phonemes_df_harvard_dataset_phonemes_dic.csv')