from data_utilities import *
import matplotlib.pyplot as plt
from model_utils import NgramModel, get_state_transition_p_bigram
patient_id = 'DM1008'
datasets_add = './Datasets/'
feature_id = [0, 128]
dt = 10
N_gram = 2
total_data = pd.read_csv(datasets_add + patient_id + '/' + 'Preprocessed_data/' + 'prepro_phoneme_neural_total_v1.csv')
''' visualization of the phoneme histograms'''
total_data.phoneme.hist(log=True)

''' re-assign the phoneme ids'''
phones_code_dic = dict(zip(total_data.phoneme.unique(), np.arange(total_data.phoneme.nunique())))

''' phonemes N-gram model'''
phones_NgramModel=NgramModel(N_gram)
phones_NgramModel.update(sentence=(total_data['phoneme'].to_list()), need_tokenize=False)
# print(phones_NgramModel.prob(('HH',),'IY1'))
# print(phones_NgramModel.map_to_probs(('HH',)))
num_state = total_data['phoneme'].nunique()
pwtwt1 = get_state_transition_p_bigram(phones_code_dic,phones_NgramModel)
plt.figure()
plt.imshow(np.log(pwtwt1))

''' Phoneme duration model'''
phoneme_duration_df = pd.DataFrame()
phoneme_duration_df['mean'] = total_data.groupby(by=["phoneme_id"], dropna=False).mean().phoneme_duration
phoneme_duration_df['std'] = total_data.groupby(by=["phoneme_id"], dropna=False).std().phoneme_duration


