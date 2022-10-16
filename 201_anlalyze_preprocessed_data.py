from data_utilities import *
import seaborn as sns
import matplotlib.pyplot as plt
import json
patient_id = 'DM1008'
datasets_add = './Datasets/'
data_add = datasets_add + patient_id + '/' + 'Preprocessed_data/'
# Opening JSON file
with open(data_add + "dataset_info.json", 'r') as openfile:
    # Reading from json file
    dataset_info = json.load(openfile)

total_data = pd.read_csv(data_add + 'prepro_phoneme_neural_total_v1.csv')

''' corr. phonemes and neural features '''
corr = total_data.corr(method='pearson')
phonemes_corr = corr.phoneme_id[dataset_info['feature_id'][0]+1:dataset_info['feature_id'][1]].abs()
phonemes_corr = phonemes_corr.sort_values(ascending=False)
g = sns.barplot(x=total_data.columns[total_data.columns.str.contains("feature")], y=(phonemes_corr.to_numpy()))
g.set_xticklabels(g.get_xticklabels(), rotation=90)
g.set_title('neural signals corr. with phoneme ids')
''' phonemes '''
plt.figure()
g = total_data.phoneme_duration.hist( log=True)


