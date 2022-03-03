
from sklearn import preprocessing

from GMM_utils import *
from speech_utils import *

###



audio_file = "../Datasets/DM1008/sub-DM1008_ses-intraop_task-lombard_run-03_recording-directionalmicaec_physio.wav"
# Extracting MFCCs

mfccs_features, sr, signal = mfcc_feature_extraction(audio_file, 13)
mfccs_features = preprocessing.scale(mfccs_features, axis=1)
# plt.figure(figsize=(25, 10))
# out=librosa.display.specshow(mfccs_features,
#                          x_axis="time",
#                          sr=sr)
# plt.colorbar(format="%+2.f")
# plt.title("delta2_mfccs")
# plt.show()
X= mfccs_features.transpose()
gmm_accustic = GaussianMixture(n_components=16).fit(X)