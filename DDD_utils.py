from scipy.linalg import hankel
import numpy as np
import tensorflow as tf

from keras import layers
from keras import Model
# import tensorflow_probability as tfp
# tfd=tfp.distributions


def calDesignMatrix(X,h):
    PadX = np.zeros([h , X.shape[1]])
    PadX =np.concatenateexit([PadX,X],axis=0)
    XDsgn=np.zeros([X.shape[0], X.shape[1]*h])
    for i in range(X.shape[1]):
         XDsgn[:, i* h : (i+1) * h]= hankel(PadX[:  - h , i],PadX[ - h :, i])
    return XDsgn

def calDesignMatrix_V2(X,h):
    '''

    :param X: [samples*Feature]
    :param h: hist
    :return: [samples*hist*Feature]

    '''
    PadX = np.zeros([h , X.shape[1]])
    PadX =np.concatenate([PadX,X],axis=0)
    XDsgn=np.zeros([X.shape[0], h, X.shape[1]])
    # print(PadX.shapepe)
    for i in range(0,XDsgn.shape[0]):
         #print(i)
         XDsgn[i, : , :]= (PadX[i:h+i,:])
    return XDsgn

def get_state_transition_p_bigram(phones_code_dic, Biogram):
    pwtwt1=np.zeros((len(phones_code_dic),len(phones_code_dic)))

    for key,value in phones_code_dic.items():
        # print(key)
        fu_dic=Biogram.map_to_probs((key,))
        for key_temp, value_temp in fu_dic.items():
            pwtwt1[value,phones_code_dic[key_temp]]=value_temp
    return pwtwt1

# def get_state_transition_p_Nigram(phones_code_dic, Ngram,N):
#     pwtwt1=np.zeros((len(phones_code_dic),len(phones_code_dic)))
#
#     for key,value in phones_code_dic.items():
#         # print(key)
#         fu_dic=Biogram.map_to_probs((key,))
#         for key_temp, value_temp in fu_dic.items():
#             pwtwt1[value,phones_code_dic[key_temp]]=value_temp
#     return pwtwt1

def get_model(In,out):
    spec_start = layers.Input((In.shape[1],In.shape[2]))


    # spec_x = layers.GRU(10, activation='tanh', dropout=.2, recurrent_dropout=.2, return_sequences=True)(spec_start)
    spec_x = layers.GRU(In.shape[1], activation='tanh', dropout=.2, recurrent_dropout=.2, return_sequences=False)(spec_start)
    spec_x=layers.Dense(out.shape[1])(spec_x)
    out = layers.Activation('softmax')(spec_x)
    _model = Model(inputs=spec_start, outputs=out)
    _model.compile(optimizer='Adam', loss='binary_crossentropy',metrics = ['accuracy'])
    # _model.summary()
    return _model