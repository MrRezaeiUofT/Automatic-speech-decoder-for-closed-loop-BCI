'''train d4'''

############################################# D4 Model ######################################################


# from torch import nn
# ''' build state model'''
# class State_model(nn.Module):
#
#     def __init__(self, max_sent_length,steps, embedding_dim, vocab_size):
#         super(State_model, self).__init__()
#         self.steps=steps
#         self.embedding_dim=embedding_dim
#         self.vocab_size=vocab_size
#         self.embeddings = nn.Embedding(max_sent_length, embedding_dim)
#         self.linear1 = nn.Linear(vocab_size * steps, embedding_dim)
#
#         self.linear2 = nn.Linear(2*embedding_dim+vocab_size, embedding_dim)
#         self.linear3 = nn.Linear(embedding_dim, vocab_size)
#
#     def forward(self, inputs):
#         p, k, pp = inputs
#         k_embeds = self.embeddings(k)
#
#         x=torch.flatten(p, start_dim=1)
#
#         x = torch.nn.functional.relu(self.linear1(x))
#         x = torch.cat((x, k_embeds,pp), axis=-1)
#         out = torch.nn.functional.relu(self.linear2(x))
#         out=torch.nn.functional.relu(self.linear3(out))
#         log_probs = torch.nn.functional.log_softmax(out, dim=1)
#         return torch.exp(log_probs)
# state_model= State_model(max_sent_length=data_in.shape[1],
#                          steps=2,
#                          embedding_dim=5,
#                          vocab_size=vocab_size)
#
#
# ''' train d4 model'''
# def D4_model(state_model, st,y_true,y_predic_probs, PL0, epochs,training ):
#
#     optim = torch.optim.Adam(state_model.parameters(),
#                              lr=1e-3, weight_decay=1e-6)
#     PL0=torch.Tensor(PL0)
#     # PL0=torch.ones((vocab_size,))/vocab_size
#     uniques, counts = np.unique(y_true, return_counts=True)
#     counts=counts/counts.sum()
#     weights=np.zeros((vocab_size,))
#     weights[uniques]=1/counts
#     weights=weights/weights.sum()
#     loss = nn.CrossEntropyLoss(torch.tensor(weights))
#     total_loss = []
#     if training:
#         for ii_ep in range(epochs):
#             for cc, ii_snt in enumerate(np.unique(st).astype('int')):
#                 sent_indexes = np.where(st == ii_snt)[0]
#                 y_sent = y_true[sent_indexes]
#                 y_probs_sent = y_predic_probs[sent_indexes]
#
#                 P_pp = y_probs_sent
#                 P_lang = np.zeros_like(P_pp)
#                 P_pp = to_t(P_pp)
#                 P_lang = to_t(P_lang)
#                 y_sent = to_t(y_sent)
#                 y_sent_one_hot=torch.nn.functional.one_hot(y_sent.squeeze().to(torch.int64), state_model.vocab_size).to(torch.float32)
#
#                 temp=torch.zeros((len(y_sent),state_model.steps,P_pp.shape[1]))
#                 P_pp_pad=torch.cat([PL0.repeat(state_model.steps,1),P_pp],axis=0)
#
#                 for ii_wrd in range(len(y_sent)):
#                         temp[ii_wrd,:,:]=P_pp_pad[ii_wrd :ii_wrd+state_model.steps, :]
#
#                 optim.zero_grad()
#                 P_total=state_model([temp,torch.tensor(np.arange(len(y_sent)), dtype=torch.int),P_pp])
#
#                 loss_state= loss(P_total,y_sent_one_hot)
#
#                 loss_state.backward()
#                 optim.step()
#
#                 total_loss.append(loss_state.detach().numpy())
#                 if cc == 0:
#                     P_all = P_total.detach().numpy()
#                     y_all = y_sent_one_hot.detach().numpy()
#                     P_p_all = P_pp.detach().numpy()
#                     P_lang_all = P_lang.detach().numpy()
#                 else:
#
#                     P_all = np.concatenate([P_all, P_total.detach().numpy()], axis=0)
#                     y_all = np.concatenate([y_all, y_sent_one_hot.detach().numpy()], axis=0)
#                     P_p_all = np.concatenate([P_p_all, P_pp.detach().numpy()], axis=0)
#                     P_lang_all = np.concatenate([P_lang_all, P_lang.detach().numpy()], axis=0)
#     else:
#
#         for cc, ii_snt in enumerate(np.unique(st).astype('int')):
#                 sent_indexes = np.where(st == ii_snt)[0]
#                 y_sent = y_true[sent_indexes]
#                 y_probs_sent = y_predic_probs[sent_indexes]
#
#                 P_pp = y_probs_sent
#                 P_lang = np.zeros_like(P_pp)
#                 P_pp = to_t(P_pp)
#                 P_lang = to_t(P_lang)
#                 y_sent = to_t(y_sent)
#                 temp=torch.zeros((len(y_sent),state_model.steps,P_pp.shape[1]))
#                 P_pp_pad=torch.cat([PL0.repeat(state_model.steps,1),P_pp],axis=0)
#
#                 for ii_wrd in range(len(y_sent)):
#                         temp[ii_wrd,:,:]=P_pp_pad[ii_wrd :ii_wrd+state_model.steps, :]
#
#                 with torch.no_grad():
#                     P_total=state_model([temp,torch.tensor(np.arange(len(y_sent)), dtype=torch.int),P_pp])
#
#
#                 if cc == 0:
#                     P_all = P_total.detach().numpy()
#                     y_all = y_sent.detach().numpy()
#                     P_p_all = P_pp.detach().numpy()
#                     P_lang_all = P_lang.detach().numpy()
#                 else:
#
#                     P_all = np.concatenate([P_all, P_total.detach().numpy()], axis=0)
#                     y_all = np.concatenate([y_all, y_sent.detach().numpy()], axis=0)
#                     P_p_all = np.concatenate([P_p_all, P_pp.detach().numpy()], axis=0)
#                     P_lang_all = np.concatenate([P_lang_all, P_lang.detach().numpy()], axis=0)
#
#     return P_all, y_all, P_p_all, P_lang_all, state_model, total_loss
# P_all_D4_tr,y_all_D4_tr,P_p_all_D4_tr,P_lang_all_D4_tr, state_model, total_loss = D4_model(state_model,
#                                                                                            st_tr,
#                                                                                            y_org_tr,
#                                                                                            predic_probs_xgb_convert_back_tr,
#                                                                                            PL0 ,
#                                                                                            epochs=100,
#                                                                                            training=True)
# plt.figure()
# plt.plot(total_loss)
# ''' test d4 model'''
# P_all_D4_te,y_all_D4_te,P_p_all_D4_te,P_lang_all_D4_te, state_model, total_loss = D4_model(state_model,
#                                                                                            st_te,
#                                                                                            y_org_te,
#                                                                                            predic_probs_xgb_convert_back_te,
#                                                                                            PL0 ,
#                                                                                            epochs=0,
#                                                                                            training=False)
''' test gpt performance'''
# prediction_result = np.zeros((500,2))
# p=0
# for ii in np.random.randint(0,data_in.shape[0],prediction_result.shape[0]):
#     input_length =  np.random.randint(1,15)
#     x = torch.tensor(data_in[ii][:input_length], dtype=torch.long).to(trainer.device)
#     x = x.expand(10, -1)
#
#     y, probs = gpt_model.generate(x, max_new_tokens=1, do_sample=True, top_k=40)
#     prediction_result[p, 0] = data_in[ii][input_length ]
#     prediction_result[p, 1] = (y.detach().cpu()[0, -1])
#     # print('-' * 80)
#     p =p+1
# print('accuracy= %f', accuracy_score(prediction_result[:,0],prediction_result[:,1])*100)

''''''
# def correct_D4_result_gpt_simple(model, st,y_true,y_predic_probs, PL0 ):
#     steps_L = 2
#     do_sample_L = True
#     for cc, ii_snt in enumerate(np.unique(st).astype('int')):
#         sent_indexes = np.where(st == ii_snt)[0]
#
#         y_sent= y_true[sent_indexes]
#         y_probs_sent = y_predic_probs[sent_indexes]
#
#         P_pp = y_probs_sent
#         P_lang = np.zeros_like(P_pp)
#         P_total = np.zeros_like(P_pp)
#
#
#
#         for ii_wrd in range(len(y_sent)):
#             input_length_L = ii_wrd
#             if ii_wrd == 0:
#                 P_total[ii_wrd,:] = np.multiply( PL0, P_pp[0,:])
#                 P_lang[ii_wrd,:] =  np.multiply( PL0, P_pp[0,:])
#
#             elif ii_wrd ==1:
#                 x = torch.tensor(np.argmax(P_pp[ii_wrd-1, :].reshape([1,-1]), axis=1), dtype=torch.long).to(trainer.device)
#                 x = x.expand(num_samples_langmodel, -1)
#                 y, probs = model.generate(x, max_new_tokens=steps_L, do_sample=do_sample_L, top_k=40)
#                 P_lang[ii_wrd, :] = probs.mean(axis=0)[0, :]
#                 P_total[ii_wrd,:] = np.multiply( P_lang[ii_wrd, :], P_pp[ii_wrd,:])
#             else:
#
#                 x = torch.tensor(np.argmax(P_pp[:ii_wrd, :], axis=1), dtype=torch.long).to(trainer.device)
#                 x = x.expand(num_samples_langmodel, -1)
#                 y, probs = model.generate(x, max_new_tokens=steps_L, do_sample=do_sample_L, top_k=40)
#                 P_lang[ii_wrd, :] = probs.mean(axis=0)[0, :]
#                 P_total[ii_wrd, :] = np.multiply( P_lang[ii_wrd, :], P_pp[ii_wrd, :])
#         if cc == 0:
#             P_all = P_total
#             y_all = y_sent
#             P_p_all = P_pp
#             P_lang_all = P_lang
#         else:
#
#             P_all = np.concatenate([P_all,P_total],axis=0)
#             y_all = np.concatenate([y_all,y_sent],axis=0)
#             P_p_all = np.concatenate([P_p_all,P_pp],axis=0)
#             P_lang_all = np.concatenate([P_lang_all, P_lang], axis=0)
#
#
#     return P_all, y_all, P_p_all, P_lang_all

''' tune data language model'''
# temp_data_in_tune=np.concatenate([np.argmax(PL0)*np.ones((vocab_size,)), np.argmax(predic_probs_xgb_convert_back_tr, axis=1)], axis=0)
# temp_data_out_tune=np.concatenate([np.argmax(PL0)*np.ones((vocab_size,)), y_org_tr.squeeze()], axis=0)
# data_in_tune=np.zeros((y_org_tr.shape[0],data_in.shape[1]))
# data_out_tune=np.zeros_like(data_in_tune)
# for ii_indx in range(data_in_tune.shape[0]):
#     data_in_tune[ii_indx]=temp_data_in_tune[ii_indx:ii_indx+data_in.shape[1]]
#     data_out_tune[ii_indx] = temp_data_out_tune[ii_indx:ii_indx + data_in.shape[1]]
