close all
clear all
clc


%% patient DM1008 read daata
hist_len = 10;
%thr_spike=.02;
Cross_val_K= 5;
audio_strt_time = 63623.7147033506;
[data_w, header_w, raw_w] = tsvread( "../Datasets/DM1008/sub-DM1008_ses-intraop_task-lombard_annot-produced-words.tsv" );
[data_s, header_s, raw_s] = tsvread( "../Datasets/DM1008/sub-DM1008_ses-intraop_task-lombard_annot-produced-sentences.tsv" );
[audioIn,fs] = audioread("../Datasets/DM1008/sub-DM1008_ses-intraop_task-lombard_run-03_recording-directionalmicaec_physio.wav");
[coeffs,delta,deltaDelta,loc] = mfcc(audioIn,fs);
dt_features = (length(audioIn)/fs)/length(coeffs);
%% synchronize audio and word-sent. daata
data_w(:,1)= data_w(:,1)- audio_strt_time;
data_s(:,1)= data_s(:,1)- audio_strt_time;
sentence_IDs = find(strcmp(raw_s(:,8), 'HE OFFERED PROOF IN THE FORM OF A LARGE CHART'))-1;
tune_cif=true;
tune_cif_w=ones(10,1);
tune_cif_w(6)=1/10;
%% get the featurs
[XD_snt_tr,events_times_snt_tr,synth_CIF_snt_tr,wrd_event_snt_tr] =get_data(data_w,raw_w,data_s,coeffs, sentence_IDs(2:end),dt_features, hist_len,tune_cif, tune_cif_w, true);
[XD_snt_te,events_times_snt_te,synth_CIF_snt_te,wrd_event_snt_te] =get_data(data_w,raw_w,data_s,coeffs, sentence_IDs(1),dt_features, hist_len,tune_cif, tune_cif_w, true);
%% prediction process
[yhat_te_pre,GLM_coef] = prediction_process(XD_snt_tr,synth_CIF_snt_tr,XD_snt_te,Cross_val_K);

%% build the prior model
[prior_filter_events_times,prior_filter_events_delay,delays]=build_prior_test(events_times_snt_tr,XD_snt_te);

%% Direct decoder
thr_spike=.05;
[numb_snt,numb_evnts]=size(events_times_snt_te);
[y_hat_te]= run_DDD(yhat_te_pre,delays,prior_filter_events_times,prior_filter_events_delay,numb_evnts,thr_spike);
figure
plot(synth_CIF_snt_te,'r')
hold on
plot(yhat_te_pre,'b')
plot(y_hat_te,'k')
hold off


%% online decoding

% y_hat_ep = zeros(length(yhat_te_pre),1);
% t_past=0;
% event_ID=1;
% for t=1:length(y_hat_ep)
%     audio_ep=wavrecord(fs/100,fs);
%     [coeffs_ep,delta_ep,deltaDelta_ep,loc_ep] = mfcc(audio_ep,fs);
%     XD_snt_ep= design_matrix(coeffs_ep,hist_len);
% 
%    if (event_ID <= numb_evnts )
%       history_diff(t) = (t - t_past)/length(y_hat_ep);
% 
%       if ((t - t_past) >= mean(delays(:, event_ID)))
%           y_hat_te(t) = glmval(GLM_coef,XD_snt_ep(end,:),'log');
%           y_hat_te(t) = y_hat_te(t)*prior_filter_events_times(event_ID, t);
% 
%           if  y_hat_te(t)>thr_spike
%                   
%                         event_hat_te(t)=1;
%                         event_ID = event_ID + 1;
%                         t_past = t;
%           end
%       end
%    end
% end
% y_hat_te=y_hat_te/max(y_hat_te);
% 
% figure
% plot(synth_CIF_snt_te,'r')
% hold on
% plot(y_hat_te,'b')
% hold off
