function [XD_snt,events_times_snt,synth_CIF_snt,wrd_event_snt] =get_data(data_w,raw_w,data_s,coeffs, sentence_IDs,dt_features, hist_len,tune_cif, tune_cif_w,add_bias)

snet_id = sentence_IDs(1);
[coeffs_snt, wrd_event_snt,synth_CIF_snt,events_times_snt]= extract_sentence_data(snet_id,data_w, raw_w,data_s,coeffs,dt_features,hist_len,tune_cif, tune_cif_w);
% coeffs_snt=cat(2,coeffs_snt,ones(length(coeffs_snt),1));
% size(events_times_snt)
%% design matrix
XD_snt= design_matrix(coeffs_snt,hist_len);

%% add other sentences
if length(sentence_IDs)>1
for ii_snt= 2:length(sentence_IDs)
snet_id_temp = sentence_IDs(ii_snt);
[coeffs_snt_temp, wrd_event_snt_temp,synth_CIF_snt_temp,events_times_snt_temp]= extract_sentence_data(snet_id_temp,data_w, raw_w,data_s,coeffs,dt_features,hist_len,tune_cif, tune_cif_w);
% coeffs_snt_temp=cat(2,coeffs_snt_temp,ones(length(coeffs_snt_temp),1));
% size(events_times_snt)
XD_snt_temp= design_matrix(coeffs_snt_temp,hist_len);
% coeffs_snt=cat(1,coeffs_snt,coeffs_snt_temp);
wrd_event_snt=cat(1,wrd_event_snt,wrd_event_snt_temp);
synth_CIF_snt=cat(1,synth_CIF_snt,synth_CIF_snt_temp);
events_times_snt=cat(1,events_times_snt,events_times_snt_temp);
XD_snt=cat(1,XD_snt,XD_snt_temp);
% figure
% plot(synth_CIF_snt)
end

end
% add constant column
if add_bias
XD_snt = cat(2,XD_snt,ones(length(XD_snt),1));
end