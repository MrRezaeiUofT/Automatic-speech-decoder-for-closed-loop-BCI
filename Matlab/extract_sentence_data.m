function [coeffs_snt,out_data, synth_CIF,events_times]= extract_sentence_data(snet_id,data_w, raw_w,data_s,coeffs,dt_features, zero_pad_len, tune_cif, tune_cif_w)
%% extract sentence data
data_w_temp = data_w(find(data_w(:,7) == snet_id),:);
raw_w_temp = raw_w(find(data_w(:,7) == snet_id),:);
snt_duration = data_s(snet_id+1,2);
snt_onset = data_s(snet_id+1,1);
%% get acustic and behavioral data of the sentence
snt_onset_indx = (floor(snt_onset /dt_features));
snt_duration_len = (ceil(snt_duration / dt_features));
coeffs_snt = coeffs(snt_onset_indx:snt_onset_indx + snt_duration_len,:);
[M,N]=size(coeffs_snt);
out_data= zeros(length(coeffs_snt),4);
last_phoneme_time=0;
for ii = 1: length(out_data)+1
    time_int = [snt_onset+(ii-1)*dt_features,snt_onset+(ii)*dt_features];
    identifier_phones = find((data_w_temp(:,1) <=time_int(2)) & (data_w_temp(:,1) >=time_int(1)));
    
    if (length(identifier_phones) ~= 0)  
        if ~strcmp( char(raw_w_temp(identifier_phones,9)), 'sp')
%        char(raw_w_temp(identifier_phones,9))
%         One phoneme for the time interval
        identified_phone_duration = data_w_temp(identifier_phones,2);
        out_data(ii,1) = 1;
        out_data(ii,2) =floor( identified_phone_duration/dt_features);
        out_data(ii,3) = ii - last_phoneme_time;
        last_phoneme_time = ii;
        out_data(ii,4)=data_w_temp(identifier_phones,10);
        end
    end
end
%% add zero pads
out_data=cat(1, zeros(zero_pad_len,4),out_data);
coeffs_snt=cat(1,zeros(zero_pad_len,N),coeffs_snt);
%% get the generated CIF, true means fine_tune the events 
[synth_CIF,events_times] = get_tuned_cif(coeffs_snt,out_data,tune_cif_w,tune_cif);
end



