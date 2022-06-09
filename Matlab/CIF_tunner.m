function Q = CIF_tunner(par,coeffs_snt,out_data,tune_cif_w)

tau_events=par;
history_diff=  zeros(length(coeffs_snt),1);
lambda_diff=  zeros(length(coeffs_snt),1);
numbe_events = (sum(out_data(:,1)));
events_times=zeros(1,numbe_events);
qq =1;
t_past = 0;
for t = 1:length(coeffs_snt)
     if qq <=numbe_events
       history_diff(t) =( t - t_past);
       lambda_diff(t) = exp((t - t_past)/tau_events(qq))/length(coeffs_snt);
     end
        if out_data(t,1)>0
            events_times(qq)=t;
           
            qq=qq+1;
            t_past = t;
      
     end
end
synth_CIF =(lambda_diff);
synth_CIF=synth_CIF/max(synth_CIF);

Q=mean(tune_cif_w.*abs((synth_CIF(events_times)-1)));