function [y_hat_te]= run_DDD(yhat_te_pre,delays,prior_filter_events_times,prior_filter_events_delay,numb_evnts,thr_spike)
y_hat_te = zeros(length(yhat_te_pre),1);
t_past=0;
event_ID=1;
for t=1:length(yhat_te_pre)
    %tic
   if (event_ID <= numb_evnts )
      history_diff(t) = (t - t_past)/length(yhat_te_pre);

      if true%((t - t_past) >= mean(delays(:, event_ID)))
          y_hat_te(t) =yhat_te_pre(t);
          y_hat_te(t) = y_hat_te(t)*prior_filter_events_times(event_ID, t)*prior_filter_events_delay(event_ID, t-t_past);

          if  y_hat_te(t)>thr_spike
                  
                        event_hat_te(t)=1;
                        event_ID = event_ID + 1;
                        t_past = t;
          end
      end
   end
   %toc
end
 
y_hat_te=y_hat_te/max(y_hat_te);




end