function [prior_filter_events_times,prior_filter_events_delay,delays]=build_prior_test(events_times_snt,XD_snt_te)

[numb_snt,numb_evnts]=size(events_times_snt);
prior_filter_events_times=zeros(numb_evnts,length(XD_snt_te));
prior_filter_events_delay=zeros(numb_evnts,length(XD_snt_te));
%% get delay between words
delays = zeros(numb_snt,numb_evnts);
delays(:,1)=events_times_snt(:,1);
delays(:,2:end)=abs(events_times_snt(:,2:end)-events_times_snt(:,1:end-1));
%% build prior
        for events=1:numb_evnts
            if std( events_times_snt(:, events)) == 0
                std_n = 1;
                std_d = 1;
            else
                std_n = 1*std(events_times_snt(:, events));
                std_d = 1 * std(delays(:, events));
            end   
            prior_filter_events_times(events,:) = normpdf(1:length(XD_snt_te), mean(events_times_snt(:, events)), std_n);
            prior_filter_events_times(events,:) = prior_filter_events_times(events,:)/max(prior_filter_events_times(events,:));
            prior_filter_events_delay(events, :) = normpdf(1:length(XD_snt_te),mean(delays(:, events)), std_d);
            prior_filter_events_delay(events, :) = prior_filter_events_delay(events, :) / max(prior_filter_events_delay(events, :));
        end


end