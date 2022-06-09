function [yhat_te,GLM_coef] = prediction_process(XD_snt,synth_CIF_snt,XD_snt_te,K)

%% train
[B,FitInfo] = lassoglm(XD_snt,synth_CIF_snt,'poisson', 'CV',K);
lassoPlot(B,FitInfo,'plottype','CV'); 
legend('show') % Show legend
idxLambdaMinDeviance = FitInfo.IndexMinDeviance;
idxLambda1SE = FitInfo.Index1SE;
B0 = FitInfo.Intercept(idxLambda1SE);
GLM_coef = [B0; B(:,idxLambda1SE)];
%% predictions on train
yhat = glmval(GLM_coef,XD_snt,'log');
%% predictions on test
yhat_te = glmval(GLM_coef,XD_snt_te,'log');
figure
plot(yhat,'b');
hold on
plot(synth_CIF_snt,'r');
hold off
title('train_result');
legend(['true, pred']);
end