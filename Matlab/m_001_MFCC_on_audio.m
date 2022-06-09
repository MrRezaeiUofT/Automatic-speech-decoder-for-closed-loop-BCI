close all
clear all
clc

[audioIn,fs] = audioread("../Datasets/DM1008/sub-DM1008_ses-intraop_task-lombard_run-03_recording-directionalmicaec_physio.wav");
for i=1:10
tic
[coeffs,delta,deltaDelta,loc] = mfcc(audioIn(1:floor(fs/20)),fs);
toc
end