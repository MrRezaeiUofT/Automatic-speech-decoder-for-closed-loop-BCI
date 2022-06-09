close all
clear all
clc

recObj = audiorecorder(48000,16,1,4);
disp('Start speaking.')
recordblocking(recObj, 5);
disp('End of Recording.');
play(recObj);
y = getaudiodata(recObj);