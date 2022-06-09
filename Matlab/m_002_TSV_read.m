close all
clear all
clc


% patient DM1008
audio_strt_time = 63623.7147033506
[data_w, header_w, raw_w] = tsvread( "../Datasets/DM1008/sub-DM1008_ses-intraop_task-lombard_annot-produced-words.tsv" )
[data_s, header_s, raw_s] = tsvread( "../Datasets/DM1008/sub-DM1008_ses-intraop_task-lombard_annot-produced-sentences.tsv" )