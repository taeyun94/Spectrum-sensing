%% training set generation (CNN + RNN)
% time_table = [4,8,16,32,64];
time_table = [64,32,16,8,4];

for n=1:length(time_table)
    Noise_off_OFDM_sig_gen_training(time_table(n));
end
%% CNN, RNN network training
S = sprintf('Noise_off_training_data_OFDM_timeblock_%d.mat', 4);
load(S);
Noise_off_mainCNN(4,XTrain_cnn,YTrain);
Noise_off_mainRNN(4,XTrain_rnn,YTrain);
S = sprintf('Noise_off_training_data_OFDM_timeblock_%d.mat', 8);
load(S);
Noise_off_mainCNN(8,XTrain_cnn,YTrain);
Noise_off_mainRNN(8,XTrain_rnn,YTrain);
S = sprintf('Noise_off_training_data_OFDM_timeblock_%d.mat', 16);
load(S);
Noise_off_mainCNN(16,XTrain_cnn,YTrain);
Noise_off_mainRNN(16,XTrain_rnn,YTrain);
S = sprintf('Noise_off_training_data_OFDM_timeblock_%d.mat', 32);
load(S);
Noise_off_mainCNN(32,XTrain_cnn,YTrain);
Noise_off_mainRNN(32,XTrain_rnn,YTrain);
S = sprintf('Noise_off_training_data_OFDM_timeblock_%d.mat', 64);
load(S);
Noise_off_mainCNN(64,XTrain_cnn,YTrain);
Noise_off_mainRNN(64,XTrain_rnn,YTrain);
clear;
%% test set generation
% for n=1:length(time_table)
%     Noise_off_OFDM_cnn_sig_gen_testing(time_table(n));
%     Noise_off_OFDM_rnn_sig_gen_testing(time_table(n));
% end
% Noise_off_OFDM_cnn_sig_gen_testing(32);
% Noise_off_OFDM_cnn_sig_gen_testing(16);
Noise_off_OFDM_cnn_sig_gen_testing(8);
Noise_off_OFDM_cnn_sig_gen_testing(4);

Noise_off_OFDM_rnn_sig_gen_testing(64); 
Noise_off_OFDM_rnn_sig_gen_testing(32);
Noise_off_OFDM_rnn_sig_gen_testing(16); 
Noise_off_OFDM_rnn_sig_gen_testing(8);
Noise_off_OFDM_rnn_sig_gen_testing(4); 
%% CNN, RNN test
for n=1:length(time_table)
    Noise_off_mainCNN_test(time_table(n));
    Noise_off_mainRNN_test(time_table(n));
end
