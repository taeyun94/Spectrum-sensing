%% training set generation (CNN + RNN)
Noise_on_OFDM_sig_gen_training(4);
Noise_on_OFDM_sig_gen_training(8);
Noise_on_OFDM_sig_gen_training(16);
Noise_on_OFDM_sig_gen_training(32);
Noise_on_OFDM_sig_gen_training(64);

%% CNN, RNN network training
S = sprintf('Noise_on_training_data_OFDM_timeblock_%d.mat', 4);
load(S);
Noise_on_mainCNN(4,XTrain_cnn,YTrain);
Noise_on_mainRNN(4,XTrain_rnn,YTrain);
S = sprintf('Noise_on_training_data_OFDM_timeblock_%d.mat', 8);
load(S);
Noise_on_mainCNN(8,XTrain_cnn,YTrain);
Noise_on_mainRNN(8,XTrain_rnn,YTrain);
S = sprintf('Noise_on_training_data_OFDM_timeblock_%d.mat', 16);
load(S);
Noise_on_mainCNN(16,XTrain_cnn,YTrain);
Noise_on_mainRNN(16,XTrain_rnn,YTrain);
S = sprintf('Noise_on_training_data_OFDM_timeblock_%d.mat', 32);
load(S);
Noise_on_mainCNN(32,XTrain_cnn,YTrain);
Noise_on_mainRNN(32,XTrain_rnn,YTrain);
S = sprintf('Noise_on_training_data_OFDM_timeblock_%d.mat', 64);
load(S);
Noise_on_mainCNN(64,XTrain_cnn,YTrain);
Noise_on_mainRNN(64,XTrain_rnn,YTrain);
clear;
%% test set generation
% Noise_on_OFDM_cnn_sig_gen_testing(4);
% Noise_on_OFDM_cnn_sig_gen_testing(8);
% Noise_on_OFDM_cnn_sig_gen_testing(16);
% Noise_on_OFDM_cnn_sig_gen_testing(32);
Noise_on_OFDM_cnn_sig_gen_testing(64);

% Noise_on_OFDM_rnn_sig_gen_testing(4); 
Noise_on_OFDM_rnn_sig_gen_testing(8);
Noise_on_OFDM_rnn_sig_gen_testing(16); 
Noise_on_OFDM_rnn_sig_gen_testing(32);
Noise_on_OFDM_rnn_sig_gen_testing(64); 

%% CNN, RNN test
Noise_on_mainCNN_test(4);
% Noise_on_mainCNN_test(8);
Noise_on_mainCNN_test(16);
% Noise_on_mainCNN_test(32);
Noise_on_mainCNN_test(64);

Noise_on_mainRNN_test(4);
Noise_on_mainRNN_test(8);
Noise_on_mainRNN_test(16);
Noise_on_mainRNN_test(32);
Noise_on_mainRNN_test(64);
