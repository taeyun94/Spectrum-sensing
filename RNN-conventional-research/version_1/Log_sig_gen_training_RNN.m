clc
clear;

N = 100;
N_sym = N*2; % Conventional RNN model (input size, 2N)
mod_idx = 4; % 4:QPSK, 32:32QAM, 64:64QAM

N_training = 160000;
% N_training = 1;
ch_status = randi([0 1],N_training,1);

XTrain_tmp = zeros(1, N_sym, N_training);
YTrain = zeros(N_training, 1);

for loop=1:N_training
    
%     SNR = rand(1)*24-20; % SNR -20~4dB random
    SNR = 8;

    noise = sqrt(0.5)*crandn(N_sym,1);
    
    if ch_status(loop) == 1
        data = randi([0 mod_idx-1], N_sym, 1);
        SYM_tx = qammod(data, mod_idx)*sqrt(0.5);
        tx_sig = sqrt(10^(SNR/10))*SYM_tx + noise;
    else
        tx_sig = noise;
    end
    XTrain_tmp(:,:,loop) = 10*log(abs(tx_sig).^2);
end

XTrain = cell(length(XTrain_tmp(1,1,:)),1); % cell, used for RNN
for n=1:length(XTrain_tmp(1,1,:))
    XTrain{n} = XTrain_tmp(1,:,n);
end
YTrain = ch_status;

S = sprintf('Log_conventional_rnn_trinainig_data_%d.mat', N_sym/2);
save(S, 'XTrain', 'YTrain', '-v7.3');
% save conventional_rnn_trinainig_data_100.mat 'XTrain' 'YTrain' '-v7.3'

% figure(1);
% plot(abs(tx_sig));
% 
% figure(2);
% plot(abs(tx_sig).^2);
% 
% figure(3);
% plot(10*log(abs(tx_sig).^2));


% figure(2);
% [pxx,f] = pwelch(tx_sig, [],[],[], 0.5,'centered','power');
% plot(f,10*log10(pxx),'b','LineWidth',1.5);
% xlabel('frequency (MHz)'); ylabel('power spectral density (dB)');
% % axis([-Fs/2 Fs/2 -100 -20]);
% grid on;
% 
