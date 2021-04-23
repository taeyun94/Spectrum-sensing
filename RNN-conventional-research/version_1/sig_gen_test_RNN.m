clc
clear;

N = 100
N_sym = N*2; % Conventional RNN model (input size, 2N)
mod_idx = 4; % 4:QPSK, 32:32QAM, 64:64QAM

N_test = 40000;
SNR_table = [-20:2:4];

XTest_tmp = zeros(1, 200, N_test, length(SNR_table));
XTest = cell(N_test,length(SNR_table)); % cell, used for RNN
YTest = zeros(N_test, length(SNR_table));

for s_loop=1:length(SNR_table)
    ch_status = randi([0 1],N_test,1);
    SNR = SNR_table(s_loop);
    SNR    
    for loop=1:N_test
        
        noise = sqrt(0.5)*crandn(N_sym,1);
        if ch_status(loop) == 1
            data = randi([0 mod_idx-1], N_sym, 1);
            SYM_tx = qammod(data, mod_idx)*sqrt(0.5);
            tx_sig = sqrt(10^(SNR/10))*SYM_tx + noise;
        else
            tx_sig = noise;
        end
        XTest_tmp(:,:,loop,s_loop) = abs(tx_sig);
    end
    
    for n=1:length(XTest_tmp(1,1,:,1))
        XTest{n,s_loop} = XTest_tmp(1,:,n,s_loop);
    end
    YTest(:,s_loop) = ch_status;
end

save conventional_rnn_test_data_40000.mat 'XTest' 'YTest' '-v7.3'

% figure(1);
% plot(abs(tx_sig));

% figure(1)
% plot(XTrain(:,:,4));

% figure(2);
% [pxx,f] = pwelch(tx_sig, [],[],[], 0.5,'centered','power');
% plot(f,10*log10(pxx),'b','LineWidth',1.5);
% xlabel('frequency (MHz)'); ylabel('power spectral density (dB)');
% % axis([-Fs/2 Fs/2 -100 -20]);
% grid on;

