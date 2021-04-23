clc
clear;

N_sym = 50;

mod_idx = 4; % 4:QPSK, 32:32QAM, 64:64QAM
over_s1 = 4; % oversampling ratio for PSF
psf = rcosdesign(0.22,30,over_s1,'normal'); psf = psf/sum(psf);

N_test = 40000;
SNR_table = [-20:2:8];

XTest_tmp = zeros(1, 200, N_test, length(SNR_table));
XTest = cell(N_test,length(SNR_table)); % cell, used for RNN
YTest = zeros(N_test, length(SNR_table));

for s_loop=1:length(SNR_table)
    ch_status = randi([0 1],N_test,1);
    SNR = SNR_table(s_loop);
    SNR    
    for loop=1:N_test
        Up_sample = zeros(N_sym*over_s1, 1);
        noise = sqrt(0.5)*crandn(N_sym*over_s1,1);
        if ch_status(loop) == 1
            data = randi([0 mod_idx-1], N_sym, 1);
            SYM_tx = qammod(data, mod_idx)*sqrt(0.5);
            Up_sample(1:over_s1:end) = SYM_tx;
            temp = conv(Up_sample,psf)*sqrt(over_s1);
            psf_out = temp((length(psf)+1)/2:end-(length(psf)-1)/2);
            tx_sig = sqrt(10^(SNR/10))*psf_out + noise;
        else
            tx_sig = noise;
        end
        tmp = conv(tx_sig, psf)*sqrt(over_s1);
        rx_sig = tmp((length(psf)+1)/2:end-(length(psf)-1)/2);
        
        XTest_tmp(:,:,loop,s_loop) = abs(rx_sig);
%         XTest_tmp(:,:,loop,s_loop) = 10*log(abs(rx_sig).^2);
    end
    
    for n=1:length(XTest_tmp(1,1,:,1))
        XTest{n,s_loop} = XTest_tmp(1,:,n,s_loop);
    end
    YTest(:,s_loop) = ch_status;
end

S = sprintf('RNN_test_data_input_size_%d.mat', N_sym*over_s1);
save(S, 'XTest', 'YTest', '-v7.3');


% figure(1);
% plot(abs(rx_sig));

% figure(1)
% plot(XTrain(:,:,4));

% figure(2);
% [pxx,f] = pwelch(tx_sig, [],[],[], 0.5,'centered','power');
% plot(f,10*log10(pxx),'b','LineWidth',1.5);
% xlabel('frequency (MHz)'); ylabel('power spectral density (dB)');
% % axis([-Fs/2 Fs/2 -100 -20]);
% grid on;

