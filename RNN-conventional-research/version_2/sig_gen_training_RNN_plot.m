clc
clear;

N_sym = 50;

mod_idx = 4; % 4:QPSK, 32:32QAM, 64:64QAM

over_s1 = 4; % oversampling ratio for PSF
psf = rcosdesign(0.22,30,over_s1,'normal'); psf = psf/sum(psf);

N_training = 1;
% ch_status = randi([0 1],N_training,1);
ch_status = 1;

XTrain_tmp = zeros(1, N_sym*over_s1, N_training);
YTrain = zeros(N_training, 1);

for loop=1:N_training
    
%     SNR = rand(1)*28-20; % SNR -20~8dB random
    SNR = 20;
    
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
    
    XTrain_tmp(:,:,loop) = abs(rx_sig);
%         XTrain_tmp(:,:,loop) = 10*log(abs(rx_sig).^2);
end

XTrain = cell(length(XTrain_tmp(1,1,:)),1); % cell, used for RNN
for n=1:length(XTrain_tmp(1,1,:))
    XTrain{n} = XTrain_tmp(1,:,n);
end
YTrain = ch_status;

% S = sprintf('Log_RNN_training_data_input_size_%d.mat', N_sym*over_s1);
% save(S, 'XTrain', 'YTrain', '-v7.3');

figure(1);
plot(abs(rx_sig));

figure(2);
[pxx,f] = pwelch(rx_sig, [],[],[], 0.5*4,'centered','power');
plot(f,10*log10(pxx),'b','LineWidth',1.5);
xlabel('frequency (MHz)'); ylabel('power spectral density (dB)');
% axis([-Fs/2 Fs/2 -100 -20]);
grid on;

