function Total_sig_gen_training(time_block)
% clear;

Fs = 16; % sampling frequency in MHz
BW =0.5; % bandwidth in MHz
% N_ch = Fs/BW; % total number of channels

N_fft = 512;
L_overlap = N_fft/2;
N_time_block = time_block;

N_training = 20000;

roll_off = 0.22;
over_s1 = 4; % oversampling ratio for PSF
over_s2 = 2; % oversampling ratio for 1st interpolation
over_s3 = 5; % oversampling ratio for 2nd interpolation
over_s = over_s1*over_s2*over_s3; % total oversampling ratio

L_time_total = (N_fft-L_overlap)*(N_time_block-1)+N_fft;    %(512-256)*(64-1)+512  -> 16640
N_sym = ceil(L_time_total/over_s)+1;    % 416+1

psf = rcosdesign(roll_off,30,over_s1,'normal');   % rcosdesign 필터 설명
intp_flt1 = rcosdesign(0.5,20,over_s2*2,'normal');  % *2에 대한 설명
intp_flt2 = rcosdesign(0.8,20,over_s3*2,'normal');
psf = psf/sum(psf);
intp_flt1 = intp_flt1/sum(intp_flt1);
intp_flt2 = intp_flt2/sum(intp_flt2);

mod_idx = 4; % 4:QPSK, 32:32QAM, 64:64QAM

% candidates of carrier frequencies
fc_table = -Fs/2+BW:BW:+Fs/2-BW;
fc_table = fc_table + max(fc_table);

width = round(N_fft/(Fs/BW));
N_ch = length(fc_table);

XTrain_cnn = zeros(width, N_time_block, 1, N_training*N_ch);
YTrain = zeros(N_training*N_ch, 1);

for loop=1:N_training
    
    SNR = rand(1)*70-20; % SNR -20~50dB random
    
    ch_status = randi([0 1],length(fc_table),1);
    YTrain(N_ch*(loop-1)+1:N_ch*loop) = ch_status;
    clear tx_sig
    for n=1:length(fc_table)
        if ch_status(n) == 1
            data_t = randi([0 mod_idx-1], N_sym, 1);
            data = [data_t; data_t; data_t];
            SYM_t = qammod(data, mod_idx)*sqrt(0.5);
            % Pulse shaping filter
            temp = upfirdn(SYM_t,psf,over_s1,1)*sqrt(over_s1);
            psf_out = temp((length(psf)+1)/2:end-(length(psf)-1)/2);
            % interpolation (x2)
            temp = upfirdn(psf_out,intp_flt1,over_s2,1)*sqrt(over_s2);
            intp_out1 = temp((length(intp_flt1)+1)/2:end-(length(intp_flt1)-1)/2);
            % interpolation (x5)
            temp = upfirdn(intp_out1,intp_flt2,over_s3,1)*sqrt(over_s3);
            intp_out2 = temp((length(intp_flt2)+1)/2:end-(length(intp_flt2)-1)/2);
            % extract center block
            if exist('tx_sig','var')
                block = intp_out2(N_sym*over_s+1:2*N_sym*over_s);
                tx_sig = tx_sig + block.*exp(1j*2*pi*fc_table(n)/Fs*[1:length(block)]');
            else
                block = intp_out2(N_sym*over_s+1:2*N_sym*over_s);
                tx_sig = block.*exp(1j*2*pi*fc_table(n)/Fs*[1:length(block)]');
            end
        end
    end
    
    % noise generation
    if exist('tx_sig','var')
        noise = sqrt(0.5)*crandn(size(tx_sig));
    else
        noise = sqrt(0.5)*crandn(L_time_total,1);
    end
    % rx signal + noise
    rx_sig = sqrt(10^(SNR/10))*tx_sig + noise;
    
    % rx signal scalining
    rx_sig = rx_sig/300;
    
    % rx signal vector to matrix
    rx_sig_mat = zeros(N_fft,N_time_block);
    idx = 1;
    for n=1:N_time_block
        temp = rx_sig(idx:idx+N_fft-1).*hanning(N_fft);
        temp = abs(fft(temp,N_fft)/sqrt(N_fft));
        temp = wshift('1D',temp,-round(N_fft/(Fs/BW)/2));
        rx_sig_mat(:,n) = temp/max(temp);
        idx = idx + N_fft - L_overlap;
    end
    
    for n=1:N_ch
        XTrain_cnn(:,:,1,(loop-1)*N_ch+n) = rx_sig_mat(width*(n-1)+1:width*n,:);
    end
    
    if mod(loop,1000) == 0
        disp(loop);
    end
    
end
rand_idx = randperm(N_training*N_ch);
XTrain_cnn = XTrain_cnn(:,:,1,rand_idx); % shuffle
XTrain_rnn = cell(length(XTrain_cnn(1,1,1,:)),1); % cell, used for RNN

for n=1:length(XTrain_cnn(1,1,1,:))
    XTrain_rnn{n} = XTrain_cnn(:,:,1,n);
end
YTrain = YTrain(rand_idx);

im_h = width;
im_w = N_time_block;

S = sprintf('C_RNN_training_data_timeblock_%d.mat', N_time_block);
save(S, 'XTrain_cnn', 'XTrain_rnn', 'YTrain', 'im_h', 'im_w', '-v7.3');


% figure(1)
% imshow(rx_sig_mat);
% figure(2)
% imshow(XTrain_cnn(:,:,1,1));
% figure(3)
% imshow(XTrain_rnn{1});


