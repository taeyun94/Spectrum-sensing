function Noise_off_OFDM_cnn_sig_gen_testing(time)
% clc
% clear

Fs = 16; % sampling frequency in MHz
BW =0.5; % bandwidth in MHz

N_fft = 512;
L_overlap = N_fft/2;

N_time_block = time;
N_test = 4000;

roll_off = 0.22;
over_s1 = 4; % oversampling ratio for PSF
over_s2 = 2; % oversampling ratio for 1st interpolation
over_s3 = 5; % oversampling ratio for 2nd interpolation
% over_s = over_s1*over_s2*over_s3; % total oversampling ratio
% L_time_total = (N_fft-L_overlap)*(N_time_block-1)+N_fft;

psf = rcosdesign(roll_off,30,over_s1,'normal');
intp_flt1 = rcosdesign(0.5,20,over_s2*2,'normal');
intp_flt2 = rcosdesign(0.8,20,over_s3*2,'normal');
psf = psf/sum(psf);
intp_flt1 = intp_flt1/sum(intp_flt1);
intp_flt2 = intp_flt2/sum(intp_flt2);

% candidates of carrier frequencies
fc_table = -Fs/2+BW:BW:+Fs/2-BW;
fc_table = fc_table + max(fc_table);

width = round(N_fft/(Fs/BW));
N_ch = length(fc_table);

N_fftpt = 64;
N = 48;
L_cp = N_fftpt - N;
OFDM_symbol=64;
tx_sym=zeros(1,(N_fftpt+L_cp)*OFDM_symbol);
mod_Num = 2;      %%%% 1..BPSK  2..QPSK  3..16-QAM
trel = poly2trellis(7,[171 133]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Training data set generation
SNR_table = [-20:2:6];

XTest_cnn = zeros(width, N_time_block, 1, N_test*N_ch,length(SNR_table));
YTest = zeros(N_test*N_ch, length(SNR_table));

for SNR_loop=1:length(SNR_table)
    N_time_block
    SNR_table(SNR_loop)
    SNR = SNR_table(SNR_loop);
    
    for loop=1:N_test
        
        ch_status = randi([0 1],length(fc_table),1);
        YTest(N_ch*(loop-1)+1:N_ch*loop, SNR_loop) = ch_status;
        clear tx_sig
        for n=1:length(fc_table)
            if ch_status(n) == 1
                for nn=1:OFDM_symbol
                    msg = randi([0 1],1,N);
                    conv_msg = convenc(msg,trel);
                    mod_msg = mod_sel(conv_msg,mod_Num);
                    
                    pilot_msg = pilot_insertion(mod_msg);
                    sort_msg = ifft_sort(pilot_msg,N_fftpt);
                    ifft_msg = ifft_make(sort_msg,N_fftpt);
                    data = cp_insert(ifft_msg,N_fftpt);
                    tx_sym((N_fftpt+L_cp)*(nn-1)+1:(N_fftpt+L_cp)*nn)=data;
                end
                gain = sqrt(mean(abs(tx_sym).^2));
                tx_sym = tx_sym/gain;
                
                % Pulse shaping filter
                temp = upfirdn(tx_sym,psf,over_s1,1)*sqrt(over_s1);
                psf_out = temp((length(psf)+1)/2:end-(length(psf)-1)/2);
                % interpolation (x2)
                temp = upfirdn(psf_out,intp_flt1,over_s2,1)*sqrt(over_s2);
                intp_out1 = temp((length(intp_flt1)+1)/2:end-(length(intp_flt1)-1)/2);
                % interpolation (x5)
                temp = upfirdn(intp_out1,intp_flt2,over_s3,1)*sqrt(over_s3);
                intp_out2 = temp((length(intp_flt2)+1)/2:end-(length(intp_flt2)-1)/2);
                % extract center block
                if exist('tx_sig','var')
                    tx_sig = tx_sig + intp_out2.*exp(1j*2*pi*fc_table(n)/Fs*[1:length(intp_out2)]);
                else
                    tx_sig = intp_out2.*exp(1j*2*pi*fc_table(n)/Fs*[1:length(intp_out2)]);
                end
            end
        end
        
        tx_sig = tx_sig.';
        
        % rx singal noise generation
        noise = sqrt(0.5)*crandn(size(tx_sig));
        if exist('tx_sig','var')
            rx_sig = sqrt(10^(SNR/10))*tx_sig + noise;
        else
            rx_sig =  noise;
        end
        
        % rx signal scalining
        rx_sig = (rx_sig/300);
        
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
            XTest_cnn(:,:,1,(loop-1)*N_ch+n,SNR_loop) = rx_sig_mat(width*(n-1)+1:width*n,:);
        end
        
        if mod(loop,1000) == 0
            disp(loop);
        end
    end
end

S = sprintf('Noise_off_test_data_OFDM_cnn_timeblock_%d.mat', N_time_block);
save(S, 'XTest_cnn', 'YTest', '-v7.3');

% figure(1);
% imshow(rx_sig_mat);
% figure(2);
% imshow(XTest(:,:,1,1,1));

% fig4=figure(6);
% [pxx, f] = pwelch(rx_sig , [], [], [], Fs,'centered','power');
% plot(f,10*log10(pxx),'b','LineWidth',1.5);
% xlabel('frequency (MHz)'); ylabel('power spectral density (dB)');
% axis([-Fs/2 Fs/2 -110 0]);
% grid on;