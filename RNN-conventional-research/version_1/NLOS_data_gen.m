clc
clear;

load amp24.mat % SNR 8 dB

sig_temp = XData(1:16,:,1,1);
imshow(sig_temp);

sig_temp_=ifft(sig_temp,16)*sqrt(16);

Fs = 0.5;
figure(2); 
[pxx,f] = pwelch(sig_temp_, [],[],[], Fs,'centered','power');
plot(f,10*log10(pxx),'b','LineWidth',1.5);
xlabel('frequency (MHz)'); ylabel('power spectral density (dB)');
axis([-Fs/2 Fs/2 -150 0]);
% axis([-0.25 0.25 -150 0]);
grid on;