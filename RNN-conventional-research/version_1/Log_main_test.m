% function main_test(Ep,Numcell)
clc
clear;

Ep = 5;
% S = sprintf('./Log_network/Log_net_adam_Epoch%d_0001_batch_32', Ep);
% load(S);
load('./Log_network/Log_net_adam_Epoch5_00002_batch_32');

SS = sprintf('Log_lstm_test_data_N_100_40000.mat');
load(SS);

YTest = categorical(YTest,[1 0],{'ON','OFF'});

FDR = zeros(length(YTest(1,:)),1); MDR = zeros(length(YTest(1,:)),1); ACC = zeros(length(YTest(1,:)),1);
total_ON_count = 0;
total_OFF_count = 0;
for loop=1:length(YTest(1,:))
    temp = classify(net,XTest(:,loop));

    for n=1:length(temp)
        if YTest(n,loop) == 'ON'
            if temp(n) ~= YTest(n,loop)
                FDR(loop) = FDR(loop) + 1;
            end
            total_ON_count = total_ON_count + 1;
        else
            if temp(n) ~= YTest(n,loop)
                MDR(loop) = MDR(loop) + 1;
            end
            total_OFF_count = total_OFF_count + 1;
        end
    end
    FDR(loop) = FDR(loop)/total_ON_count;
    MDR(loop) = MDR(loop)/total_OFF_count;
    ACC(loop) = 1 - sum(temp~=YTest(:,loop))/length(temp);
end

SNR = -20:2:4;

SSS = sprintf('./Log_performance/Log_performance_adam_epoch%d_00002_batch_32', Ep);
save(SSS, 'MDR', 'FDR', 'ACC', 'SNR');

%%
figure(1)
semilogy(SNR, MDR); hold on;
semilogy(SNR, FDR); hold on;
axis([-20 4 1e-4 1]); 

figure(2)
semilogy(SNR, ACC);
axis([-20 4 0.45 1]); 
