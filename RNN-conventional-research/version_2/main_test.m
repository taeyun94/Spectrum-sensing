clc
clear;

% S = sprintf('./network/net_adam_Epoch%d_0001_LSTMcell_%d', Ep,Numcell);
% load(S);

load ./network/never_batch100_net_adam_Ep_5_001_LSTMcell_1.mat;

SS = sprintf('RNN_test_data_input_size_%d.mat', 200);
load(SS);

YTest = categorical(YTest,[1 0],{'ON','OFF'});

FDR = zeros(length(YTest(1,:)),1); MDR = zeros(length(YTest(1,:)),1); ACC = zeros(length(YTest(1,:)),1);
total_ON_count = 0;
total_OFF_count = 0;
for loop=1:length(YTest(1,:))
    %temp = predict(net,XTest(:,:,:,:,loop));
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

SNR = -20:2:8;

SSS = sprintf('./performance/performance_ba100_Ep5_001_Cell_1');
save(SSS, 'MDR', 'FDR', 'ACC', 'SNR');

%%
% figure(1)
% semilogy(SNR, MDR); hold on;
% semilogy(SNR, FDR); hold on;
% axis([-20 4 1e-4 1]); 
% 
% figure(2)
% semilogy(SNR, ACC);
% axis([-20 4 0.5 1]); 