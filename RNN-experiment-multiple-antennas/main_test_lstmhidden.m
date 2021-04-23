clear;

load('RNN_net_AWGN_timeblock_64_hidden_64.mat');

load('Experiment_NLOS_multiAnt_RNN_data.mat');

YData = categorical(YData,[1 0],{'ON','OFF'});

FDR = zeros(length(YData(1,:)),1); MDR = zeros(length(YData(1,:)),1); ACC = zeros(length(YData(1,:)),1);
total_ON_count = 0;
total_OFF_count = 0;
for loop=1:length(YData(1,:))
    temp = classify(net,XData(:,loop));
    
    for n=1:length(temp)
        if YData(n,loop) == 'ON'
            if temp(n) ~= YData(n,loop)
                FDR(loop) = FDR(loop) + 1;
            end
            total_ON_count = total_ON_count + 1;
        else
            if temp(n) ~= YData(n,loop)
                MDR(loop) = MDR(loop) + 1;
            end
            total_OFF_count = total_OFF_count + 1;
        end
    end
    FDR(loop) = FDR(loop)/total_ON_count;
    MDR(loop) = MDR(loop)/total_OFF_count;
    ACC(loop) = 1 - sum(temp~=YData(:,loop))/length(temp);
end

estimated_SNR = -12:2:10;
ACC = flip(ACC);
FDR = flip(FDR);
MDR = flip(MDR);

SSS = sprintf('RNN_NLOS_performance_multiple_ant');

save(SSS, 'MDR', 'FDR', 'ACC', 'estimated_SNR');


