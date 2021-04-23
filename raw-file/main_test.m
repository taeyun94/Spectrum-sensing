function main_test(Z)

S = sprintf('net_AWGN_noise_%d', Z);
load(S);

SS = sprintf('test_data_set_AWGN_noise_%d_scale.mat', Z);
load(SS);

YTest = categorical(YTest,[1 0],{'ON','OFF'});

FDR = zeros(length(YTest(1,:)),1); MDR = zeros(length(YTest(1,:)),1); ACC = zeros(length(YTest(1,:)),1);
total_ON_count = 0;
total_OFF_count = 0;
for loop=1:length(YTest(1,:))
    temp = classify(net,XTest(:,:,:,:,loop));
    
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

SNR = -20:2:6;

SSS = sprintf('performance_noise_%d_scale', Z);

save(SSS, 'MDR', 'FDR', 'ACC', 'SNR');


