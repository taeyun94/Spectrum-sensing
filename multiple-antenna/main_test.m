function main_test(M, r)

% S = sprintf('./network/net_AWGN_M=%d_r=%d.mat', M, (r*100));
% load(S);
load('./network/net_AWGN_fft512_time64.mat');

SS = sprintf('test_dasta_set_AWGN_M=%d_r=%d.mat', M, (r*100));
load(SS);

YTest = categorical(YTest,[1 0],{'ON','OFF'});

FDR = zeros(length(YTest(1,:)),1); MDR = zeros(length(YTest(1,:)),1); ACC = zeros(length(YTest(1,:)),1);
total_ON_count = 0;
total_OFF_count = 0;
for loop=1:length(YTest(1,:))
    %temp = predict(net,XTest(:,:,:,:,loop));
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

SNR = -20:2:4;
% SSS = sprintf('./performance2/New_net_performance_M=%d, r=%d', M, (r*100));
SSS = sprintf('./performance2/performance_M=%d, r=%d', M, (r*100));

save(SSS, 'MDR', 'FDR', 'ACC', 'SNR');


