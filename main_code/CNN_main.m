%% Network train

load CNN_training_data_set_AWGN_fft512_64  % trainig set load

YTrain = categorical(YTrain,[1 0],{'ON','OFF'});

warning off parallel:gpu:device:DeviceLibsNeedsRecompiling

try
    gpuArray.eye(2)^2;
catch ME
end
try
    nnet.internal.cnngpu.reluForward(1);
catch ME
end

layers = [ ...
    imageInputLayer([im_h im_w])
    
    convolution2dLayer([3 3], 8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer([2 2],'Stride',2)
    
    convolution2dLayer([3 3], 16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer([2 2],'Stride',2)
    
    convolution2dLayer([3 3], 32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
    
    ];

options = trainingOptions('sgdm', ...
    'ExecutionEnvironment','auto',...
    'LearnRateSchedule','piecewise', ...
    'InitialLearnRate',0.001, ...
    'shuffle','every-epoch', ...
    'MaxEpochs',10, ... 
    'MiniBatchSize', 31*20, ... 
    'plots','training-progress', ...
    'Verbose',false);

net = trainNetwork(XTrain,YTrain,layers,options);

clear XTrain YTrain;
save CNN_network_timeblock_64 net;

%% TEST
load CNN_test_data_set_AWGN_fft512_64  % test set load

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
save CNN_performance_64block MDR FDR ACC SNR

figure(1); 
s1 = semilogy(SNR,MDR,'bo--','LineWidth',2); hold on;
s2 = semilogy(SNR,FDR,'bo-','LineWidth',2); hold on;
grid on;
axis([-20 6 1e-4 1]); xticks(SNR);
title('Miss & False Detection Ratio', 'fontsize', 12);
xlabel('SNR (dB)'); ylabel('Probability');
legend([s1, s2], 'Miss detection ratio', 'False detection ratio');

figure(2); 
plot(SNR,ACC,'bo-','LineWidth',2);
grid on;
axis([-20 6 0.55 1]);
xlabel('SNR (dB)'); ylabel('Total Accuracy');


