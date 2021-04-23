% clear
% close all;
function mainCNN(M, r)

S = sprintf('training_dasta_set_AWGN_M=%d_r=%d.mat', M, (r*100));
load(S);
% load training_dasta_set_AWGN_M=2_r=1;
% load training_dasta_set_AWGN_M=2_r=25;
% load training_dasta_set_AWGN_M=2_r=50;
% load training_dasta_set_AWGN_M=2_r=75;
% load training_dasta_set_AWGN_M=2_r=99;
% 
% load training_dasta_set_AWGN_M=4_r=1;
% load training_dasta_set_AWGN_M=4_r=25;
% load training_dasta_set_AWGN_M=4_r=50;
% load training_dasta_set_AWGN_M=4_r=75;
% load training_dasta_set_AWGN_M=4_r=99;

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
    'LearnRateSchedule','piecewise', ...
    'InitialLearnRate',0.001, ...
    'shuffle','every-epoch', ...
    'MaxEpochs',10, ...
    'MiniBatchSize', 31*20, ...
    'plots','training-progress', ...
    'Verbose',false);

net = trainNetwork(XTrain,YTrain,layers,options);

SS = sprintf('./network/net_AWGN_M=%d_r=%d', M, (r*100));
% save ./network/net_AWGN_M=2_r=1 net;
save(SS, 'net');


%     load test_data_set_AWGN;
% else
%     load net_AWGN;
%     load test_data_set_AWGN;
% end

%%%%%%%%%%%%%%%%%%%%%%%%
% load net2_AWGN_dropX;
% load test_data_set_AWGN;
% YTest = categorical(YTest,[1 0],{'ON','OFF'});
%
% FDR = zeros(11,1); MDR = zeros(11,1); ACC = zeros(11,1);
% total_ON_count = 0;
% total_OFF_count = 0;
% for loop=1:11
%     %temp = predict(net,XTest(:,:,:,:,loop));
%     temp = classify(net,XTest(:,:,:,:,loop));
%
%     for n=1:length(temp)
%         if YTest(n,loop) == 'ON'
%             if temp(n) ~= YTest(n,loop)
%                 MDR(loop) = MDR(loop) + 1;
%             end
%             total_ON_count = total_ON_count + 1;
%         else
%             if temp(n) ~= YTest(n,loop)
%                 FDR(loop) = FDR(loop) + 1;
%             end
%             total_OFF_count = total_OFF_count + 1;
%         end
%     end
%     MDR(loop) = MDR(loop)/total_ON_count;
%     FDR(loop) = FDR(loop)/total_OFF_count;
%     ACC(loop) = 1 - sum(temp~=YTest(:,loop))/length(temp);
% end
%
% SNR = -16:2:4;
% figure(1); hold off;
% semilogy(SNR,MDR,'bs-','LineWidth',1.5);
% grid on;
% % axis([-10 40 0. 2.5]);
% xlabel('SNR (dB)'); ylabel('Miss Detection Probability');
%
% figure(2); hold off;
% semilogy(SNR,FDR,'bs-','LineWidth',1.5);
% grid on;
% % axis([-10 40 0. 2.5]);
% xlabel('SNR (dB)'); ylabel('False Detection Probability');
%
% figure(3); hold off;
% plot(SNR,ACC,'bs-','LineWidth',1.5);
% grid on;
% % axis([-10 40 0. 2.5]);
% xlabel('SNR (dB)'); ylabel('Total Accuracy');
%
% figure(4); hold off;
% semilogy(SNR,FDR,'bs-','LineWidth',1.5);
% grid on;
% hold on;
% plot(SNR,MDR,'ro-','LineWidth',1.5);
% xlabel('SNR (dB)'); ylabel('Probability');
% legend('False detection','Miss detection')

% save performance_512FFT_64block MDR FDR ACC SNR






