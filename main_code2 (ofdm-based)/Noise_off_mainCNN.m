function Noise_off_mainCNN(time_block,XTrain_cnn,YTrain)

% S = sprintf('Noise_off_training_data_OFDM_timeblock_%d.mat', time_block);
% load(S);

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
    imageInputLayer([16 time_block])
    
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
    'MaxEpochs',2, ...
    'MiniBatchSize', 31*20, ...
    'plots','training-progress', ...
    'Verbose',false);

net = trainNetwork(XTrain_cnn,YTrain,layers,options);

SS = sprintf('./network/Noise_off_CNN_net_timeblock_%d', time_block);
save(SS, 'net');



