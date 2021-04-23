function CNN_network_train(time_block)

S = sprintf('C_RNN_training_data_timeblock_%d.mat', time_block);
load(S);

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

net = trainNetwork(XTrain_cnn,YTrain,layers,options);

SS = sprintf('./network/CNN_net_timeblock_%d', time_block);
save(SS, 'net');



