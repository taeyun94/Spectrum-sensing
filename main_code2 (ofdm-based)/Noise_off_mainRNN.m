function Noise_off_mainRNN(time_block,XTrain_rnn,YTrain)

% S = sprintf('Noise_off_training_data_OFDM_timeblock_%d', time_block);
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
width = 16;

layers = [ ...
    sequenceInputLayer(width) % observation frequency width x 2
        
    lstmLayer(64,'OutputMode','last')
    fullyConnectedLayer(32)
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
    ];

options = trainingOptions('sgdm', ...
    'ExecutionEnvironment','auto',...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.9, ...
    'LearnRateDropPeriod',1, ...
    'InitialLearnRate',0.001, ...
    'shuffle','every-epoch', ...
    'MaxEpochs',3, ...
    'MiniBatchSize', 31*20, ...
    'plots','training-progress', ...
    'Verbose',false);

net = trainNetwork(XTrain_rnn,YTrain,layers,options);

SS = sprintf('./network/Noise_off_RNN_net_timeblock_%d', time_block);
save(SS, 'net');
