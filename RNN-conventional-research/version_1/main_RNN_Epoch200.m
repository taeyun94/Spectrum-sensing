% clc
% clear
% function main_RNN(Ep, Numcell)
 Ep = 200;
 Numcell = 3;
load conventional_rnn_trinainig_data_160000

idx = randperm(size(XTrain,1),160000/4);
XValidation = XTrain(idx,1);
XTrain(idx,:) = [];
YValidation = YTrain(idx,1);
YTrain(idx,:) = [];

YValidation = categorical(YValidation,[1 0],{'ON','OFF'});
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
    sequenceInputLayer(1) % scalar
        
    lstmLayer(Numcell,'OutputMode','last')
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
    ];

options = trainingOptions('adam', ...
    'ExecutionEnvironment','auto',...
    'InitialLearnRate',0.001, ...
    'shuffle','every-epoch', ...
    'MaxEpochs',Ep, ...
    'MiniBatchSize', 32, ...
    'ValidationData', {XValidation, YValidation}, ...
    'ValidationFrequency', 3750, ...
    'plots','training-progress', ...
    'Verbose',false);

[net, info] = trainNetwork(XTrain,YTrain,layers,options);

SS = sprintf('./network/Hyper_net_Epoch%d_LSTMcell_%d', Ep, Numcell);
save(SS, 'net', 'info');


%     'LearnRateDropFactor',0.9, ...
%     'LearnRateDropPeriod',1, ...
%     'InitialLearnRate',0.001, ...