% clc
% clear
function main_RNN(Ep, Numcell)
 
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
    'MiniBatchSize', 100, ...
    'ValidationData', {XValidation, YValidation}, ...
    'ValidationFrequency', 200, ...
    'plots','training-progress', ...
    'Verbose',false);

[net, info] = trainNetwork(XTrain,YTrain,layers,options);

SS = sprintf('./network/net_adam_Epoch%d_0001_LSTMcell_%d', Ep, Numcell);
save(SS, 'net', 'info');


%     'LearnRateDropFactor',0.9, ...
%     'LearnRateDropPeriod',1, ...
%     'InitialLearnRate',0.001, ...