clc
clear;

time_block = 64;

S = sprintf('C_Occlusion_test_busy_%d', time_block);
load(S);
Xtest = Xtest(1:16,:,:,:);

S2 = sprintf('./network/CNN_net_timeblock_%d', time_block);
load(S2);

SNR_table = -20:2:6;

fig1 = figure(2);

for loop=1:length(SNR_table)
    img = Xtest(:,:,1,loop); 
    
    [label, score] = classify(net, img);
    if label == 'ON'
        Pred = 'Busy';
    else
        Pred = 'Idle';
    end
    
    scoreMap = occlusionSensitivity(net,img,label);
    scoreMap = rescale(scoreMap);
    
    subplot(2,length(SNR_table),loop);
    imshow(img); 
    S = sprintf('SNR %ddB', SNR_table(loop));
    title(S);
    
    s = subplot(2,length(SNR_table),loop+length(SNR_table));
    imshow(img); hold on;
    imagesc(scoreMap,'AlphaData',0.6);
    colormap(s, jet);
    S2 = sprintf("Pred: %s (%.2f)", Pred, score(label));
    title(S2);
end


