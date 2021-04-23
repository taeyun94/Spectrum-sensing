clear
% close all

S = sprintf('net_AWGN_noise_2');
load(S);

SS = sprintf('test_data_set_AWGN_noise_2.mat');
load(SS);

XData = XTest(:,:,1,:,12);
YData = YTest(:, 12);

% XTest = XData;
% YTest = YData;

YData = categorical(YData,[1 0],{'ON','OFF'});

FDR = zeros(length(YData(1,:)),1); MDR = zeros(length(YData(1,:)),1); ACC = zeros(length(YData(1,:)),1);
total_ON_count = 0;
total_OFF_count = 0;

for loop=1:length(YData(1,:))
    temp = classify(net,XData(:,:,:,:,loop));
    buffer1 = [];
    buffer2 = [];
    
    for n=1:length(temp)
        if YData(n,loop) == 'ON'
            if temp(n) ~= YData(n,loop)
                FDR(loop) = FDR(loop) + 1;
                buffer1 = [buffer1;n];
            end
            total_ON_count = total_ON_count + 1;
        else
            if temp(n) ~= YData(n,loop)
                MDR(loop) = MDR(loop) + 1;
                buffer2 = [buffer2;n];
            end
            total_OFF_count = total_OFF_count + 1;
        end
    end
    FDR = FDR/total_ON_count;
    MDR = MDR/total_OFF_count;
    ACC = 1 - sum(temp~=YData(:,1))/length(temp);
end

% SNR = -20:2:6;
% SSS = sprintf('./performance/performance_First_distacne_5=%dm', distance);
% save(SSS, 'MDR', 'FDR', 'ACC', 'SNR');


% for q = 1:length(XData(:,:,1,:))
%     XData(:,:,1,q) = XData(:,:,1,q)/max(max(XData(:,:,1,q)));
% end

% temp_data=zeros(32,64,1,length(buffer));
% for n=1:length(buffer)
%         a=XData(:,:,1,buffer(n));
%         a=a/max(max(a));
%         temp_data(:,:,1,n)=a;
% end
% t=classify(net,temp_data);


