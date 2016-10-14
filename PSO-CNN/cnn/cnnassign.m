function net = cnnassign(net,num)
n = numel(net.layers); % 层数
inputmaps = 1; % 输入层只有一个特征map，也就是原始的输入图像
pos_start=1;
for l = 2 : n   %  for each layer
    if strcmp(net.layers{l}.type, 'c') % 卷积层
        % 对每一个输入map，或者说我们需要用outputmaps个不同的卷积核去卷积图像
        for j = 1 : net.layers{l}.outputmaps   %  for each output map
            for i = 1 : inputmaps   %  for each input map
                %pos_start到pos_start+net.layers{l}.kernelsize^2-1为卷积核对应离子的元素
                net.layers{l}.k_ij=net.par{num}(pos_start:pos_start+net.layers{l}.kernelsize^2-1);
                net.layers{l}.k{i}{j}=reshape(net.layers{l}.k_ij,net.layers{l}.kernelsize,net.layers{l}.kernelsize);
                %此时加上平方而不减1，位置向后移了一位，此时为偏置b的位置
                pos_start=pos_start+net.layers{l}.kernelsize^2;
            end
            
            %  add bias, pass through nonlinearity
            % 对应位置的基b
            net.layers{l}.b{j}=net.par{num}(pos_start);
            %顺移一位到下一个卷积核的起始位置
            pos_start=pos_start+1;
        end
        %  set number of input maps to this layers number of outputmaps
        inputmaps = net.layers{l}.outputmaps;
    end
end

%计算出net.ffW的行与列之积得到对应位置
net.ff_W=net.par{num}(pos_start:pos_start+numel(net.ffW)-1);
%将对应粒子元素转变权值矩阵
net.ffW=reshape(net.ff_W,size(net.ffW,1),size(net.ffW,2));

pos_start=pos_start+numel(net.ffW);
%计算出net.ffb的行与列之积得到对应位置
net.ff_b=net.par{num}(pos_start:pos_start+numel(net.ffb)-1);
%将对应粒子元素转变偏置矩阵
net.ffb=reshape(net.ff_b,size(net.ffb,1),size(net.ffb,2));
end
