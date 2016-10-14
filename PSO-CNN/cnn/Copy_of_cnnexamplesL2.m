clear all; close all; clc;
addpath('F:\鲍毛毛\CNN mat 1.0 - 副本\CNN mat 1.0 - 副本\data');
addpath('F:\鲍毛毛\CNN mat 1.0 - 副本\CNN mat 1.0 - 副本\util');
load m_data10_58;

train_x = double(reshape(train_x',58,58,1038))/255;
test_x = double(reshape(test_x',58,58,165))/255;
train_y = double(train_y');
test_y = double(test_y');

%% ex1 
%will run 1 epoch in about 200 second and get around 11% error. 
%With 100 epochs you'll get around 1.2% error

   % disp([num2str(i) '/' num2str(j)]);
cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 8, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize',4) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
    struct('type', 'c', 'outputmaps',16, 'kernelsize',3) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
};

% 这里把cnn的设置给cnnsetup，它会据此构建一个完整的CNN网络，并返回
cnn = cnnsetup(cnn, train_x, train_y);

% 学习率
opts.alpha = 1;
% 每次挑出一个batchsize的batch来训练，也就是每用batchsize个样本就调整一次权值，而不是
% 把所有样本都输入了，计算所有样本的误差了才调整一次权值
opts.batchsize = 2; 
% 训练次数，用同样的样本集。我训练的时候：
% 1的时候 11.41% error
% 5的时候 4.2% error
% 10的时候 2.73% errorclear
opts.numepochs = 600;

% 然后开始把训练样本给它，开始训练这个CNN网络
cnn = cnntrain(cnn, train_x, train_y, opts , test_x , test_y);

% 然后就用测试样本来测试
[mmy, er, bad] = cnntest(cnn, test_x, test_y);
 %cao=[cao er*100];

   % caoi=[caoi;cao,zeros(1,i-5)];

%save i200error0320 caoi;
save m_traf4_58 cnn;
%plot mean squared error
plot(cnn.rL);
%show test error
disp([num2str(er*100) '% error']);
