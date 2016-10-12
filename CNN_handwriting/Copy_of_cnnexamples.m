clear all; close all; clc;
addpath('C:\Users\Administrator\Desktop\dingjianping\CNN_handwriting\util');
load mnist_uint8.mat;

train_x = double(reshape(train_x',28,28,60000))/255;  %ѵ��������
test_x = double(reshape(test_x',28,28,10000))/255;    %���Լ�����
train_y = double(train_y');                            %ѵ������ǩ
test_y = double(test_y');                              %���Լ���ǩ 

% train_x(train_x>128)=1;
% train_x(train_x<=128)=0;
% test_x(test_x>128)=1;
% test_x(test_x<=128)=0;

%% ex1 
%will run 1 epoch in about 200 second and get around 11% error. 
%With 100 epochs you'll get around 1.2% error

   % disp([num2str(i) '/' num2str(j)]);
cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 8, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize',3) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
    struct('type', 'c', 'outputmaps',16, 'kernelsize',2) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
};

% �����cnn�����ø�cnnsetup������ݴ˹���һ��������CNN���磬������
cnn = cnnsetup(cnn, train_x, train_y);

% ѧϰ��
opts.alpha = 1;
% ÿ������һ��batchsize��batch��ѵ����Ҳ����ÿ��batchsize�������͵���һ��Ȩֵ��������
% �����������������ˣ�������������������˲ŵ���һ��Ȩֵ
opts.batchsize = 10; 
% ѵ����������ͬ��������������ѵ����ʱ��
% 1��ʱ�� 11.41% error
% 5��ʱ�� 4.2% error
% 10��ʱ�� 2.73% errorclear
opts.numepochs = 15;

% Ȼ��ʼ��ѵ��������������ʼѵ�����CNN����
cnn = cnntrain(cnn, train_x, train_y, opts , test_x , test_y);

% Ȼ����ò�������������
[mmy, er, bad] = cnntest(cnn, test_x, test_y);
 %cao=[cao er*100];

   % caoi=[caoi;cao,zeros(1,i-5)];

%save i200error0320 caoi;
save m_traf_48 cnn;
%plot mean squared error
plot(cnn.rL);
%show test error
disp([num2str(er*100) '% error']);
