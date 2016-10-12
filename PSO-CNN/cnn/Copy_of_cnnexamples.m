clear all; close all; clc;
addpath('C:\Users\Administrator\Desktop\dingjianping\PSO-CNN\data');
addpath('C:\Users\Administrator\Desktop\dingjianping\PSO-CNN\util');
load data512newb48;

train_x = double(reshape(train_x',48,48,596))/255;
test_x = double(reshape(test_x',48,48,501))/255;
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
    struct('type', 'c', 'outputmaps', 12, 'kernelsize',3) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
    struct('type', 'c', 'outputmaps',16, 'kernelsize',3) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
};

% clpso�Ĳ�����ʼ��
opts.w0=0.9;
opts.w1=0.4;
opts.c=1.49445;
opts.sizepar=30;%sizeparΪ����Ⱥ������
opts.m=3; % refreshing map ��Ϊ7���μ�clpso���� E Adjusting the Refreshing gap m
cnn.Pc=zeros(1,opts.sizepar); %����ʱʹ�õ�pc��i��
cnn.flag=zeros(1,opts.sizepar); %����ÿһ�����ӵ��ж�flag������cnn�Ľṹ���ܹ���������

% ѧϰ��
opts.alpha = 1;
% ÿ������һ��batchsize��batch��ѵ����Ҳ����ÿ��batchsize�������͵���һ��Ȩֵ��������
% �����������������ˣ�������������������˲ŵ���һ��Ȩֵ
opts.batchsize = 4; 

%ѭ����������
opts.numepochs = 300;


cnn.par=cell(1,opts.sizepar+1);
cnn.vel=cell(1,opts.sizepar);
% �����cnn�����ø�cnnsetup������ݴ˹���һ��������CNN���磬������
cnn = cnnsetup(cnn, train_x, train_y,opts);
% cnn = cnnsetup_original(cnn, train_x, train_y);
% Ȼ��ʼ��ѵ��������������ʼѵ�����CNN����
cnn = cnntrain_clpso(cnn, train_x, train_y, opts );

% Ȼ����ò�������������
[mmy, er, bad] = cnntest(cnn, test_x, test_y,opts);
% [mmy, er, bad] = cnntest_original(cnn, test_x, test_y);
 %cao=[cao er*100];

   % caoi=[caoi;cao,zeros(1,i-5)];

%save i200error0320 caoi;
save PSOCNN_48_46 cnn opts;

%show test error
disp([num2str(er*100) '% error']);
