clear all; close all; clc;
addpath('E:\���ѧϰ��CNN\CNN mat 1.0 - ����\CNN mat 1.0 - ����/data');
addpath('E:\���ѧϰ��CNN\CNN mat 1.0 - ����\CNN mat 1.0 - ����/util');
load data70;
load sjwltan;

train_x = double(reshape(train_x',48,48,864))/255;
test_x = double(reshape(test_x',48,48,1019))/255;
train_y = double(train_y');
test_y = double(test_y');




% Ȼ����ò�������������
[mmy, er, bad] = cnntest(cnn, test_x, test_y);
%[mmy2, er2, bad2] = cnntest(cnn, train_x, train_y);


%plot mean squared error
plot(cnn.rL);
%show test error
disp([num2str(er*100) '% error']);
