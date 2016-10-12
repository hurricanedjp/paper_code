clear all; close all; clc;
addpath('../data');
addpath('../util');
load data4848;
load sjwl367;

train_x = double(reshape(train_x',46,46,530))/255;
test_x = double(reshape(test_x',46,46,435))/255;
train_y = double(train_y');
test_y = double(test_y');




% 然后就用测试样本来测试
[mmy, er, bad] = cnntest(cnn, test_x, test_y);
%[mmy2, er2, bad2] = cnntest(cnn, train_x, train_y);


%plot mean squared error
plot(cnn.rL);
%show test error
disp([num2str(er*100) '% error']);
