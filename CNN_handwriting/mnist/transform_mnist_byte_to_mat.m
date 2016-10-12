clear all; close all; clc;
addpath('C:\Users\HURRICANE\Desktop\CNN_handwriting\util');

train_x = fopen('C:\Users\Administrator\Desktop\dingjianping\CNN_handwriting\mnist\train-images.idx3-ubyte', 'rb');  %以二进制形式打开文件
train_y = fopen('C:\Users\Administrator\Desktop\dingjianping\CNN_handwriting\mnist\train-labels.idx1-ubyte', 'rb');
test_x = fopen('C:\Users\Administrator\Desktop\dingjianping\CNN_handwriting\mnist\t10k-images.idx3-ubyte', 'rb');
test_y = fopen('C:\Users\Administrator\Desktop\dingjianping\CNN_handwriting\mnist\t10k-labels.idx1-ubyte', 'rb');

train_x = fread(train_x);               %从二进制文件读取数据
train_y = fread(train_y);
test_x = fread(test_x);
test_y = fread(test_y);

train_x = train_x(17:47040016,:);
train_y = train_y(9:60008,:);
test_x = test_x(17:7840016,:);
test_y = test_y(9:10008,:);

train_x = reshape(train_x,784,60000);  %reshape按列读取
test_x = reshape(test_x,784,10000);
train_y = train_y';
test_y = test_y';

tr_y = zeros(10,60000);
te_y =zeros(10,10000);
for i=1:60000
    k=train_y(i)+1;
    tr_y(k,i)=1;
end
for j=1:10000
    l=test_y(j)+1;
    te_y(l,j)=1;
end

train_y=tr_y;
test_y=te_y;

train_x=uint8(train_x');
test_x=uint8(test_x');
train_y=uint8(train_y');
test_y=uint8(test_y');

save mnist_uint8 train_x test_x train_y test_y






