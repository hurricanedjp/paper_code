clear all; close all; clc;
addpath('../data');
addpath('../util');
addpath('../libsvm-3.18/matlab')
load data70;
load sjwl138;

train_x = double(reshape(train_x',70,70,530))/255;
test_x = double(reshape(test_x',70,70,435))/255;
train_y = double(train_y');
test_y = double(test_y');




% 然后就用测试样本来测试
[mmy, er, bad] = cnntest(cnn, test_x, test_y);
[mmy2, er2, bad2] = cnntest(cnn, train_x, train_y);

%转置
mmy=mmy';
mmy2=mmy2';

%构造Tag
test_y=[];
for i = 1 : 435
if i>=1&&i<=30
tag=[1];
test_y=[test_y;tag];
end
if i>=31&&i<=60
tag=[2];
test_y=[test_y;tag];
end
if i>=61&&i<=90
tag=[3];
test_y=[test_y;tag];
end
if i>=91&&i<=120
tag=[4];
test_y=[test_y;tag];
end
if i>=121&&i<=150
tag=[5];
test_y=[test_y;tag];
end
if i>=151&&i<=180
tag=[6];
test_y=[test_y;tag];
end
if i>=181&&i<=210
tag=[7];
test_y=[test_y;tag];
end
if i>=211&&i<=240
tag=[8];
test_y=[test_y;tag];
end
if i>=241&&i<=270
tag=[9];
test_y=[test_y;tag];
end
if i>=271&&i<=300
tag=[10];
test_y=[test_y;tag];
end
if i>=301&&i<=315
tag=[11];
test_y=[test_y;tag];
end
if i>=316&&i<=330
tag=[12];
test_y=[test_y;tag];
end
if i>=331&&i<=345
tag=[13];
test_y=[test_y;tag];
end
if i>=346&&i<=360
tag=[14];
test_y=[test_y;tag];
end
if i>=361&&i<=375
tag=[15];
test_y=[test_y;tag];
end
if i>=376&&i<=395
tag=[16];
test_y=[test_y;tag];
end
if i>=396&&i<=415
tag=[17];
test_y=[test_y;tag];
end
if i>=416&&i<=435
tag=[18];
test_y=[test_y;tag];
end
end

train_y=[];
for i = 1 : 530
if i>=1&&i<=40
tag=[1];
train_y=[train_y;tag];
end
if i>=41&&i<=80
tag=[2];
train_y=[train_y;tag];
end
if i>=81&&i<=120
tag=[3];
train_y=[train_y;tag];
end
if i>=121&&i<=160
tag=[4];
train_y=[train_y;tag];
end
if i>=161&&i<=200
tag=[5];
train_y=[train_y;tag];
end
if i>=201&&i<=240
tag=[6];
train_y=[train_y;tag];
end
if i>=241&&i<=280
tag=[7];
train_y=[train_y;tag];
end
if i>=281&&i<=320
tag=[8];
train_y=[train_y;tag];
end
if i>=321&&i<=360
tag=[9];
train_y=[train_y;tag];
end
if i>=361&&i<=400
tag=[10];
train_y=[train_y;tag];
end
if i>=401&&i<=420
tag=[11];
train_y=[train_y;tag];
end
if i>=421&&i<=440
tag=[12];
train_y=[train_y;tag];
end
if i>=441&&i<=460
tag=[13];
train_y=[train_y;tag];
end
if i>=461&&i<=480
tag=[14];
train_y=[train_y;tag];
end
if i>=481&&i<=500
tag=[15];
train_y=[train_y;tag];
end
if i>=501&&i<=510
tag=[16];
train_y=[train_y;tag];
end
if i>=511&&i<=520
tag=[17];
train_y=[train_y;tag];
end
if i>=521&&i<=530
tag=[18];
train_y=[train_y;tag];
end
end

%svm predict
m=svmtrain(train_y,mmy2,'-t 0');
[predicted_label, accuracy, decision_values]=svmpredict(test_y,mmy,m);