clear all; close all; clc;
addpath('../data');
addpath('../util');
addpath('../libsvm-3.18/matlab')
load data512newb;
load sjwl_7533_25948_8_8_12_12_501;

train_x = double(reshape(train_x',70,70,596))/255;
test_x = double(reshape(test_x',70,70,501))/255;
train_y = double(train_y');
test_y = double(test_y');




% 然后就用测试样本来测试
[mmy, er, bad] = cnntest(cnn, test_x, test_y);
[mmy2, er2, bad2] = cnntest(cnn, train_x, train_y);

%转置
mmy=mmy';
mmy2=mmy2';

%构造Tag
train_y=[];
for i = 1 : 596
   if i>=1&&i<=50
       tag=[1];
       train_y=[train_y;tag];
   end
   if i>=51&&i<=90
       tag=[2];
       train_y=[train_y;tag];
   end
   if i>=91&&i<=140
       tag=[3];
       train_y=[train_y;tag];
   end
   if i>=141&&i<=190
       tag=[4];
       train_y=[train_y;tag];
   end
   if i>=191&&i<=230
       tag=[5];
       train_y=[train_y;tag];
   end
   if i>=231&&i<=280
       tag=[6];
       train_y=[train_y;tag];
   end
   if i>=281&&i<=320
       tag=[7];
       train_y=[train_y;tag];
   end
   if i>=321&&i<=360
       tag=[8];
       train_y=[train_y;tag];
   end
   if i>=361&&i<=400
       tag=[9];
       train_y=[train_y;tag];
   end
   if i>=401&&i<=440
       tag=[10];
       train_y=[train_y;tag];
   end
   if i>=441&&i<=466
       tag=[11];
       train_y=[train_y;tag];
   end
   if i>=467&&i<=486
       tag=[12];
       train_y=[train_y;tag];
   end
   if i>=487&&i<=511
       tag=[13];
       train_y=[train_y;tag];
   end
   if i>=512&&i<=536
       tag=[14];
       train_y=[train_y;tag];
   end
   if i>=537&&i<=561
       tag=[15];
       train_y=[train_y;tag];
   end
    if i>=562&&i<=576
       tag=[16];
       train_y=[train_y;tag];
    end
    if i>=577&&i<=586
       tag=[17];
       train_y=[train_y;tag];
    end
    if i>=587&&i<=596
       tag=[18];
       train_y=[train_y;tag];
   end
end

test_y=[];
for i = 1 : 501
   if i>=1&&i<=40
       tag=[1];
       test_y=[test_y;tag];
   end
   if i>=41&&i<=70
       tag=[2];
       test_y=[test_y;tag];
   end
   if i>=71&&i<=110
       tag=[3];
       test_y=[test_y;tag];
   end
   if i>=111&&i<=150
       tag=[4];
       test_y=[test_y;tag];
   end
   if i>=151&&i<=180
       tag=[5];
       test_y=[test_y;tag];
   end
   if i>=181&&i<=220
       tag=[6];
       test_y=[test_y;tag];
   end
   if i>=221&&i<=250
       tag=[7];
       test_y=[test_y;tag];
   end
   if i>=251&&i<=280
       tag=[8];
       test_y=[test_y;tag];
   end
   if i>=281&&i<=310
       tag=[9];
       test_y=[test_y;tag];
   end
   if i>=311&&i<=340
       tag=[10];
       test_y=[test_y;tag];
   end
      if i>=341&&i<=361
       tag=[11];
       test_y=[test_y;tag];
   end
   if i>=362&&i<=376
       tag=[12];
       test_y=[test_y;tag];
   end
   if i>=377&&i<=396
       tag=[13];
       test_y=[test_y;tag];
   end
   if i>=397&&i<=416
       tag=[14];
       test_y=[test_y;tag];
   end
   if i>=417&&i<=436
       tag=[15];
       test_y=[test_y;tag];
   end
    if i>=437&&i<=461
       tag=[16];
       test_y=[test_y;tag];
   end
   if i>=462&&i<=481
       tag=[17];
       test_y=[test_y;tag];
   end
   if i>=482&&i<=501
       tag=[18];
       test_y=[test_y;tag];
   end
end
%svm predict
m=svmtrain(train_y,mmy2,'-t 0 -b 1');
[predicted_label, accuracy, decision_values]=svmpredict(test_y,mmy,m);