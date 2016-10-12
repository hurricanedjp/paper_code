clear all;close all;clc;
addpath('C:\Users\Administrator\Desktop\dingjianping\资料整理-彭博\车标训练测试集\new train b');

train_x=[];

for i = 1 : 596
    imgpath=sprintf('%s%d%s','a (',i,').png');
    img=imread(imgpath);
    gray=rgb2gray(img);
    %grayimg=im2bw(img);
    GrayImgRe=imresize(gray,[48,48]);
    mat=reshape(GrayImgRe,1,numel(GrayImgRe));
    train_x=[train_x;mat];
end

train_y=[];
for i = 1 : 596
   if i>=1&&i<=50
       tag=[1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
       train_y=[train_y;tag];
   end
   if i>=51&&i<=90
       tag=[0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
       train_y=[train_y;tag];
   end
   if i>=91&&i<=140
       tag=[0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
       train_y=[train_y;tag];
   end
   if i>=141&&i<=190
       tag=[0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
       train_y=[train_y;tag];
   end
   if i>=191&&i<=230
       tag=[0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0];
       train_y=[train_y;tag];
   end
   if i>=231&&i<=280
       tag=[0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0];
       train_y=[train_y;tag];
   end
   if i>=281&&i<=320
       tag=[0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0];
       train_y=[train_y;tag];
   end
   if i>=321&&i<=360
       tag=[0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0];
       train_y=[train_y;tag];
   end
   if i>=361&&i<=400
       tag=[0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0];
       train_y=[train_y;tag];
   end
   if i>=401&&i<=440
       tag=[0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0];
       train_y=[train_y;tag];
   end
   if i>=441&&i<=466
       tag=[0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0];
       train_y=[train_y;tag];
   end
   if i>=467&&i<=486
       tag=[0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0];
       train_y=[train_y;tag];
   end
   if i>=487&&i<=511
       tag=[0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0];
       train_y=[train_y;tag];
   end
   if i>=512&&i<=536
       tag=[0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0];
       train_y=[train_y;tag];
   end
   if i>=537&&i<=561
       tag=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0];
       train_y=[train_y;tag];
   end
    if i>=562&&i<=576
       tag=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0];
       train_y=[train_y;tag];
    end
    if i>=577&&i<=586
       tag=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0];
       train_y=[train_y;tag];
    end
    if i>=587&&i<=596
       tag=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1];
       train_y=[train_y;tag];
   end
end

test_x=[];
for i = 1 : 501
    imgpath=sprintf('%s%d%s','C:\Users\Administrator\Desktop\dingjianping\资料整理-彭博\车标训练测试集\new test b\b (',i,').png');
    img=imread(imgpath);
    gray=rgb2gray(img);
    %grayimg=im2bw(img);
    GrayImgRe=imresize(gray,[48,48]);
    mat=reshape(GrayImgRe,1,numel(GrayImgRe));
    test_x=[test_x;mat];
end

test_y=[];
for i = 1 : 501
   if i>=1&&i<=40
       tag=[1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
       test_y=[test_y;tag];
   end
   if i>=41&&i<=70
       tag=[0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
       test_y=[test_y;tag];
   end
   if i>=71&&i<=110
       tag=[0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
       test_y=[test_y;tag];
   end
   if i>=111&&i<=150
       tag=[0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
       test_y=[test_y;tag];
   end
   if i>=151&&i<=180
       tag=[0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0];
       test_y=[test_y;tag];
   end
   if i>=181&&i<=220
       tag=[0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0];
       test_y=[test_y;tag];
   end
   if i>=221&&i<=250
       tag=[0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0];
       test_y=[test_y;tag];
   end
   if i>=251&&i<=280
       tag=[0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0];
       test_y=[test_y;tag];
   end
   if i>=281&&i<=310
       tag=[0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0];
       test_y=[test_y;tag];
   end
   if i>=311&&i<=340
       tag=[0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0];
       test_y=[test_y;tag];
   end
      if i>=341&&i<=361
       tag=[0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0];
       test_y=[test_y;tag];
   end
   if i>=362&&i<=376
       tag=[0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0];
       test_y=[test_y;tag];
   end
   if i>=377&&i<=396
       tag=[0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0];
       test_y=[test_y;tag];
   end
   if i>=397&&i<=416
       tag=[0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0];
       test_y=[test_y;tag];
   end
   if i>=417&&i<=436
       tag=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0];
       test_y=[test_y;tag];
   end
    if i>=437&&i<=461
       tag=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0];
       test_y=[test_y;tag];
   end
   if i>=462&&i<=481
       tag=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0];
       test_y=[test_y;tag];
   end
   if i>=482&&i<=501
       tag=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1];
       test_y=[test_y;tag];
   end
end

save data_hurricane train_x train_y test_x test_y;

figure;imshow(gray);