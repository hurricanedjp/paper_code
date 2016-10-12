clear all;close all;clc;
addpath('C:\Users\Rhyme\Desktop\new train');

train_x=[];

for i = 1 : 530
    imgpath=sprintf('%s%d%s','a (',i,').png');
    img=imread(imgpath);
    gray=rgb2gray(img);
    %grayimg=im2bw(img);
    GrayImgRe=imresize(gray,[70,70]);
    mat=reshape(GrayImgRe,1,numel(GrayImgRe));
    train_x=[train_x;mat];
end

train_y=[];
for i = 1 : 530
   if i>=1&&i<=40
       tag=[1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
       train_y=[train_y;tag];
   end
   if i>=41&&i<=80
       tag=[0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
       train_y=[train_y;tag];
   end
   if i>=81&&i<=120
       tag=[0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
       train_y=[train_y;tag];
   end
   if i>=121&&i<=160
       tag=[0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
       train_y=[train_y;tag];
   end
   if i>=161&&i<=200
       tag=[0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0];
       train_y=[train_y;tag];
   end
   if i>=201&&i<=240
       tag=[0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0];
       train_y=[train_y;tag];
   end
   if i>=241&&i<=280
       tag=[0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0];
       train_y=[train_y;tag];
   end
   if i>=281&&i<=320
       tag=[0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0];
       train_y=[train_y;tag];
   end
   if i>=321&&i<=360
       tag=[0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0];
       train_y=[train_y;tag];
   end
   if i>=361&&i<=400
       tag=[0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0];
       train_y=[train_y;tag];
   end
   if i>=401&&i<=420
       tag=[0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0];
       train_y=[train_y;tag];
   end
   if i>=421&&i<=440
       tag=[0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0];
       train_y=[train_y;tag];
   end
   if i>=441&&i<=460
       tag=[0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0];
       train_y=[train_y;tag];
   end
   if i>=461&&i<=480
       tag=[0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0];
       train_y=[train_y;tag];
   end
   if i>=481&&i<=500
       tag=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0];
       train_y=[train_y;tag];
   end
    if i>=501&&i<=510
       tag=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0];
       train_y=[train_y;tag];
    end
    if i>=511&&i<=520
       tag=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0];
       train_y=[train_y;tag];
    end
    if i>=521&&i<=530
       tag=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1];
       train_y=[train_y;tag];
   end
end

test_x=[];
for i = 1 : 435
    imgpath=sprintf('%s%d%s','C:\Users\Rhyme\Desktop\new test\b (',i,').png');
    img=imread(imgpath);
    gray=rgb2gray(img);
    %grayimg=im2bw(img);
    GrayImgRe=imresize(gray,[70,70]);
    mat=reshape(GrayImgRe,1,numel(GrayImgRe));
    test_x=[test_x;mat];
end

test_y=[];
for i = 1 : 435
   if i>=1&&i<=30
       tag=[1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
       test_y=[test_y;tag];
   end
   if i>=31&&i<=60
       tag=[0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
       test_y=[test_y;tag];
   end
   if i>=61&&i<=90
       tag=[0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
       test_y=[test_y;tag];
   end
   if i>=91&&i<=120
       tag=[0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
       test_y=[test_y;tag];
   end
   if i>=121&&i<=150
       tag=[0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0];
       test_y=[test_y;tag];
   end
   if i>=151&&i<=180
       tag=[0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0];
       test_y=[test_y;tag];
   end
   if i>=181&&i<=210
       tag=[0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0];
       test_y=[test_y;tag];
   end
   if i>=211&&i<=240
       tag=[0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0];
       test_y=[test_y;tag];
   end
   if i>=241&&i<=270
       tag=[0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0];
       test_y=[test_y;tag];
   end
   if i>=271&&i<=300
       tag=[0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0];
       test_y=[test_y;tag];
   end
      if i>=301&&i<=315
       tag=[0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0];
       test_y=[test_y;tag];
   end
   if i>=316&&i<=330
       tag=[0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0];
       test_y=[test_y;tag];
   end
   if i>=331&&i<=345
       tag=[0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0];
       test_y=[test_y;tag];
   end
   if i>=346&&i<=360
       tag=[0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0];
       test_y=[test_y;tag];
   end
   if i>=361&&i<=375
       tag=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0];
       test_y=[test_y;tag];
   end
    if i>=376&&i<=395
       tag=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0];
       test_y=[test_y;tag];
   end
   if i>=396&&i<=415
       tag=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0];
       test_y=[test_y;tag];
   end
   if i>=416&&i<=435
       tag=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1];
       test_y=[test_y;tag];
   end
end

save data70  train_x train_y test_x test_y;

figure;imshow(gray);