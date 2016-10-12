clear all;close all;clc;
addpath('E:\Ñ¸À×ÏÂÔØ\TrainIJCNN2013\train');

train_x=[];

for i = 1 : 2058
    imgpath=sprintf('%s%d%s','a (',i,').png');
    img=imread(imgpath);
    gray=rgb2gray(img);
    %grayimg=im2bw(img);
    GrayImgRe=imresize(gray,[48,48]);
    h = fspecial('gaussian',48,4);
    GrayImRe = imfilter(GrayImgRe,h);
    mat=reshape(GrayImgRe,1,numel(GrayImgRe));
    train_x=[train_x;mat];
end

train_y=[];
for i = 1 : 2058
   if i>=1&&i<=675
       tag=[1 0 0 0];
       train_y=[train_y;tag];
   end
   if i>=676&&i<=861
       tag=[0 1 0 0];
       train_y=[train_y;tag];
   end
   if i>=862&&i<=1113
       tag=[0 0 1 0];
       train_y=[train_y;tag];
   end
   if i>=1114&&i<=2058
       tag=[0 0 0 1];
       train_y=[train_y;tag];
   end
%    if i>=423&&i<=476
%        tag=[0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0];
%        train_y=[train_y;tag];
%    end
%    if i>=477&&i<=528
%        tag=[0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0];
%        train_y=[train_y;tag];
%    end
%    if i>=529&&i<=550
%        tag=[0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0];
%        train_y=[train_y;tag];
%    end
%    if i>=551&&i<=560
%        tag=[0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0];
%        train_y=[train_y;tag];
%    end
%    if i>=561&&i<=567
%        tag=[0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0];
%        train_y=[train_y;tag];
%    end
%    if i>=568&&i<=592
%        tag=[0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0];
%        train_y=[train_y;tag];
%    end
%    if i>=593&&i<=722
%        tag=[0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0];
%        train_y=[train_y;tag];
%    end
%    if i>=723&&i<=853
%        tag=[0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0];
%        train_y=[train_y;tag];
%    end
%    if i>=441&&i<=460
%        tag=[0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0];
%        train_y=[train_y;tag];
%    end
%    if i>=461&&i<=480
%        tag=[0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0];
%        train_y=[train_y;tag];
%    end
%    if i>=481&&i<=500
%        tag=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0];
%        train_y=[train_y;tag];
%    end
%     if i>=501&&i<=510
%        tag=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0];
%        train_y=[train_y;tag];
%     end
%     if i>=511&&i<=520
%        tag=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0];
%        train_y=[train_y;tag];
%     end
%     if i>=521&&i<=530
%        tag=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1];
%        train_y=[train_y;tag];
%    end
end

test_x=[];
for i = 1 : 428
    imgpath=sprintf('%s%d%s','E:\Ñ¸À×ÏÂÔØ\TestIJCNN2013\test\b (',i,').png');
    img=imread(imgpath);
    gray=rgb2gray(img);
    %grayimg=im2bw(img);
    GrayImgRe=imresize(gray,[48,48]);
    h = fspecial('gaussian',48,4);
    GrayImRe = imfilter(GrayImgRe,h);
    mat=reshape(GrayImgRe,1,numel(GrayImgRe));
    test_x=[test_x;mat];
end

test_y=[];
for i = 1 : 428
   if i>=1&&i<=133
       tag=[1 0 0 0];
       test_y=[test_y;tag];
   end
   if i>=134&&i<=159
       tag=[0 1 0 0];
       test_y=[test_y;tag];
   end
   if i>=160&&i<=188
       tag=[0 0 1 0];
       test_y=[test_y;tag];
   end
   if i>=189&&i<=428
       tag=[0 0 0 1];
       test_y=[test_y;tag];
   end
%    if i>=121&&i<=150
%        tag=[0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0];
%        test_y=[test_y;tag];
%    end
%    if i>=151&&i<=180
%        tag=[0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0];
%        test_y=[test_y;tag];
%    end
%    if i>=181&&i<=210
%        tag=[0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0];
%        test_y=[test_y;tag];
%    end
%    if i>=211&&i<=240
%        tag=[0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0];
%        test_y=[test_y;tag];
%    end
%    if i>=241&&i<=270
%        tag=[0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0];
%        test_y=[test_y;tag];
%    end
%    if i>=271&&i<=300
%        tag=[0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0];
%        test_y=[test_y;tag];
%    end
%       if i>=301&&i<=315
%        tag=[0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0];
%        test_y=[test_y;tag];
%    end
%    if i>=316&&i<=330
%        tag=[0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0];
%        test_y=[test_y;tag];
%    end
%    if i>=331&&i<=345
%        tag=[0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0];
%        test_y=[test_y;tag];
%    end
%    if i>=346&&i<=360
%        tag=[0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0];
%        test_y=[test_y;tag];
%    end
%    if i>=361&&i<=375
%        tag=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0];
%        test_y=[test_y;tag];
%    end
%     if i>=376&&i<=395
%        tag=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0];
%        test_y=[test_y;tag];
%    end
%    if i>=396&&i<=415
%        tag=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0];
%        test_y=[test_y;tag];
%    end
%    if i>=416&&i<=435
%        tag=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1];
%        test_y=[test_y;tag];
%    end
end

save data70_¦Ò4  train_x train_y test_x test_y;

figure;imshow(GrayImRe);