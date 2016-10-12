clear all; close all; clc;
addpath('E:\Ñ¸À×ÏÂÔØ\TrainIJCNN2013\train');

    imgpath=sprintf('%s%d%s','a (',0,').png');
    Gray=imread(imgpath);
%     Gray=rgb2gray(img);
   
    imshow(Gray);
%     Y = filter2(GrayImgRe,3,'full');
%     imshow(Y);
    h = fspecial('gaussian',size(Gray),1);
    GSSImg = imfilter(Gray,h);
    figure
    imshow(GSSImg);
    h = fspecial('gaussian',size(Gray),2);
    GSSImg = imfilter(Gray,h);
    figure
    imshow(GSSImg);
     h = fspecial('gaussian',size(Gray),4);
    GSSImg = imfilter(Gray,h);
    figure
    imshow(GSSImg);
         h = fspecial('gaussian',size(Gray),8);
    GSSImg = imfilter(Gray,h);
    figure
    imshow(GSSImg);
     h = fspecial('gaussian',size(Gray),16);
    GSSImg = imfilter(Gray,h);
    figure
    imshow(GSSImg);
