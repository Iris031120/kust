% 清空环境
clear; clc; close all;

%% 1. 读取图片
img = imread('beans.jpg');  % 将beans.jpg替换为你的图片路径
figure;
imshow(img);
title('原始图像');

%% 2. 转换到HSV颜色空间（便于分离红色和绿色）
hsvImg = rgb2hsv(img);

H = hsvImg(:,:,1); % 色调 Hue
S = hsvImg(:,:,2); % 饱和度 Saturation
V = hsvImg(:,:,3); % 亮度 Value

%% 3. 提取红豆区域
% 红色Hue通常接近0或者接近1
redMask = ( (H < 0.05 | H > 0.95) & S > 0.4 & V > 0.2 );

% 形态学处理，去掉小噪点
redMask = imopen(redMask, strel('disk',3));
redMask = imclose(redMask, strel('disk',5));
redMask = imfill(redMask, 'holes');

%% 4. 提取绿豆区域
% 绿色Hue大约在0.25~0.45之间
greenMask = ( H > 0.25 & H < 0.45 & S > 0.3 & V > 0.2 );

% 形态学处理，去掉小噪点
greenMask = imopen(greenMask, strel('disk',3));
greenMask = imclose(greenMask, strel('disk',5));
greenMask = imfill(greenMask, 'holes');

%% 5. 标记红豆
redStats = regionprops(redMask, 'BoundingBox', 'Centroid');
numRedBeans = numel(redStats);

%% 6. 标记绿豆
greenStats = regionprops(greenMask, 'BoundingBox', 'Centroid');
numGreenBeans = numel(greenStats);

%% 7. 显示结果
figure;
imshow(img);
hold on;

% 标注红豆（红色框）
for i = 1:numRedBeans
    thisBB = redStats(i).BoundingBox;
    rectangle('Position', thisBB, 'EdgeColor', 'r', 'LineWidth', 2);
    text(thisBB(1), thisBB(2)-10, sprintf('Red %d', i), ...
        'Color', 'r', 'FontSize', 10, 'FontWeight', 'bold');
end

% 标注绿豆（绿色框）
for i = 1:numGreenBeans
    thisBB = greenStats(i).BoundingBox;
    rectangle('Position', thisBB, 'EdgeColor', 'g', 'LineWidth', 2);
    text(thisBB(1), thisBB(2)-10, sprintf('Green %d', i), ...
        'Color', 'g', 'FontSize', 10, 'FontWeight', 'bold');
end

title(sprintf('识别结果: 红豆=%d, 绿豆=%d', numRedBeans, numGreenBeans));
hold off;

%% 8. 输出数量
fprintf('红豆数量: %d\n', numRedBeans);
fprintf('绿豆数量: %d\n', numGreenBeans);
