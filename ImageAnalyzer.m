clear; close all;

I1 = imread('001.png');
I2 = imread('005.png');
I3 = imread('004.png');

%To detect Face
FDetect = vision.CascadeObjectDetector;

%Returns Bounding Box values based on number of objects
BB = step(FDetect,I1);
BB2 = step(FDetect,I2);
BB3 = step(FDetect,I3);

figure,
subplot(1,2,1),
imshow(I1); hold on
for i = 1:size(BB,1)
    rectangle('Position',BB(i,:),'LineWidth',2,'LineStyle','-','EdgeColor','r');
end
title('Image 1');
hold off;
subplot(1,2,2),
imshow(I3); hold on
for i = 1:size(BB2,1)
    rectangle('Position',BB2(i,:),'LineWidth',2,'LineStyle','-','EdgeColor','r');
end
title('Image 2');

Face1 = I1(BB(3,2):(BB(3,2) + BB(3,3)), BB(3,1):(BB(3,1) + BB(3,4)));
Face2 = I2(BB2(2,2):(BB2(2,2) + BB2(2,3)), BB2(2,1):(BB2(2,1) + BB2(2,4)));
Face3 = I3(BB3(2,2):(BB3(2,2) + BB3(2,3)), BB3(2,1):(BB3(2,1) + BB3(2,4)));

[I1, E1, F2_E1] = TextDetect(Face1);
[I2, E2, F2_E2] = TextDetect(Face2);
[I3, E3, F2_E3] = TextDetect(Face3);

figure; 
subplot(2,3,1),
imshow(Face1); title('Image 1');
subplot(2,3,2);
imshow(Face1);
green = cat(3, zeros(size(E1)), ones(size(E1)), zeros(size(E1)));
hold on
h = imshow(green);
hold off
set(h, 'AlphaData', I1);
title('Hologram candidate');
subplot(2,3,3);
boxPoints = detectSURFFeatures(Face1);
imshow(Face1);
title('SURF points');
hold on;
plot(selectStrongest(boxPoints, 100));

subplot(2,3,4),
imshow(Face2); title('Image 2');
subplot(2,3,5);
imshow(Face2);
green = cat(3, zeros(size(E2)), ones(size(E2)), zeros(size(E2)));
hold on
h = imshow(green);
hold off
set(h, 'AlphaData', I2);
title('Hologram candidate');
subplot(2,3,6);
boxPoints2 = detectSURFFeatures(Face2);
imshow(Face2);
title('SURF points');
hold on;
plot(selectStrongest(boxPoints2, 100));

% I2b = I2(4:(size(I2,1)-4), 4:(size(I2,2)-4));
% 
% figure; 
% imshow(Face2(4:(size(I2,1)-4), 4:(size(I2,2)-4)));
% E2bis = E2(4:(size(I2,1)-4), 4:(size(I2,2)-4));
% green = cat(3, zeros(size(E2bis)), ones(size(E2bis)), zeros(size(E2bis)));
% red = cat(4, zeros(size(E2bis)), ones(size(E2bis)), zeros(size(E2bis)));
% hold on
% h = imshow(green);
% hold on
% set(h, 'AlphaData', I2b);
% hold off;

function [I, E, F2_E1] = TextDetect(Face)
    % Use features enhancement filter
    F2_P = edge(Face,'canny');

    % Erosion and dilation
    SE = strel('disk',1);
    F2_D = imdilate(F2_P, SE);
    F2_E = imerode(F2_D, SE);
    F2_F = imfill(F2_E,'holes');
    F2_E1 = imerode(F2_F, SE);

    % Remove small objects
    CC = bwconncomp(F2_E1);
    S = regionprops(CC, 'Area');
    L = labelmatrix(CC);
    BW2 = ismember(L, find([S.Area] >= 60));

    % Read out text from image (to do so, read text before)
    % text = ocr(BW2);

    %print(sum(sum(double(F2_D))));
    
    figure; 
    subplot(2,3,1); imshow(Face); title('Original image');
    subplot(2,3,2); imshow(F2_P); title('After Canny edge detection');
    subplot(2,3,3); imshow(F2_D); title('Erosion');
    subplot(2,3,4); imshow(F2_E); title('Dilation');
    subplot(2,3,5); imshow(F2_E1); title('Final image');
    subplot(2,3,6);  
    imshow(Face, 'InitialMag', 'fit');
    I = BW2;
    E = Face;
    green = cat(3, zeros(size(E)), ones(size(E)), zeros(size(E)));
    hold on
    h = imshow(green);
    hold off
    set(h, 'AlphaData', I);

end

