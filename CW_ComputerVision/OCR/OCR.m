%% setup path
close all; clc; clear;
path = 'C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW\OCR';
cd(path);

%% Step 1: resize the image
I = imread('IMG_0653.JPG');
I = imresize(I, .3);
figure; imshow(I);

%% Detect the Face
FaceDetector = vision.CascadeObjectDetector('FrontalFaceCART','MergeThreshold',15);
BBOX = step(FaceDetector,I);
[y,x,z] = size(I);

% Rotate the picture
if isempty(BBOX)

    if x > y
        I = imrotate(I,270);
        BBOX = step(FaceDetector,I);
        [y,x,z] = size(I);
        figure; imshow(I);
    end
end

xx1 = BBOX(1) - BBOX(3)*0.7;
yy1 = BBOX(2) + BBOX(4)*1.7;
xx2 = BBOX(3) * 2.5;
yy2 = BBOX(4) * 2.1; % 2.1

save_big_image = [xx1, yy1 , xx2, yy2];

Idigits1 = insertObjectAnnotation(I, 'rectangle', [xx1, yy1 , xx2, yy2], 'ZOOM');
figure; 
imshow(Idigits1);

sub_yy2 = yy1+yy2;
sub_xx2 = xx1+xx2;

if sub_yy2 > y
   sub_yy2 = y - (y/6);
end

subIm = I(fix(yy1):fix(sub_yy2), fix(xx1):fix(sub_xx2), 1:end);
figure;
imshow(subIm);

%% Instagram filter

[height, width, channels] = size(subIm);
imshow(subIm);
hold on;
plot(height/2, width/2, 'g+');
hold off;

blur_I = subIm;

cx = width/2;
cy = height/2;

for x = 1:width
    for y = 1:height
        X = [cx, cy; x, y];
        r = pdist(X,'euclidean');
        f = exp(-r/height);
        blur_I(y, x, 1) = f * subIm(y, x, 1); % invert red
        blur_I(y, x, 2) = f * subIm(y, x, 2); % invert green
        blur_I(y, x, 3) = f * subIm(y, x, 3); % invert blue
    end
end

blur_I = imadjust(blur_I, [.2 .3 0; .5 .8 1],[]);
figure, imshow(blur_I);

%% Task 1b: Colour Segmentation
R = blur_I(:,:,1); 
G = blur_I(:,:,2); 
B = blur_I(:,:,3);

[rows, cols, planes] = size(blur_I);

% ind = find(235 < R & R <= 255 &...
%            233 < G & G <= 255 &...
%            230 < B & B <= 235);

ind = find(R > 250 & G > 120 & B > 140);
             
starts = zeros(rows, cols);
starts(ind) = 1;

figure;
imshow(starts); 
hold on;

%% Task 1c: Morphological processing
K = imfill(starts, 'holes');
figure; imshow(K);

%% Task 1d: Region filtering
stats = regionprops('table',logical(K), 'Area', 'Solidity','Centroid','MajorAxisLength','MinorAxisLength');

T = max(stats.Area);

ind1 = ([stats.Area] >= T);
L = bwlabel(K);
result = ismember(L, find(ind1));
imshow(result);
hold on;

%% find new regions
stats_new = regionprops(logical(result), 'Area', 'Solidity','Centroid','MajorAxisLength','MinorAxisLength');
centers = stats_new.Centroid;
diameters = mean([stats_new.MajorAxisLength stats_new.MinorAxisLength],2);
radii = diameters/2;

%%
x1 = fix(stats_new.Centroid(1) - stats_new.MajorAxisLength/2);
x2 = fix(stats_new.MajorAxisLength);

y1 = fix(stats_new.Centroid(2) - stats_new.MinorAxisLength/2);
y2 = fix(stats_new.MinorAxisLength);

[y_r,x_r,z_r] = size(result);

y_test = y1 + y2;
x_text = x1 + x2;

if y_test >= y_r
   y2 = y_r - y1;
elseif y_test < 0
   y2 = stats_new.Centroid(2);
end

if x_text >= x_r
   x2 = x_r - x1;
elseif x_text < 0
   x2 = stats_new.Centroid(1);
end

if y1 < 0
    y1 = 1;
end

cordinates = [x1, y1, x2, y2];

hold on
viscircles(centers,radii);
hold off

BW1 = rgb2gray(subIm);
Idigits = insertObjectAnnotation(BW1, 'rectangle', cordinates, 'Area');
figure; 
imshow(Idigits);

%% New cordinates in the original Image
x1_last = x1 + xx1;
y1_last = y1 + yy1;
x2_last = x2;
y2_last = y2;

new_cordinates = [x1_last, y1_last, x2_last, y2_last];

%% Apply OCR
results = ocr(I, new_cordinates, 'CharacterSet', '0123456789');
results.Words()
% max(results.WordConfidences)

%%
I_last = insertObjectAnnotation(I, 'rectangle', new_cordinates, results.Words(1));
figure; 
imshow(I_last);

% %%
% numberRecognition(I)
% 
% %%
% function labelNum = numberRecognition(Image)
%     labelNum(Image);
%     label = '01';
% end