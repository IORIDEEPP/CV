%% Detect SURF features
% I = imread('gantrycrane.png');
I = imread('Face_067_f_4_r_1_b_3.jpg');
imshow(I);
% I = rgb2gray(I);

%%
I = imresize(I, [128,128]);
points = detectSURFFeatures(I)
% points = detectSURFFeatures(I, 'FeatureSize', 128)
% plot(selectStrongest(points, 30));

imshow(I); hold on;
plot(points.selectStrongest(180));
    
%% Detect HOG features
I = imread('Face_168_1.jpg');
[featureVector,hogVisualization] = extractHOGFeatures(I);
imshow(I); hold on;
plot(hogVisualization);

%% Detect Fast features
I2 = imread('gantrycrane.png');
corners = detectFASTFeatures(rgb2gray(I2));
strongest = selectStrongest(corners,3);
[hog2, validPoints,ptVis] = extractHOGFeatures(I2,strongest);

%% SURF
I = imread('cameraman.tif');
points = detectSURFFeatures(I, 'MetricThresh', 200,...
    'NumOctaves', 3,...
    'NumScaleLevels', 4,... 
    'SURFSize', 64 );
% [features, valid_points] = extractFeatures(I, points);
% vfeature(:) = feature;

