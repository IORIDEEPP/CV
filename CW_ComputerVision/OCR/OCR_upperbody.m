close all;
list = dir('*.JPG');

%%

% for i = 51: numel(list)
for i = 30:30
    I = imread(list(i).name);
    I = imresize(I, 0.3);
    imshow(I);
    numberRecognition(I);
end

function numberRecognition(I)

    FaceDetector = vision.CascadeObjectDetector('FrontalFaceCART','MergeThreshold',15);
    BBOX = step(FaceDetector,I);

    xx1 = BBOX(1);
    yy1 = BBOX(2);
    xx2 = BBOX(3);
    yy2 = BBOX(4); 

    Idigits1 = insertObjectAnnotation(I, 'rectangle', [xx1, yy1 , xx2, yy2], 'Face');
    figure; 
    imshow(Idigits1);
    
    sub_yy2 = yy1+yy2;
    sub_xx2 = xx1+xx2;
    
    subIm = I(fix(yy1):fix(sub_yy2), fix(xx1):fix(sub_xx2), 1:end);
    figure;
    imshow(subIm);
    
%     I = imread('cameraman.tif');
    subIm = rgb2gray(subIm);
    [featureVector,hogVisualization] = extractHOGFeatures(subIm);
    imshow(subIm); hold on;
    plot(hogVisualization);
    
    % SURF
    subIm = imresize(subIm, [128,128]);
    points = detectSURFFeatures(subIm, 'FeatureSize', 128)
    imshow(subIm); hold on;
end