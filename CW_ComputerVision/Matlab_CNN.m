%% create the datastore
digitDatasetPath = fullfile('C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW','Faces_DataSource');
digitData = imageDatastore(digitDatasetPath,...
    'IncludeSubfolders',true,'LabelSource','foldernames');

%% display images
figure;
perm = randperm(644,25);
for i = 1:25
    subplot(5,5,i);
    imshow(digitData.Files{perm(i)});
end

%% get the count of the labels
labelCount = countEachLabel(digitData);

%% Display Montage of First Face
figure;
montage(digitData(20).ImageLocation);
title('Images of Single Face');

%% check the size of the first image
img = readimage(digitData,1);
% img = rgb2gray(img);
size_i = size(img);

%% specify traing and validation sets
minSetCount = min([labelCount.Count]);
% minSetCount = 256;

[training,test] = splitEachLabel( digitData , minSetCount ,'randomize');

%% define the Network Architecture
layers = [
    imageInputLayer([128 128])

    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,64,'Padding',1)
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(54)
    softmaxLayer
    classificationLayer];

%% specify the Training options
% options = trainingOptions('sgdm',...
%     'MaxEpochs',3, ...
%     'ValidationData',valDigitData,...
%     'ValidationFrequency',50,...
%     'Verbose',false,...
%     'Plots','training-progress');

options = trainingOptions('sgdm',...
    'MaxEpochs',1,...
    'MiniBatchSize', 64,...
    'Plots','training-progress');

%% Train network using Traing Data
CV_net_CNN = trainNetwork(training ,layers, options);
disp('net Train completed!');

%% Classify Validation Images and Compute Accuracy
predictedLabelsTest = classify(CV_net_CNN,test);
valLabelsTest = test.Labels;

accuracyTest = sum(predictedLabelsTest == valLabelsTest)/numel(valLabelsTest);
disp('Classification completed Test');

predictedLabelsTrain = classify(CV_net_CNN,training);
valLabelsTrain = training.Labels;

accuracyTrain = sum(predictedLabelsTrain == valLabelsTrain)/numel(valLabelsTrain);
disp('Classification completed Train');

%% Classify all the faces
face_path_classifier = ('C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW\Faces_Classified_Video');

cd('C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW\Faces_Group');

sub_file_faces_list = dir('*.jpg');
count = 0;

% for i = 1: 100
for i = 1: numel(sub_file_faces_list)
    face_path_num = fullfile(sub_file_faces_list(i).folder, [sub_file_faces_list(i).name]);
%     disp(face_path_num);
    X = imread(face_path_num);
%     X = rgb2gray(X);
    X = imresize(X, [128, 128]);
    imshow(X);
    
    Y_pred = predict(CV_net_CNN,X);
%     disp('Classification completed!!');

    [I,J] = max(Y_pred);
% 
    if I > 0.90
        count = count+1;
        disp([num2str(count), ' : ' , num2str(I)]);
        label_folder = char(labelCount{J,1});
        path_class = fullfile(face_path_classifier, label_folder, [sub_file_faces_list(i).name]);
        disp(path_class);
        imwrite(X, path_class);
    end
end
disp('Classification Completed!!');

%% Predict one image
% X = imread('C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW\Faces_Group\Face_group_7.jpg');
% X = imresize(X, [128, 128]);
% imshow(X);
% 
% %%
% Y_pred = predict(net,X);
% disp('Prediction completed!!');
% %%
% disp(num2str(Y_pred));
% [I,J] = max(Y_pred);
% label_folder = char(labelCount{J,1});
% disp(['Probability : ', num2str(I)]);
% disp(['Face Index  : ', num2str(J)]);
% disp(['Label Face  : ', label_folder]);
% 
% %% Save the Image in a Label Folder
% face_path_clasi = ('C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW\Faces_Classified_Video');
% path_class = fullfile(face_path_clasi, label_folder);
% disp(path_class);
% %imwrite(subIm, outputGroupFileName);
% 
%% save
cd('C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW\Models');
save CV_net_CNN;
disp('Save completed');

labelCount{20,1} = {'037'};

%% load
cd('C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW\Models');
load CV_net_CNN;
disp('Load completed');


%% Predict the faces in the FaceGroup Video
path = 'C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW\Group11of11';
cd(path);

v = VideoReader('IMG_0612.m4v');
while hasFrame(v)    
    video = readFrame(v);
    video = rgb2gray(video);
    imshow(video);
end

%% Predict all the faces in the FaceGroup Video
clc; close all;
path = 'C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW\Group11of11';
cd(path)

FaceDetector = vision.CascadeObjectDetector('FrontalFaceCART','MergeThreshold',10);

v = VideoReader('IMG_0627.mov');
while hasFrame(v)    
    groupImage = readFrame(v);
%     groupImage = imresize(groupImage, .5);
    I_last = groupImage;
    groupImage = rgb2gray(groupImage);

    BBOX = step(FaceDetector, groupImage);
%     disp(['detection completed, we found: ', num2str(size(BBOX(:,1)))]);

    for i = 1: size(BBOX(:,1)) 
        xbox = BBOX(i,:);
        subIm = groupImage(xbox(2):xbox(2)+xbox(4),xbox(1):xbox(1)+xbox(3),1:end);
        subIm = imresize(subIm, [128, 128]);
%         disp(['Face num: ', num2str(i)]);

        Y_pred = predict(CV_net_CNN,subIm);
%         disp('Classification completed!!');

        [I,J] = max(Y_pred);
        label_folder = char(labelCount{J,1});
        I_last = insertObjectAnnotation(I_last, 'rectangle',  BBOX(i,:), label_folder);

    %     figure; 
    %     imshow(I_last);
    end
%         figure; 
        imshow(I_last);
end

%% Predict all the faces in the FaceGroup Image
clc; close all;
source_folder = 'C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW\Group11of11';
cd(source_folder);

FaceDetector = vision.CascadeObjectDetector('FrontalFaceCART','MergeThreshold',10);

groupImage = imread('IMG_0627.jpg');
groupImage = rgb2gray(groupImage);
imshow(groupImage);

BBOX = step(FaceDetector, groupImage);
numFaces = BBOX(:,:);
disp(['detection completed, we found: ', num2str(size(BBOX(:,1)))]);

I_last = groupImage;

for i = 1: size(BBOX(:,1)) 
    xbox = BBOX(i,:);
    subIm = groupImage(xbox(2):xbox(2)+xbox(4),xbox(1):xbox(1)+xbox(3),1:end);
    subIm = imresize(subIm, [128, 128]);
%     disp(['Face num: ', num2str(i)]);
    
    Y_pred = predict(CV_net_CNN,subIm);
%     disp('Classification completed!!');

    [I,J] = max(Y_pred);
    label_folder = char(labelCount{J,1});
    
    I_last = insertObjectAnnotation(I_last, 'rectangle',  BBOX(i,:), label_folder);
    
%     figure; 
%     imshow(I_last);
end
%     figure; 
    imshow(I_last);
    
%% recognize in realtime
load models\CV_net_CNN
clear('cam');
output_folder = 'C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW\WebCam';
count_face = 0;
cam = webcam;
FaceDetector = vision.CascadeObjectDetector('FrontalFaceCART','MergeThreshold',25);

for index = 1:10000
    img = snapshot(cam);
    
        BBOX = step(FaceDetector, img);

    for i = 1: size(BBOX(:,1)) 
        xbox = BBOX(i,:);
        subIm = img(xbox(2):xbox(2)+xbox(4),xbox(1):xbox(1)+xbox(3),1:end);
        subIm = imresize(subIm, [128, 128]);
        subIm = rgb2gray(subIm);
        
        Y_pred = predict(CV_net_CNN, subIm);

        [I,J] = max(Y_pred);
        label_folder = char(labelCount{J,1});

        img = insertObjectAnnotation(img, 'rectangle',  BBOX(i,:), label_folder);
    end
    imshow(imresize(img,2));
end
clear('cam');