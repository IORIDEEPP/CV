close; clc; clear;
path = 'C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW';
cd(path);
faceDatabase = imageSet('Faces_DataSource','recursive');

%% Display Query Image and Database Side-Side
% personToQuery = 1;
% galleryImage = read(faceDatabase(personToQuery),1);
% figure;
% for i=1:size(faceDatabase,2)
%     imageList(i) = faceDatabase(i).ImageLocation(5);
% end
% subplot(1,2,1);imshow(galleryImage);
% subplot(1,2,2);montage(imageList);
% diff = zeros(1,9);

%% Split Database into Training & Test Sets
% minSetCount = min([faceDatabase.Count]);
minSetCount = 100;
imgSets = partition(faceDatabase, minSetCount, 'randomize');

[training, test] = partition(imgSets, .80, 'randomize');

%%
[imageWidth, imageHeight] = size(read(faceDatabase(1),1));

inputSize = imageWidth * imageHeight;

s1 = size(training,2);
s2 = training.Count;

xTrainM = zeros(inputSize , (s1 * s2));

yTrainM_results = zeros(1, (s1 * s2));
yTrainM_t = zeros(s1, (s1 * s2));

Count = 1;
for i=1:size(training,2)
    for j = 1:training(i).Count
        im = read(training(i),j);
        xTrainM(:,Count) = im(:);
        yTrainM_t(i,Count) = 1;
        yTrainM_results(1, Count) = str2num(training(i).Description);
        Count = Count + 1;
    end
end

disp('Train set completed');

%% create and train the network
net = feedforwardnet([128, 64], 'trainscg');  % Scaled Conjugate Gradient Backpropagation
net = configure(net, xTrainM, yTrainM_t);
net = train(net, xTrainM, yTrainM_t);

disp('Net Trained');

%% preprocess the test set
s1_t = size(test,2);
s2_t = test.Count;

xTestM = zeros(inputSize , (s1_t * s2_t));

yTrainM_results = zeros(s1_t, (s1_t * s2_t));

Count = 1;
for i=1:size(test,2)
    for j = 1:test(i).Count
        im = read(test(i),j);
        xTestM(:,Count) = im(:);
        yTrainM_results(1, Count) = str2num(test(i).Description);
        Count = Count + 1;
    end
end

disp('Test set completed');

%% test the neural network on the train and test images
outPutsTrain = net(xTrainM);
outPutsTest = net(xTestM);

for i = 1 : size(outPutsTrain,2)
    [value outPutLabelsTrain(1,i)] = max(outPutsTrain(:,i));
end

for i = 1 : size(outPutsTest,2)
    [value outPutLabelsTest(1,i)] = max(outPutsTest(:,i));
end

disp('Prediction Completed');

%% calculate the accuracy
ACCTrain = sum(outPutLabelsTrain == yTrainM_results) / size(outPutsTrain,2)
ACCTest = sum(outPutLabelsTest == yTestM_results) / size(outPutsTest,2)

disp('Calculate accuracy');

%% Map back to training set to find identity
person = 20;
queryImage = read(test(person),1);
queryFeatures = extractHOGFeatures(queryImage);

personLabel = net(queryFeatures');

probabilityIndex = max(personLabel(:)');
integerIndex = find(personLabel(:,1) == probabilityIndex);

subplot(1,2,1); imshow(queryImage); title('Query Face');
subplot(1,2,2); imshow(read(training(integerIndex),1)); title('Matched Class');

%% save
cd('C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW');
CV_face_MPL = net;
save CV_face_MLP_2503;

%% load
cd('C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW');
load CV_face_MLP_2503;