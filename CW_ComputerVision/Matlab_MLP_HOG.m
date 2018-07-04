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
% % diff = zeros(1,9);

%% Split Database into Training & Test Sets
minSetCount = min([faceDatabase.Count]);
imgSets = partition(faceDatabase, minSetCount, 'randomize');

[training,test] = partition(imgSets, .70, 'randomize');

%% Extract and display Histogram of Oriented Gradient Features for single face
person = 20;
[hogFeature, visualization]= extractHOGFeatures(read(training(person),1),'CellSize',[8 8],'BlockSize',[1 1]);

hogFeatureSize = size(hogFeature,2);

figure;
subplot(2,1,1);imshow(read(training(person),1));title('Input Face');
subplot(2,1,2);plot(visualization);title('HoG Feature');

%% Preprocess the train set
s1 = size(training,2);
s2 = training.Count;

xTrainM = zeros(hogFeatureSize , size(training,2) * training(1).Count);

yTrainM = zeros(s1, (s1 * s2));
yTrainM_results = zeros(1, (s1 * s2));

featureCount = 1;
for i=1:size(training,2)
    for j = 1:training(i).Count
        xTrainM(:,featureCount) = extractHOGFeatures(read(training(i),j),'CellSize',[8 8],'BlockSize',[1 1]);
        yTrainM(i,featureCount) = 1;
%         yTrainM_results(featureCount) = str2num(training(i).Description);
        yTrainM_results(featureCount) = i;
        featureCount = featureCount + 1;
    end
end

disp('Train set Completed!');

%% create and train the network
MLP_HOGF_net = feedforwardnet([128, 64], 'trainscg');  % Scaled Conjugate Gradient Backpropagation
MLP_HOGF_net.trainParam.epochs=5000;
MLP_HOGF_net = configure(MLP_HOGF_net, xTrainM, yTrainM);
MLP_HOGF_net = train(MLP_HOGF_net, xTrainM, yTrainM);
disp('Net Trained');

%% Preprocess the test set
s1_t = size(test,2);
s2_t = test.Count;

xTestM = zeros(hogFeatureSize , (s1_t * s2_t));

yTestM = zeros(s1_t, (s1_t * s2_t));
yTestM_results = zeros(1, (s1_t * s2_t));

featureCount = 1;
for i=1:size(test,2)
    for j = 1:test(i).Count
        xTestM(:,featureCount) = extractHOGFeatures(read(test(i),j),'CellSize',[8 8],'BlockSize',[1 1]);
        yTestM(i,featureCount) = 1;
%         yTestM_results(featureCount) = str2num(training(i).Description);
        yTestM_results(featureCount) = i;
        featureCount = featureCount + 1;
    end
end
 
disp('Test set Completed!');

%% test the neural network on the train and test images
outPutsTrain = MLP_HOGF_net(xTrainM);
outPutsTest = MLP_HOGF_net(xTestM);
 
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

%% Map back to training set to find identity
person = 20;
queryImage = read(test(person),1);
queryFeatures = extractHOGFeatures(queryImage,'CellSize',[8 8],'BlockSize',[1 1]);

personLabel = MLP_HOGF_net(queryFeatures');

probabilityIndex = max(personLabel(:)')
integerIndex = find(personLabel(:,1) == probabilityIndex);

subplot(1,2,1); imshow(queryImage); title('Query Face');
subplot(1,2,2); imshow(read(training(integerIndex),1)); title('Matched Class');

%% save
cd('C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW\Models');
save MLP_HOGF_net;
disp('Save completed');

%% load
cd('C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW\Models');
load MLP_HOGF_net;
disp('Load completed');