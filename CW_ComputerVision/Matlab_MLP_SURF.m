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
% minSetCount = 256;
imgSets = partition(faceDatabase, minSetCount, 'randomize');

[training,test] = partition(imgSets, .70, 'randomize');

%% Create the bag of Features
bag_MDL_SURF = bagOfFeatures(training,'VocabularySize', 128,'StrongestFeatures', 0.700);

%% Encode the Train and the Test set
featureVector_train = encode(bag_MDL_SURF, training);
featureVector_test = encode(bag_MDL_SURF, test);
%% Preprocess the train set
s1 = size(training,2);
s2 = training.Count;

% xTrainM = zeros(248 , size(training,2) * training(1).Count);

yTrainM = zeros(s1, (s1 * s2));
yTrainM_results = zeros(1, (s1 * s2));

featureCount = 1;
for i=1:size(training,2)
    for j = 1:training(i).Count
        img = read(imgSets(1), 3);
%         xTrainM(:,featureCount) = encode(bag, read(imgSets(i), j));
        yTrainM(i,featureCount) = 1;
%         yTrainM_results(featureCount) = str2num(training(i).Description);
        yTrainM_results(featureCount) = i;
        featureCount = featureCount + 1;
    end
end

disp('SURF Features MLP Completed!');

%% create and train and test Vectors for he network
featureVector_train_NET = featureVector_train';
featureVector_test_NET = featureVector_test';
%% Train the network
MLP_SURF_net = feedforwardnet([248, 128], 'trainscg');  % Scaled Conjugate Gradient Backpropagation
MLP_SURF_net.trainParam.epochs=5000;
MLP_SURF_net = configure(MLP_SURF_net, featureVector_train_NET, yTrainM);
MLP_SURF_net = train(MLP_SURF_net, featureVector_train_NET, yTrainM);
disp('Net Trained');

%% Preprocess the test set
s1_t = size(test,2);
s2_t = test.Count;

% xTestM = zeros(248 , (s1_t * s2_t));

yTestM = zeros(s1_t, (s1_t * s2_t));
yTestM_results = zeros(1, (s1_t * s2_t));

featureCount = 1;
for i=1:size(test,2)
    for j = 1:test(i).Count
%         xTestM(:,featureCount) = encode(bag, read(imgSets(i), j));
        yTestM(i,featureCount) = 1;
%         yTestM_results(featureCount) = str2num(training(i).Description);
        yTestM_results(featureCount) = i;
        featureCount = featureCount + 1;
    end
end

disp('Test set Completed!');

%% test the neural network on the train and test images
outPutsTrain = MLP_SURF_net(featureVector_train_NET); % xTrainM
outPutsTest = MLP_SURF_net(featureVector_test_NET); % xTestM
 
for i = 1 : size(outPutsTrain,2)
    [value outPutLabelsTrain(1,i)] = max(outPutsTrain(:,i));
end

for i = 1 : size(outPutsTest,2)
    [value outPutLabelsTest(1,i)] = max(outPutsTest(:,i));
end

disp('Prediction Completed');

%% Correct accuracy
% trainresults = [outPutLabelsTrain', yTrainM_results'];
% correct = 0;
% for i = 1: size(trainresults,1)
%     if trainresults(i,1) == trainresults(i,2)
%         correct = correct+ 1;
%     else
%         disp(i)
%     end
% end

%% calculate the accuracy
ACCTrain = sum(outPutLabelsTrain == yTrainM_results) / size(outPutsTrain,2)
ACCTest = sum(outPutLabelsTest == yTestM_results) / size(outPutsTest,2)

%% Map back to training set to find identity
person = 29;
queryImage = read(test(person),4);
queryFeatures = encode(bag_MDL_SURF, queryImage);

personLabel = MLP_SURF_net(queryFeatures');
probabilityIndex = max(personLabel(:));

integerIndex = find(personLabel(:,1) == probabilityIndex);
subplot(1,2,1); imshow(queryImage); title('Query Face');
subplot(1,2,2); imshow(read(training(integerIndex),1)); title('Matched Class');

%% save
cd('C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW\Models');
save MLP_SURF_net;
disp('Save completed!');

%% load
cd('C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW\Models');
load MLP_SURF_net;
disp('Load completed!');