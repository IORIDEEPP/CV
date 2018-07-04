close; clc; clear;
path = 'C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW';
cd(path);
faceDatabase = imageSet('Faces_DataSource','recursive');

%% Display Query Image and Database Side-Side
personToQuery = 1;
galleryImage = read(faceDatabase(personToQuery),1);
figure;
for i=1:size(faceDatabase,2)
    imageList(i) = faceDatabase(i).ImageLocation(5);
end
subplot(1,2,1);imshow(galleryImage);
subplot(1,2,2);montage(imageList);
% diff = zeros(1,9);

%% Split Database into Training & Test Sets
% minSetCount = min([faceDatabase.Count]);
minSetCount = 100;
imgSets = partition(faceDatabase, minSetCount, 'randomize');

[training,test] = partition(imgSets, .80, 'randomize');

%% Extract and display Histogram of Oriented Gradient Features for single face
person = 20;
[hogFeature, visualization]= extractHOGFeatures(read(training(person),1));

figure;
subplot(2,1,1);imshow(read(training(person),1));title('Input Face');
subplot(2,1,2);plot(visualization);title('HoG Feature');

%%
% [imageWidth, imageHeight] = size(read(training(1),1));
% 
% inputSize = imageWidth * imageHeight;
% 
% s1 = size(training,2);
% s2 = training.Count;
% 
% xTrainM = zeros(inputSize , (s1 * s2));
% yTrainM = zeros(1, (s1 * s2));
% 
% yTrainM_t = zeros(s1, (s1 * s2));
% 
% %%
% Count = 1;
% for i=1:size(training,2)
%     for j = 1:training(i).Count
%         im = read(training(i),j);
%         xTrainM(:,Count) = im(:);
%         yTrainM(1, Count) = str2num(training(i).Description);
%         yTrainM_t(i,Count) = 1;
%         Count = Count + 1;
%     end
% end
% 
% disp('Train set completed');
% 
% %% create and train the network
% net = feedforwardnet(100, 'trainscg');  % Scaled Conjugate Gradient Backpropagation
% net = configure(net, xTrainM, yTrainM_t);
% net = train(net, xTrainM, yTrainM_t);
% disp('Net Trained');
% 
% %% preprocess the test set
% s1_t = size(test,2);
% s2_t = test.Count;
% 
% xTestM = zeros(inputSize , (s1_t * s2_t));
% 
% Count = 1;
% for i=1:size(test,2)
%     for j = 1:test(i).Count
%         im = read(test(i),j);
%         xTestM(:,Count) = im(:);
%         yTestM(1, Count) = str2num(test(i).Description);
%         Count = Count + 1;
%     end
% end
% 
% disp('Test set completed');
% 
% %% test the neural network on the train and test images
% outPutsTrain = net(xTrainM);
% outPutsTest = net(xTestM);
% 
% for i = 1 : size(outPutsTrain,2)
%     [value outPutLabelsTrain(1,i)] = max(outPutsTrain(:,i));
% end
% 
% for i = 1 : size(outPutsTest,2)
%     [value outPutLabelsTest(1,i)] = max(outPutsTest(:,i));
% end
% 
% disp('Prediction Completed');
% 
% %% calculate the accuracy
% ACCTrain = sum(outPutLabelsTrain == yTrainM) / size(outPutsTrain,2)
% ACCTest = sum(outPutLabelsTest == yTestM) / size(outPutsTest,2)

%% Extract HOG Features for training set
imageWidth = 128;
imageHeight = 128;
inputSize = imageWidth*imageHeight;

% trainingFeatures = zeros(inputSize, 8100);
trainingFeatures = zeros(size(training,2) * training(1).Count, 8100);

featureCount = 1;
for i=1:size(training,2)
    for j = 1:training(i).Count
        trainingFeatures(featureCount,:) = extractHOGFeatures(read(training(i),j),strongest);
        trainingLabel{featureCount} = training(i).Description;
        featureCount = featureCount + 1;
    end
    personIndex{i} = training(i).Description;
end

trainingFeatures_new = trainingFeatures';

disp('Completed!');

%% Create 40 class classifier using fitcecoc
faceClassifier = fitcecoc(trainingFeatures,trainingLabel);
disp('Classifier Completed!');

%% Map back to training set to find identity
person = 54;
queryImage = read(test(person),1);
queryFeatures = extractHOGFeatures(queryImage);
personLabel = predict(faceClassifier,queryFeatures);
 
booleanIndex = strcmp(personLabel, personIndex);
integerIndex = find(booleanIndex);
subplot(1,2,1);imshow(queryImage);title('Query Face');
subplot(1,2,2);imshow(read(training(integerIndex),1));title('Matched Class');