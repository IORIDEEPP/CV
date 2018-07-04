%% Setup the environment
path = 'C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW';
cd(path);
faceDatabase = imageSet('Faces_DataSource','recursive');

%% Display Query Image and Database Side-Side
perm = randperm(644,42);

% for i=1:size(faceDatabase,2)
for i=1:size(perm,2)
    imageList_person(i) = faceDatabase(7).ImageLocation(perm(i));
end

% subplot(1,2,1);imshow(galleryImage);
% subplot(1,2,2);montage(imageList);
montage(imageList_person);
% diff = zeros(1,9);

%% Partition the data set regarding the smallest amount of images
minSetCount = min([faceDatabase.Count]);
imgSets = partition(faceDatabase, minSetCount, 'randomize');

[train_set, test_set] = partition(imgSets, .70, 'randomize');

%% Extract and display Histogram of Oriented Gradient Features for single face
person = 20;
[hogFeature, visualization]= extractHOGFeatures(read(train_set(person),1),'CellSize',[8 8],'BlockSize',[1 1]);

hogFeatureSize = size(hogFeature,2);

figure;
subplot(2,1,1);imshow(read(train_set(person),1));title('Input Face');
subplot(2,1,2);plot(visualization);title('HoG Feature');

%% Extract HOG Features for training set
imageWidth = 128;
imageHeight = 128;
inputSize = imageWidth*imageHeight;

% trainingFeatures = zeros(inputSize, 8100);
trainingFeatures_HOGF = zeros(size(train_set,2) * train_set(1).Count, hogFeatureSize);

featureCount = 1;
for i=1:size(train_set,2)
    for j = 1:train_set(i).Count
        trainingFeatures_HOGF(featureCount,:) = extractHOGFeatures(read(train_set(i),j),'CellSize',[8 8],'BlockSize',[1 1]);
        trainingLabel{featureCount} = train_set(i).Description;
        featureCount = featureCount + 1;
    end
    personIndex{i} = train_set(i).Description;
end

disp('Completed!');

%% Create All the class classifier using fitcecoc
SVM_Classifier_HOGF = fitcecoc(trainingFeatures_HOGF, trainingLabel);
disp('Classifier Completed!');

%% Map back to training set to find identity
person = 52;
queryImage = read(test_set(person),1);
queryFeatures = extractHOGFeatures(queryImage,'CellSize',[8 8],'BlockSize',[1 1]);
predictFaceSVM = predict(SVM_Classifier_HOGF , queryFeatures);
 
booleanIndex = strcmp(predictFaceSVM, personIndex);
integerIndex = find(booleanIndex);
subplot(1,2,1);imshow(queryImage);title('Query Face');
subplot(1,2,2);imshow(read(train_set(integerIndex),1));title('Matched Class');

%% recognize in realtime
% cam = webcam
FaceDetector = vision.CascadeObjectDetector('FrontalFaceCART','MergeThreshold',10);

for index = 1:10000
    img = snapshot(cam);
    imshow(img);
    
        BBOX = step(FaceDetector, img);

    for i = 1: size(BBOX(:,1)) 
        xbox = BBOX(i,:);
        subIm = img(xbox(2):xbox(2)+xbox(4),xbox(1):xbox(1)+xbox(3),1:end);
        subIm = imresize(subIm, [128, 128]);

        queryFeatures = extractHOGFeatures(subIm,'CellSize',[8 8],'BlockSize',[1 1]);
        predictFaceSVM = predict(SVM_Classifier_HOGF , queryFeatures)

%         [I,J] = max(predictFaceSVM);        

        I_last = insertObjectAnnotation(img, 'rectangle',  BBOX(i,:), 'test');

        imshow(I_last);
    end    
end

%% save
cd('C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW\Models');
save SVM_Classifier_HOGF
disp('Completed Save');

%% load
cd('C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW\Models');
load SVM_Classifier_HOGF
disp('Completed');