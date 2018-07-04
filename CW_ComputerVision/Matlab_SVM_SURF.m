%% Setup the environment
path = 'C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW';
cd(path);
faceDatabase = imageSet('Faces_DataSource','recursive');

%% Partition the data set regarding the smallest amount of images
minSetCount = min([faceDatabase.Count]); 
faceDatabase = partition(faceDatabase, minSetCount, 'randomize'); % Notice that each set now has exactly the same number of images.

[train_set, test_set] = partition(faceDatabase, .70, 'randomize');

%% Getting the fetures to train the Classifier
bag_SVM_SURF = bagOfFeatures(train_set, 'VocabularySize', 128, 'StrongestFeatures', 0.70);

%% Plot the histogram of visual word occurrences
img = read(faceDatabase(1), 3);
featureVector = encode(bag_SVM_SURF, img);
figure; bar(featureVector)

title('Visual word occurrences')
xlabel('Visual word index')
ylabel('Frequency feaof occurrence')

%% Categorice the trainSet
SVM_Classifier_SURF = trainImageCategoryClassifier(train_set, bag_SVM_SURF);
disp('Completed');
%% evaluate the trainSet
confusionmat

confMatrixTrain = evaluate(SVM_Classifier_SURF, train_set);
disp('Completed');
% 
%% evaluate the testset
confMatrixTest = evaluate(SVM_Classifier_SURF, test_set);
% 
% %% Compute average accuracy
% mean(diag(confMatrix))

%% Classify all the faces
labelCount = countEachLabel(faceDatabase);
face_path_classifier = ('C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW\Faces_Classified_Video');
%%
cd('C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW\Faces_Group');
sub_file_faces_list = dir('*.jpg');
count = 0;

% for i = 1: 3
for i = 1: numel(sub_file_faces_list)
    face_path_num = fullfile(sub_file_faces_list(i).folder, [sub_file_faces_list(i).name]);
%     disp(face_path_num);
    X = imread(face_path_num);
    X = imresize(X, [128, 128]);
    imshow(X);
    
    [I,J] = predict(SVM_Classifier_SURF, X);
%     disp(num2str(I));
%     disp(num2str(max(J)));

    count = count+1;
    disp([num2str(count), ' : ' , char(labelCount{I,1})]);
    label_folder = char(labelCount{I,1});
    path_class = fullfile(face_path_classifier, label_folder, [sub_file_faces_list(i).name]);
% %         disp(path_class);
    imwrite(X, path_class);
    
%     label_folder = char(labelCount{J,1});
%     path_class = fullfile(face_path_classifier, label_folder, [sub_file_faces_list(i).name]);
%     disp(path_class);
%     imwrite(X, path_class);
end
disp('Classification Completed!!');

%% save
cd('C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW\Models');
save SVM_Classifier_SURF;

%% load
cd('C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW\Models');
load SVM_Classifier_SURF;