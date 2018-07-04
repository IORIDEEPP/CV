%% Recognize Faces on Image and Video
clear, clc, close all;
path = pwd; cd(path);

I = imread('Group11of11\IMG_0642.JPG'); % read the image or video

P = RecogniseFace(I, 'SURF', 'SVM'); % call the function

disp('Execution Completed!!'); %execution completed

%% Recognize Face in Real Time with CNN
clear, clc, close all;
face_real_time(); % call the function for face real time detector

%% Recognize Number in Images
clear, clc, close all;
list = dir('OCR\*.JPG');  %get all the images from the directory

for i = 1: size(list, 1)
    imagePath = fullfile(list(i).folder ,[list(i).name]);  % create the path for the image
    I = imread(imagePath);  % read the image
    [label, I_last] = detectNum(I);  % apply the function to detect number
    disp([list(i).name, label]);   % display the number in console
    figure; imshow(imresize((I_last),2/5));   % show the original image labeled
end
disp('Completed Image Number Recognition');

%% Recognize Number in videos
clear, clc, close all;
v = VideoReader('OCR\IMG_0882.mov');  % read the video 

labelnumber = zeros(int16(v.FrameRate),1);  % create a vector to find the number in all the frames
index = 1;

while hasFrame(v)    
    video = readFrame(v);       % read frame by frame
    [label, I_last] = detectNum(video);  % call the function
    labelnumber(index) =  str2double(string(label));  % save the number for this frame
    index = index + 1;  
%     disp(label);
    imshow(imresize(I_last, 5/5)); % show the original frame labeled
end
M = mode(labelnumber);  % get the number that appers more times
disp(['This video is for the person num: ', int2str(M)]);  % show the number in console
disp('Completed Video Number Recognition');  % execution completed


%% Define the Function for all the Classifiers
function P = RecogniseFace(I, featureType, classifierName)

    I = imresize(I, 2/3); % resize the images
    FaceDetector = vision.CascadeObjectDetector('FrontalFaceCART','MergeThreshold',10); % create the cascade object
    groupImage = rgb2gray(I); % convert to gray scale
    BBOX = step(FaceDetector, groupImage);  % apply the face detector
    
    if size(BBOX(:,1),1) > 0
            Matrix_Result = zeros(size(BBOX(:,1),1), 4);
            if isequal(classifierName, 'CNN')  % execute for CNN
            disp('CNN...');  
            load models/CV_net_CNN;  % load the model for CNN

            for i = 1: size(BBOX(:,1))
                xbox = BBOX(i,:);
                subIm = groupImage(xbox(2):xbox(2)+xbox(4),xbox(1):xbox(1)+xbox(3),1:end);  % create a sub image
                subIm = imresize(subIm, [128, 128]); % resize to 128 x 128

                Y_pred = predict(CV_net_CNN,subIm); % predict with the model

                [Ir,J] = max(Y_pred);
                label_folder = char(labelCount{J,1}); % get the label from the list
               
                [emotion_lbl, emotion_num] = emotion_detection(subIm);
                
                Matrix_Result(i,:) = [str2double(label_folder), BBOX(i,1), BBOX(i,2), emotion_num];
                I = insertObjectAnnotation(I, 'rectangle',  BBOX(i,:), [label_folder,' ',emotion_lbl]); % insert the label in the image
            end
                figure; 
                imshow(I);

        elseif isequal(featureType, 'HOG') && isequal(classifierName, 'MLP')
            disp('MLP_HOG'); 
            load models/MLP_HOGF_net;

            for i = 1: size(BBOX(:,1))
                xbox = BBOX(i,:);
                subIm = groupImage(xbox(2):xbox(2)+xbox(4),xbox(1):xbox(1)+xbox(3),1:end); % create a sub image
                subIm = imresize(subIm, [128, 128]); % resize to 128 x 128

                queryFeatures = extractHOGFeatures(subIm,'CellSize',[8 8],'BlockSize',[1 1]); % extract the features from the image
                personLabel = MLP_HOGF_net(queryFeatures'); % predict with the model
                probabilityIndex = max(personLabel(:)');
                integerIndex = find(personLabel(:,1) == probabilityIndex);

                label_folder = char(labelCount{integerIndex,1}); % get the label from the list

                [emotion_lbl, emotion_num] = emotion_detection(subIm);  % use the label to know the face number
                
                Matrix_Result(i,:) = [str2double(label_folder), BBOX(i,1), BBOX(i,2), emotion_num];
                I = insertObjectAnnotation(I, 'rectangle',  BBOX(i,:), [label_folder,' ',emotion_lbl]); % insert the label in the image
            end
                figure; 
                imshow(I);

        elseif isequal(featureType, 'SURF') && isequal(classifierName, 'MLP')
            disp('MLP_SURF');
            load models/MLP_SURF_net; % load the model for MLP SURF

            for i = 1: size(BBOX(:,1))
                xbox = BBOX(i,:);
                subIm = groupImage(xbox(2):xbox(2)+xbox(4),xbox(1):xbox(1)+xbox(3),1:end); % create a sub image
                subIm = imresize(subIm, [128, 128]); % resize to 128 x 128

                queryFeatures = encode(bag_MDL_SURF, subIm); % extract the features from the image with the bad of features

                personLabel = MLP_SURF_net(queryFeatures'); % predict with the model
                probabilityIndex = max(personLabel(:));

                integerIndex = find(personLabel(:,1) == probabilityIndex);
                label_folder = char(labelCount{integerIndex,1}); % get the label from the list
                
                [emotion_lbl, emotion_num] = emotion_detection(subIm); % use the label to know the face number
                Matrix_Result(i,:) = [str2double(label_folder), BBOX(i,1), BBOX(i,2), emotion_num];

                I = insertObjectAnnotation(I, 'rectangle',  BBOX(i,:), [label_folder, ' ', emotion_lbl]); % insert the label in the image
            end
                figure; 
                imshow(I);        

        elseif isequal(featureType, 'HOG') && isequal(classifierName, 'SVM')
            disp('SVM_HOG');        
            load models/SVM_Classifier_HOGF; % load the model for SVM HOG

            for i = 1: size(BBOX(:,1))
                xbox = BBOX(i,:);
                subIm = groupImage(xbox(2):xbox(2)+xbox(4),xbox(1):xbox(1)+xbox(3),1:end); % create a sub image
                subIm = imresize(subIm, [128, 128]); % resize to 128 x 128

                queryFeatures = extractHOGFeatures(subIm, 'CellSize', [8 8], 'BlockSize', [1 1]); % extract the features from the image
                predictFaceSVM = predict(SVM_Classifier_HOGF , queryFeatures); % predict with the model
                booleanIndex = strcmp(predictFaceSVM, personIndex);
                integerIndex = find(booleanIndex);
                label = char(personIndex(integerIndex)); % get the label from the list
               
                [emotion_lbl, emotion_num] = emotion_detection(subIm); % use the label to know the face number
                
                Matrix_Result(i,:) = [str2double(label), BBOX(i,1), BBOX(i,2), emotion_num ];
                I = insertObjectAnnotation(I, 'rectangle',  BBOX(i,:), [label,' ',emotion_lbl]); % insert the label in the image
            end
                figure; 
                imshow(I);

        elseif isequal(featureType, 'SURF') && isequal(classifierName, 'SVM')
            disp('SVM_SURF');
            load models/SVM_Classifier_SURF; % load the model for CSVM SURF

            for i = 1: size(BBOX(:,1))
                xbox = BBOX(i,:);
                subIm = groupImage(xbox(2):xbox(2)+xbox(4),xbox(1):xbox(1)+xbox(3),1:end); % create a sub image
                subIm = imresize(subIm, [128, 128]); % resize to 128 x 128

                [Idx,J] = predict(SVM_Classifier_SURF, subIm); % predict with the model

                [emotion_lbl, emotion_num] = emotion_detection(subIm); % use the label to know the face number
                               
                Matrix_Result(i,:) = [str2double(char(labelCount{Idx,1})), BBOX(i,1), BBOX(i,2), emotion_num];

                I = insertObjectAnnotation(I, 'rectangle',  BBOX(i,:), [char(labelCount{Idx,1}), ' ', emotion_lbl]); % insert the label in the image
            end
                figure; 
                imshow(I);
        else
            disp('Classifier does no exist'); % not select correct parameters
        end
        P = Matrix_Result;
    else
        disp('No face detected'); % no face detected
        P = 0;
    end
end

function [emotion_label, number_label] = emotion_detection(FaceImage)
    load models/CV_net_CNN_face_emotion % load the CNN for face emotion
    
    Y_pred = predict(CV_net_CNN_face_emotion, FaceImage); % predict the emtion

        [I,J] = max(Y_pred);  % the label of the emotion
        emotion_label = char(label_emotion{J,1});
        number_label = J-1; % return the number of the emotion
end

function face_real_time()
    load models\CV_net_CNN  % load the model for CNN
    clear('cam');  % close if the camara is in use
    cam = webcam; % get the camara
    FaceDetector = vision.CascadeObjectDetector('FrontalFaceCART','MergeThreshold',25);  % create the cascade object
    
%     v = VideoWriter('newfile.avi','Uncompressed AVI');

    for index = 1:10000 
        img = snapshot(cam); % get each fram of the video as a image
        BBOX = step(FaceDetector, img);

        for i = 1: size(BBOX(:,1)) 
            xbox = BBOX(i,:);
            subIm = img(xbox(2):xbox(2)+xbox(4),xbox(1):xbox(1)+xbox(3),1:end);
            subIm = imresize(subIm, [128, 128]); % resie to [128, 128] 
            subIm = rgb2gray(subIm); % convert to gray scale

            Y_pred = predict(CV_net_CNN, subIm); % predict the face

            [I,J] = max(Y_pred);  ; % get the label from the list
            label_folder = char(labelCount{J,1});  % use the label to know the face number

            img = insertObjectAnnotation(img, 'rectangle',  BBOX(i,:), label_folder); % insert the label in the image
        end
        imshow(imresize(img,2));        
%         writeVideo(v,img);
    end
    clear('cam');
end

function [labelNumber, I_last] = detectNum(I)
    %% Resize the image   
    [x,y,z] = size(I);
    if x > 2500 && y > 2500
        I = imresize(I, 2/5); % resize to 2/5
    else
        I = imresize(I, 4/5); % resize to 4/5
    end    
%     figure; imshow(I);

    %% Detect the Face
    FaceDetector = vision.CascadeObjectDetector('FrontalFaceCART','MergeThreshold',12);
    BBOX = step(FaceDetector,I);
    [y,x,z] = size(I);

    %% Rotate the picture if the face is not detected
    if isempty(BBOX)
        if x > y
            I = imrotate(I,270);
            BBOX = step(FaceDetector,I);
            [y,x,z] = size(I);
            if isempty(BBOX)
            else 
                return;
            end
        else
            disp('exit function');
            labelNumber = '999';
            I_last = I;
            return;
        end
    else
    end
   
    xx1 = BBOX(1) - BBOX(3)* .5; % 0.7
    yy1 = BBOX(2) + BBOX(4)* 1.4; % 1.9
    xx2 = BBOX(3) * 2; % 2.5
    yy2 = BBOX(4) * 2.5; % 2.1 , 

    Idigits1 = insertObjectAnnotation(I, 'rectangle', [xx1, yy1 , xx2, yy2], 'ZOOM');

    sub_yy2 = yy1+yy2;
    sub_xx2 = xx1+xx2;

    if sub_yy2 > y
       sub_yy2 = y - (y/6);
    end
    
    if sub_xx2 > x
        sub_xx2 = x;
    end
    
    subIm = I(fix(yy1):fix(sub_yy2), fix(xx1):fix(sub_xx2), 1:end);

    %% highlight the pixel from the center

    [height, width, channels] = size(subIm);
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

    %% Colour Segmentation to get just the white area
    R = blur_I(:,:,1); 
    G = blur_I(:,:,2); 
    B = blur_I(:,:,3);

    [rows, cols, planes] = size(blur_I);
    
    ind = find(R > 250 & G > 150 & B > 145); % 155

    starts = zeros(rows, cols);
    starts(ind) = 1;

    %% Fill the Holes to minimize the black area
    K = imfill(starts, 'holes');

    %% Find all the white region 
    stats = regionprops('table',logical(K), 'Area', 'Solidity','Centroid','MajorAxisLength','MinorAxisLength');
    T = max(stats.Area);

    ind1 = ([stats.Area] >= T);
    L = bwlabel(K);
    result = ismember(L, find(ind1));

    %% Find just one region new regions
    stats_new = regionprops(logical(result), 'Area', 'Solidity','Centroid','MajorAxisLength','MinorAxisLength');
    
    x1 = fix(stats_new.Centroid(1) - stats_new.MajorAxisLength/2.8);
    x2 = fix(stats_new.MajorAxisLength * .7);

    y1 = fix(stats_new.Centroid(2) - stats_new.MinorAxisLength/2.8); %3.6
    y2 = fix(stats_new.MinorAxisLength * .8);

    [y_r,x_r,z_r] = size(result);

    y_test = y1 + y2;
    x_text = x1 + x2;

    if y_test >= y_r
       y2 = y_r - y1;
    elseif y_test <= 0
       y2 = stats_new.Centroid(2);
    end

    if x_text >= x_r
       x2 = x_r - x1;
    elseif x_text < 0
       x2 = stats_new.Centroid(1);
    end

    if y1 <= 0
        y1 = 1;
    end
    
    %% Create new cordinates from the square
    x1_last = x1 + xx1;
    y1_last = y1 + yy1;
    x2_last = x2;
    y2_last = y2;

    new_cordinates = [x1_last, y1_last, x2_last, y2_last];

    %% Apply OCR fucntion in the new cordinates
    results = ocr(I, new_cordinates, 'CharacterSet', '0123456789','TextLayout', 'Block');
    
    if isempty(results.Words)
        disp('NO Words found');
        labelNumber = '999';
    else
        labelNumber = results.Words(1);
    end
    
    I_last = insertObjectAnnotation(I, 'rectangle', new_cordinates, labelNumber);

end