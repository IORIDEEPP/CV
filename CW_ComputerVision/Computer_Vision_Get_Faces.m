%% Load Image Information from ATT Face Database Directory
cd('C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW');
faceDatabase = imageSet('DataSource','recursive');

%% Get the subfolders
%cd('C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW\DataSource');
files = dir('C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW\DataSource');
file_list = files(3:numel(files),:);  % just use the subfolders

% for l = 1: size(file_list(:,1))
%     new_path = strcat(file_list(l,:).folder, '\', subFolders(l,:).name);
%     disp(new_path);
%     cd(new_path);
% end
     
%% create face_datasource directory
output_folder = 'C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW\Faces_Classified_Video';

for i = 1 : size(file_list(:,1))
    out = file_list(i,:).name;
    out_save = output_folder;
    output_folder = strcat(output_folder, '\', out);
    disp(output_folder);
    
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end
    
    output_folder = out_save;
end
     
%% Display Montage of First Face
figure;
montage(faceDatabase(20).ImageLocation);
title('Images of Single Face');

%% Face detection in one image
output_folder = 'C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW\Faces_DataSource';

FaceDetector = vision.CascadeObjectDetector('FrontalFaceCART','MergeThreshold',15);

subIm = ones(128,128,3);
subIm(:,:,3) = 1;

for j = 29 : 29
% for j = 1 : size(faceDatabase,2)
    personToQuery = j;
    
    out = file_list(j,:).name;
    disp(out);
    out_save = output_folder;
    output_folder = strcat(output_folder, '\', out);
    disp(output_folder);
    
    for i = 1 : size((faceDatabase(personToQuery).ImageLocation),2)
        galleryImage = read(faceDatabase(personToQuery),i);
        galleryImage = rgb2gray(galleryImage);
                
        [x,y] = size(galleryImage);
        if x > 2048% && y > 5000
            galleryImage = imresize(galleryImage, 1/5);
        else
            galleryImage = imresize(galleryImage, 2/7);
        end
        
        if x < y
            galleryImage = imrotate(galleryImage,270);
        end
           imshow(galleryImage);

        BBOX = step(FaceDetector,galleryImage);
        numFaces = BBOX(:,:);
        
        if isempty(BBOX)
            disp(['no face detected in the Fist try: ',sprintf('%0.0f',(j)),'  ', sprintf('%0.0f',(i)) ]);
                galleryImage = imrotate(galleryImage,90);
                imshow(galleryImage);
                
                BBOX = step(FaceDetector,galleryImage);
                numFaces = BBOX(:,:);
%                 disp(numFaces);
                
                if isempty(BBOX)
                    disp(['no face detected in the Second try: ',sprintf('%0.0f',(j)),'  ', sprintf('%0.0f',(i)) ]);
                    subIm(:,:,3) = 1;
                    continue;
                else
                    xbox = BBOX(1,:);
                    subIm = galleryImage(xbox(2):xbox(2)+xbox(4),xbox(1):xbox(1)+xbox(3),1:end);
                    subIm = imresize(subIm, [128, 128]);
                    imshow(subIm);
                end
        else 
            xbox = BBOX(1,:);
            subIm = galleryImage(xbox(2):xbox(2)+xbox(4),xbox(1):xbox(1)+xbox(3),1:end);
            subIm = imresize(subIm, [128, 128]);
            imshow(subIm);
        end
        
%         face_number = strcat('Face_', out, '_');
%         outputFileName = fullfile(output_folder, [face_number num2str(i) '.jpg']);
%         imwrite(subIm, outputFileName);
       
        disp(output_folder);
        disp('image saved!');
        imshow(subIm);
        
    end
        output_folder = out_save;
end

%% get images by the the video
FaceDetector = vision.CascadeObjectDetector('FrontalFaceCART','MergeThreshold',15);

source_folder = 'C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW\DataSource';
output_folder = 'C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW\Faces_DataSource';

cd('C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW\DataSource');
files = dir('C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW\DataSource');
file_list = files(3:55,:);  % just use the subfolders

missing = [1 4 15 16 24 28 46];

%for o = 36 : 36
for o = 1 : numel(missing)
%for o = 50 : size(file_list(:,1))
%    sub_dir = strcat(source_folder, '\', file_list(o,1).name);
     sub_dir = strcat(source_folder, '\', file_list(missing(o),1).name);
     disp(sub_dir);
    
%    sub_dir_faces = strcat(output_folder, '\', file_list(o,1).name);
     sub_dir_faces = strcat(output_folder, '\', file_list(missing(o),1).name);
     disp(sub_dir_faces);
    
    cd(sub_dir);
    sub_file_List = dir('*.mov');
    for p = 1 : size(sub_file_List(:,1))
       disp(sub_file_List(p,:).name);
       v = VideoReader(sub_file_List(p,:).name);
       for i = 1: v.NumberOfFrames
            images = read(v,i);
            images_video = rgb2gray(images);
%            imshow(images_video);
            
            BBOX = step(FaceDetector, images_video);
            
            if isempty(BBOX)
               disp(['no face detected!!! : ',sprintf('%0.0f',(j)),'  ', sprintf('%0.0f',(i)) ]);
               subIm(:,:,3) = 1;
               continue;
            else
               xbox = BBOX(1,:);
               subIm = images_video(xbox(2):xbox(2)+xbox(4),xbox(1):xbox(1)+xbox(3),1:end);
               subIm = imresize(subIm, [355, 355]);
%               imshow(subIm);
            end
            
        face_video_number = strcat(sub_file_List(p,:).name, '_');
        outputVideoFileName = fullfile(sub_dir_faces, [face_video_number num2str(i) '.jpg']);
        imwrite(subIm, outputVideoFileName);
       
        disp(sub_dir_faces);
        disp('image saved!');
        imshow(subIm);
       end
    end
end

%% Get all the faces form the Group Image
FaceDetector = vision.CascadeObjectDetector('FrontalFaceCART','MergeThreshold',10);

source_folder = 'C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW\Group11of11';
output_folder_group = 'C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW\Faces_Group';

cd(source_folder);

groupImage = imread('IMG_0627.jpg');
groupImage = rgb2gray(groupImage);
imshow(groupImage);

BBOX = step(FaceDetector, groupImage);
numFaces = BBOX(:,:);
disp(['detection completed, we found: ', num2str(size(BBOX(:,1)))]);

for i = 1: size(BBOX(:,1)) 
    xbox = BBOX(i,:);
    subIm = groupImage(xbox(2):xbox(2)+xbox(4),xbox(1):xbox(1)+xbox(3),1:end);
    subIm = imresize(subIm, [355, 355]);
    disp(['Face num: ', num2str(i)]);
    hold off;
    imshow(subIm);
    
    outputGroupFileName = fullfile(output_folder_group, ['Face_group_' num2str(i) '.jpg']);
    imwrite(subIm, outputGroupFileName);
end

%% Read the video
videoReader = VideoReader('IMG_0612.m4v');
v = readFrame(videoReader);
for i = 1: videoReader.FrameRate
    images = readFrame(v);
    images_video = rgb2gray(images);
    [x,y] = size(images_video);
    if x < y
        images_video = imrotate(images_video,270);
    end
    imshow(images_video);
end
%%
v = VideoReader('IMG_0627.mov');
while hasFrame(v)
    video = readFrame(v);
    imshow(video);
end
whos video
disp('Completed video');

%%
imshow(video);