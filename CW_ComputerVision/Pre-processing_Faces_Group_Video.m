%% Get all the faces form the Group Image - Video
FaceDetector = vision.CascadeObjectDetector('FrontalFaceCART','MergeThreshold',25);
source_folder = 'C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW\Group11of11';
output_folder_group = 'C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW\Faces_Group';
cd(source_folder);

sub_file_List = dir('*.m4v');
for k = 1: numel(sub_file_List)
    v = VideoReader(sub_file_List(k,:).name);
    for j = 1: v.NumberOfFrames
        images = read(v,j);
        images = rgb2gray(images);
        imshow(images);

        BBOX = step(FaceDetector, images);
%         disp(['detection completed, we found: ', num2str(size(BBOX(:,1)))]);

        for i = 1: size(BBOX(:,1)) 
            xbox = BBOX(i,:);
            subIm = images(xbox(2):xbox(2)+xbox(4),xbox(1):xbox(1)+xbox(3),1:end);
            subIm = imresize(subIm, [128, 128]);
            imshow(subIm);

            outputGroupFileName = fullfile(output_folder_group, [sub_file_List(k,:).name...
                 '_' num2str(j) '_' num2str(i) '.jpg']);
            disp(outputGroupFileName);
            imwrite(subIm, outputGroupFileName);
        end
    end
end

%% Get all the faces form the Group Image - Images
FaceDetector = vision.CascadeObjectDetector('FrontalFaceCART','MergeThreshold',25);
source_folder = 'C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW\Group11of11';
output_folder_group = 'C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW\Faces_Group';
cd(source_folder);

sub_file_List = dir('*.jpg');
for k = 1: numel(sub_file_List)
    images = imread(sub_file_List(k,:).name);
%     for j = 1: v.NumberOfFrames
%         images = read(v,j);
        images = rgb2gray(images);
        imshow(images);

        BBOX = step(FaceDetector, images);
        disp(['detection completed, we found: ', num2str(size(BBOX(:,1)))]);

        for i = 1: size(BBOX(:,1)) 
            xbox = BBOX(i,:);
            subIm = images(xbox(2):xbox(2)+xbox(4),xbox(1):xbox(1)+xbox(3),1:end);
            subIm = imresize(subIm, [128, 128]);
    %         imshow(subIm);

            outputGroupFileName = fullfile(output_folder_group, [sub_file_List(k,:).name...
                 '_' num2str(j) '_' num2str(i) '.jpg']);
            disp(outputGroupFileName);
            imwrite(subIm, outputGroupFileName);
        end
%     end
end

%% the resize fuction
sub_path = fullfile(path,'Faces_DataSource');

files = dir(sub_path);
files = files(3:55,:);

iterate the subfolders
% for i = 1: 1
for i = 1: numel(files)
    sub_path_by_face = fullfile(sub_path, files(i).name);
    
    files_by_face = dir(sub_path_by_face);
    files_by_face = files_by_face(3:numel(files_by_face),:);
    
        for j = 1: numel(files_by_face)
            image_path = fullfile(files_by_face(j).folder, files_by_face(j).name);
            disp(files_by_face(j).name);
            Image = imread(image_path);
            subIm = imresize(Image, [128, 128]);
            imwrite(subIm, image_path);           
        end
    disp('Resize Completed!');
end