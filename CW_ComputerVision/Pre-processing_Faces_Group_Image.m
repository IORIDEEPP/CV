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