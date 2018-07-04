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
