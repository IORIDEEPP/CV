%% the resize fuction
path = pwd;
sub_path = fullfile(path,'*.jpg');

list_images = dir('*.jpg');

files = dir(sub_path);
files = files(3:55,:);

% iterate the subfolders
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