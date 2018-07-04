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

% create a subfolder for all the labels contained in the source folder
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