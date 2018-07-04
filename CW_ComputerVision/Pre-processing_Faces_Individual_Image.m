%% Face detection in one image
output_folder = 'C:\Users\Francis\OneDrive - City, University of London\MSc_Data_Science_at_City_University\Course\INM460 - Computer Vision\CW\ComputerVision_CW\Faces_DataSource';

FaceDetector = vision.CascadeObjectDetector('FrontalFaceCART','MergeThreshold',25);

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
%         disp(numFaces);
        
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