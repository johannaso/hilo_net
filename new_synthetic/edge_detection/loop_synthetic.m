close all;
clear all;
% Specify the folder where the files live.
files=dir('/Users/johanna/Desktop/hilo_net/new_synthetic/image_database/Nuclei/**/*.jpg'); 

foregroundImageArray = files(1:2:end);
backgroundImageArray = files(2:2:end);

if (length(foregroundImageArray) > length(backgroundImageArray))
    foregroundImageArray(end) = [];
end

for i = 1:100
    foregroundImageElement = foregroundImageArray(i);
    foregroundImage = imread(foregroundImageElement.name);
    foregroundImage = rgb2gray(foregroundImage);

    backgroundImageElement = backgroundImageArray(i);
    backgroundImage = imread(backgroundImageElement.name);
    backgroundImage = rgb2gray(backgroundImage);

    [groundTruth,uniformWithBackground, struc] = syntheticHilo(foregroundImage, backgroundImage);

    %save images
    path_uni = '/Users/johanna/Desktop/hilo_net/new_synthetic/synthetic_data/uniform/';
    path_struc = '/Users/johanna/Desktop/hilo_net/new_synthetic/synthetic_data/structured/';
    path_groundt = '/Users/johanna/Desktop/hilo_net/new_synthetic/synthetic_data/ground_truth/';


    name_uni = append('uni_', int2str(i),'.png');
    name_struc = append('structured_', int2str(i),'.png');
    name_groundt = append('groundt_', int2str(i),'.png');

    imwrite(uniformWithBackground, append(path_uni, name_uni));
    imwrite(struc, append(path_struc, name_struc));
    imwrite(groundTruth, append(path_groundt, name_groundt));
end