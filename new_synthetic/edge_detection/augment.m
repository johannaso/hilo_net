function [uniformAug,strucAug,gtAug] = augment(uniformImg,strucImg,gtImg)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
imgArray = {uniformImg,strucImg,gtImg};

%random crop

win1 = centerCropWindow2d(size(uniformImg),size(uniformImg));

cropUni = imcrop(uniformImg,win1);
cropStruc = imcrop(strucImg,win1);
cropGt = imcrop(gtImg,win1);

figure();
imshow(cropUni),title('Uniform Crop');
figure();
imshow(cropStruc),title('Struc Crop');
figure();
imshow(cropGt),title('Gt Crop')

%random affine
tform1 = randomAffine2d(Rotation=[35 55]);

affineUni = imwarp(uniformImg,tform1);
affineStruc = imwarp(strucImg,tform1);
affineGt = imwarp(gtImg,tform1);

figure();
imshow(affineUni),title('Uniform Affine');
figure();
imshow(affineStruc),title('Struc Affine');
figure();
imshow(affineGt),title('Gt Affine')

outCellArray = augment(augmenter,imgArray);

