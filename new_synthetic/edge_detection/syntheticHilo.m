function [groundTruth,uniformWithBackground, struc] = syntheticHilo(foregroundImage, backgroundImage)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
        

%edge detection


    %find edges

    BW2 = edge(foregroundImage,'canny');
    
    %fill gaps in outlines
    se90 = strel('line',3,90);
    se0 = strel('line',3,0);
    BWsdil = imdilate(BW2,[se90 se0]);
    %figure();
    %imshow(BWsdil)
    
    %fill shapes (white)
    BWdfill = imfill(BWsdil,'holes');
    %figure();
    %imshow(BWdfill);
    
    %smoothen
    seD = strel('diamond',1);
    BWfinal = imerode(BWdfill,seD);
    %figure();
    %imshow(BWfinal);
    
    %find random numer of object at random spot
    numberOfObjects = randi([10 20],1,1);
    
    
    [whiteRows, whiteColumns] = find(BWfinal);
    
    randX = [];
    randY = [];
    
    
    for i=1:numberOfObjects
        randXValue = whiteRows(randi(length(whiteRows),1));
        randX = [randX, randXValue];
        randYValue = whiteColumns(randi(length(whiteColumns),1));
        randY = [randY, randYValue];
    end
    
    
    
    foundObject = false;
    while foundObject==false
        objects = bwselect(BWfinal,randY,randX);
        foundObject = true;
        blackImage = sum(sum(foundObject));
        if(foundObject<1)
            randX = [];
            randY = [];
            randomIndexX = randi(length(whiteRows), 1);
            randomIndexY = randi(length(whiteColumns),1);
            
            for i=1:numberOfObjects
                randXValue = whiteRows(randi(length(whiteRows),1));
                randX = [randX, randXValue];
                randYValue = whiteColumns(randi(length(whiteColumns),1));
                randY = [randY, randYValue];
            end
            foundObject=false;  
        end
    end
    
    %figure();
    
    %imshowpair(BWfinal,objects,'montage');
    
    %%
    %use randomly generated mask as mask for HiLo image
    mask = objects;
    maskInverted = imcomplement(mask);
    
    maskedImageInv = bsxfun(@times, foregroundImage, cast(maskInverted, 'like', foregroundImage));
    maskedImage = bsxfun(@times, foregroundImage, cast(mask,'like', foregroundImage ));
    
    %ground truth image 
    groundTruth = maskedImage;
    groundTruth = imgaussfilt(groundTruth,0.3);
    
    %uniform image
    H = fspecial('disk',10);
    H2 = fspecial('disk',9);
    
    blurred = imfilter(maskedImageInv,H,'replicate');
    blurredBackground = imfilter(backgroundImage,H2,"replicate");
%     figure();
%     imshow(blurredBackground);
    overlapMask = mask & maskInverted;
    uniform = blurred+maskedImage;
    uniformWithBackground = uniform + blurredBackground;
    uniformWithBackground = imgaussfilt(uniformWithBackground,0.5);
    %figure();
    %imshowpair(uniform,uniformWithBackground,'montage');
    %title('uniform');
    %figure();
    %imshow(groundTruth);
    %title('ground truth');
    
    %%
    
    %structured image
    pixels = 5;   
    checkerPattern = zeros(size(BWfinal));
    
    %find dividor of size if default value  doesn't work 
    if (mod(size(BWfinal,1),pixels) ~= 0)
        pixels = 2;
        while(mod(size(BWfinal,1),pixels) ~= 0)
            pixels= pixles+1;
        end
    end
    
    
    checkerPattern = checkerboard(pixels, (size(BWfinal,1)/pixels)/2, (size(BWfinal,2)/pixels/2)) > 0.5;
    maskedCheckerpattern = bsxfun(@times, checkerPattern, cast(mask,'like', checkerPattern ));
    %figure();
    %imshow(maskedCheckerpattern);
    
    % use MIMT tools to fudge the lighting color lazily
    sfg = size(maskedCheckerpattern);
    vol = radgrad(sfg,[0.5 0.3],0.7,[20 20 50; 169 250 89],'ease','uint8');
    maskedCheckerpattern = imflatfield(maskedCheckerpattern,50);
    maskedCheckerpattern = imblend(vol,maskedCheckerpattern,1,'multiply');
    %maskedCheckerpattern = imblend(vol,maskedCheckerpattern,1,'overlay',1.5);
    struc = maskedCheckerpattern + im2double(uniformWithBackground);
    
    
%     figure();
%     subplot(2,2,1),imshow(foregroundImage),title('Original Image');
%     subplot(2,2,2),imshow(uniformWithBackground),title('Uniform Image');
%     subplot(2,2,3),imshow(struc),title('Structured Image');
%     subplot(2,2,4),imshow(groundTruth),title('HiLo/Groundtruth Image');






end