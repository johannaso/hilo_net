function [structuredImage] = structuredIllumination(maskedImage,uniformImage,maskInv)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    [M,N] = size(maskedImage);
    block_size = 5;
    P = ceil(M / block_size);
    Q = ceil(N / block_size);
    pattern = checkerboard(block_size, P, Q) > 0.5;
    pattern = pattern(1:M, 1:N);
    checkerPattern = pattern + maskInv;
    
    %alpha_data_mask = bsxfun(@times, pattern, cast(mask,'like', pattern));


    figure();
    h = imshow(maskedImage);
    set(h, 'AlphaData', checkerPattern);

    F = getframe(gcf);
    [X, Map] = frame2im(F);
    structured = [X, Map];
    
    for k=1:size(structured,1)
        for l=1:size(structured,2)
            if(structured(k,l)==1)
                structured(k,l)=0;
            end
        end
    end

    figure();
    imshow(structured);

    structuredImage = uint8(maskedImage) + uniformImage;
end