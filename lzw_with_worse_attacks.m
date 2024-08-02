% Main script
% Step 1: Convert the image to binary (black and white)
% Read the input image
originalImage = imread('cameraman.tif');

% Convert to grayscale if the image is in color
if size(originalImage, 3) == 3
    grayImage = rgb2gray(originalImage);
else
    grayImage = originalImage;
end

% Convert the grayscale image to binary
binaryImage = imbinarize(grayImage);

% Step 2: Allow the user to select the ROI and confirm the selection
imshow(binaryImage);
title('Select the ROI by drawing a rectangle');
roi = drawrectangle();
roiMask = createMask(roi);

% Prompt the user to confirm the selection
confirm = questdlg('Do you want to confirm the selected ROI?', 'Confirm ROI', 'Yes', 'No', 'Yes');
if strcmp(confirm, 'No')
    disp('Selection not confirmed. Exiting...');
    return;
end

% Step 3: Display the pixel values of the ROI
roiPixels = binaryImage(roiMask);
disp('Pixel values of the ROI:');
disp(roiPixels);

% Step 4: Compress the pixel values of the ROI using LZW compression
% Convert logical array to uint8 for LZW compression
roiPixelsUint8 = uint8(roiPixels);

% Flatten the ROI pixels for compression
roiPixelsFlat = roiPixelsUint8(:);
compressedROI = lzwCompress(roiPixelsFlat);

% Step 5: Simulate various attacks on the original image and display the attacked versions
% Convert binary image to uint8 before adding noise
binaryImageUint8 = uint8(binaryImage) * 255;

% Example of a severe attack: adding Salt-and-Pepper noise
attackedImage1 = imnoise(binaryImageUint8, 'salt & pepper', 0.1);
figure; imshow(attackedImage1); title('Attacked Image with Salt-and-Pepper Noise');

% Example of a severe attack: applying Gaussian blur
h = fspecial('gaussian', [5 5], 2.0);
attackedImage2 = imfilter(binaryImageUint8, h);
figure; imshow(attackedImage2); title('Attacked Image with Gaussian Blur');

% Example of a severe attack: JPEG compression artifacts
imwrite(binaryImageUint8, 'temp.jpg', 'Quality', 10);
attackedImage3 = imread('temp.jpg');
figure; imshow(attackedImage3); title('Attacked Image with JPEG Compression Artifacts');

% Step 6: Restore the ROI in the attacked images using the compressed LZW data
% Decompress the ROI
decompressedROIPixels = lzwDecompress(compressedROI);
decompressedROIPixels = uint8(decompressedROIPixels - '0');

% Ensure decompressed data matches the original ROI size
decompressedROIPixels = reshape(decompressedROIPixels, size(roiPixels));

% Convert decompressed logical ROI pixels to uint8 before replacement
decompressedROIPixelsUint8 = decompressedROIPixels * 255;

% Replace the ROI in the attacked images with the original values
restoredImage1 = attackedImage1;
restoredImage1(roiMask) = decompressedROIPixelsUint8;
figure; imshow(restoredImage1); title('Restored Image from Salt-and-Pepper Noise Attack');

restoredImage2 = attackedImage2;
restoredImage2(roiMask) = decompressedROIPixelsUint8;
figure; imshow(restoredImage2); title('Restored Image from Gaussian Blur Attack');

restoredImage3 = attackedImage3;
restoredImage3(roiMask) = decompressedROIPixelsUint8;
figure; imshow(restoredImage3); title('Restored Image from JPEG Compression Attack');

disp('Process completed successfully.');

% Function definitions must go at the end of the script
% LZW Compression function
function compressed = lzwCompress(data)
    dictionary = containers.Map('KeyType', 'char', 'ValueType', 'int32');
    for i = 0:255
        dictionary(char(i)) = i;
    end
    dataStr = char(data + '0');
    w = '';
    compressed = [];
    dictSize = 256;
    for i = 1:length(dataStr)
        c = dataStr(i);
        wc = [w c];
        if isKey(dictionary, wc)
            w = wc;
        else
            compressed = [compressed dictionary(w)];
            dictionary(wc) = dictSize;
            dictSize = dictSize + 1;
            w = c;
        end
    end
    if ~isempty(w)
        compressed = [compressed dictionary(w)];
    end
end

% LZW Decompression function
function decompressed = lzwDecompress(compressed)
    dictionary = containers.Map('KeyType', 'int32', 'ValueType', 'char');
    for i = 0:255
        dictionary(i) = char(i);
    end
    w = char(compressed(1));
    decompressed = w;
    dictSize = 256;
    for i = 2:length(compressed)
        k = compressed(i);
        if isKey(dictionary, k)
            entry = dictionary(k);
        elseif k == dictSize
            entry = [w w(1)];
        else
            error('Bad compressed k: %d', k);
        end
        decompressed = [decompressed entry];
        dictionary(dictSize) = [w entry(1)];
        dictSize = dictSize + 1;
        w = entry;
    end
end
