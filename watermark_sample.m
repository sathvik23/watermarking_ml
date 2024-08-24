 %% Main Script

% Load a sample image
original_image = imread('football.jpg');
original_image = imresize(original_image, [512,512]);

% Generate a random sequence; we can replace it with an image
robust_watermark = randi([0, 1], 1, 512);
%disp(robust_watermark)

% Function to insert a watermark
watermarked_image = watermark_insertion(original_image, robust_watermark);

% PSNR - evaluation metric
psnr_value = calculate_psnr(original_image, watermarked_image);
fprintf('PSNR between original and watermarked images: %.2f dB\n', psnr_value);

% Save the watermarked image
imwrite(watermarked_image, 'image_00100_watermarked.png');
imwrite(original_image, 'image_00100.png');


% Display the original and watermarked images side by side
figure;
subplot(1, 2, 1);
imshow(original_image);
title('Original Image');


subplot(1, 2, 2);
imshow(watermarked_image);
title('Watermarked Image');


%% Functions

function watermarked_image = watermark_insertion(original_image, robust_watermark)
    block_size = 8;
    alpha = 0.1; % Embedding strength
    [rows, cols] = size(original_image);
    robust_watermarked_image = original_image;
    
    % Robust Watermark Insertion
    % Divide the image into non-overlapping blocks
    num_blocks = floor(rows / block_size) * floor(cols / block_size);
    
    for i = 1:length(robust_watermark)
        % Get the block index
        block_index = i;
        row = mod(block_index-1, floor(rows / block_size)) * block_size + 1;
        col = floor((block_index-1) / floor(rows / block_size)) * block_size + 1;
        
        % Extract the block
        img_block = double(original_image(row:row+block_size-1, col:col+block_size-1));
        
        % Apply IWT
        [LL1, LH1, HL1, HH1] = iwt(img_block);
        [LL2, LH2, HL2, HH2] = iwt(LH1);
        
        % Embed the watermark bit
        wrbit = robust_watermark(i);
        AvgLH2 = mean(LH2(:));
        AvgHL2 = mean(HL2(:));
        
        if wrbit == 1
            if AvgLH2 > AvgHL2
                LH2 = LH2 + alpha * (AvgLH2 - AvgHL2);
            else
                HL2 = HL2 + alpha * (AvgHL2 - AvgLH2);
            end
        else
            if AvgLH2 > AvgHL2
                HL2 = HL2 + alpha * (AvgLH2 - AvgHL2);
            else
                LH2 = LH2 + alpha * (AvgHL2 - AvgLH2);
            end
        end
        
        % Inverse IWT
        LH1 = iiwt(LL2, LH2, HL2, HH2);
        img_block = iiwt(LL1, LH1, HL1, HH1);
        
        % Update the robust watermarked image
        robust_watermarked_image(row:row+block_size-1, col:col+block_size-1) = uint8(img_block);
    end
    
    % Output the dual watermarked image
    watermarked_image = robust_watermarked_image;
end

function [LL, LH, HL, HH] = iwt(img_block)
    % Integer Wavelet Transform (IWT) using lifting scheme for Haar wavelet
    [m, n] = size(img_block);
    LL = zeros(m/2, n/2);
    LH = zeros(m/2, n/2);
    HL = zeros(m/2, n/2);
    HH = zeros(m/2, n/2);

    % Horizontal lifting step
    temp_L = zeros(m, n/2);
    temp_H = zeros(m, n/2);
    for i = 1:m
        for j = 1:2:n
            temp_L(i, (j+1)/2) = floor((img_block(i, j) + img_block(i, j+1)) / 2);
            temp_H(i, (j+1)/2) = img_block(i, j) - img_block(i, j+1);
        end
    end

    % Vertical lifting step
    for j = 1:n/2
        for i = 1:2:m
            LL((i+1)/2, j) = floor((temp_L(i, j) + temp_L(i+1, j)) / 2);
            LH((i+1)/2, j) = temp_L(i, j) - temp_L(i+1, j);
            HL((i+1)/2, j) = temp_H(i, j) + temp_H(i+1, j);
            HH((i+1)/2, j) = temp_H(i, j) - temp_H(i+1, j);
        end
    end
end

function img_block = iiwt(LL, LH, HL, HH)
    % Inverse Integer Wavelet Transform (IIWT) using lifting scheme for Haar wavelet
    [m, n] = size(LL);
    temp_L = zeros(m*2, n);
    temp_H = zeros(m*2, n);
    img_block = zeros(m*2, n*2);

    % Inverse vertical lifting step
    for j = 1:n
        for i = 1:m
            temp_L(2*i-1, j) = LL(i, j) + LH(i, j);
            temp_L(2*i, j) = LL(i, j) - LH(i, j);
            temp_H(2*i-1, j) = HL(i, j) + HH(i, j);
            temp_H(2*i, j) = HL(i, j) - HH(i, j);
        end
    end

    % Inverse horizontal lifting step
    for i = 1:2*m
        for j = 1:n
            img_block(i, 2*j-1) = temp_L(i, j) + temp_H(i, j);
            img_block(i, 2*j) = temp_L(i, j) - temp_H(i, j);
        end
    end
end

function psnr_value = calculate_psnr(original_image, watermarked_image)
    % Ensure the images are of the same size
    assert(all(size(original_image) == size(watermarked_image)), 'Images must have the same dimensions');
    
    % Convert images to double precision if they are not already
    original_image = double(original_image);
    watermarked_image = double(watermarked_image);
    
    % Calculate Mean Squared Error (MSE)
    mse = mean((original_image(:) - watermarked_image(:)).^2);
    
    % Calculate PSNR
    if mse == 0
        psnr_value = Inf;
    else
        max_pixel = 255; % Assuming the image has 8-bit depth
        psnr_value = 20 * log10(max_pixel / sqrt(mse));
    end
end