% Read the image
image = imread('dataset/original/image_00101.png');

% Check if the image is already grayscale
if size(image, 3) == 3
    % Convert the image to grayscale if it's an RGB image
    gray_image = rgb2gray(image);
else
    % If the image is already grayscale, no conversion is needed
    gray_image = image;
end

% Define Sobel kernels
Gx = [-1 0 1; -2 0 2; -1 0 1];
Gy = [-1 -2 -1; 0 0 0; 1 2 1];

% Apply Sobel filter to original image
original_grad_magnitude = apply_sobel(gray_image, Gx, Gy);

% Simulate unintentional attack (JPEG compression)
imwrite(gray_image, 'temp_compressed.jpg', 'jpg', 'Quality', 50);
compressed_image = imread('temp_compressed.jpg');
compressed_grad_magnitude = apply_sobel(compressed_image, Gx, Gy);

% Calculate the mean gradient magnitudes
original_mean_gradient = mean(original_grad_magnitude(:));
compressed_mean_gradient = mean(compressed_grad_magnitude(:));

fprintf('Original Mean Gradient Magnitude: %.2f\n', original_mean_gradient);
fprintf('Compressed Image Mean Gradient Magnitude: %.2f\n', compressed_mean_gradient);

% Set the threshold for intentional attack (20% increase)
threshold = original_mean_gradient * 1.20;

% Classify the attack
if compressed_mean_gradient > threshold
    classification = 'The attack is classified as intentional.';
else
    classification = 'The attack is classified as unintentional.';
end

fprintf('%s\n', classification);

% Display the original, compressed, and gradient magnitude images
figure;

subplot(2, 2, 1);
imshow(gray_image, []);
title('Original Grayscale Image');

subplot(2, 2, 2);
imshow(compressed_image, []);
title('Compressed Image');

subplot(2, 2, 3);
imshow(original_grad_magnitude, []);
title('Gradient Magnitude (Original)');

subplot(2, 2, 4);
imshow(compressed_grad_magnitude, []);
title('Gradient Magnitude (Compressed)');

% Function definition must be placed at the end of the script
function grad_magnitude = apply_sobel(image, Gx, Gy)
    % Pad the image with zeros on the borders to handle edge pixels
    padded_image = padarray(image, [1 1], 0, 'both');

    % Get the size of the padded image
    [rows, cols] = size(padded_image);

    % Initialize gradient matrices
    grad_x = zeros(rows - 2, cols - 2);
    grad_y = zeros(rows - 2, cols - 2);

    % Slide the kernel over the image
    for i = 2:rows-1
        for j = 2:cols-1
            % Extract the 3x3 neighborhood
            neighborhood = padded_image(i-1:i+1, j-1:j+1);
            
            % Apply the Sobel kernels
            grad_x(i-1, j-1) = sum(sum(double(neighborhood) .* Gx));
            grad_y(i-1, j-1) = sum(sum(double(neighborhood) .* Gy));
        end
    end

    % Calculate the gradient magnitude
    grad_magnitude = sqrt(grad_x.^2 + grad_y.^2);
end
