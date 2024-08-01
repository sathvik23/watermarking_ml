%% Load the Trained Network

% Load the trained network from the .mat file
load('trainedNet.mat', 'net');

%% Prepare the Test Image

% Define the paths to the directories
testImageFolder = 'dataset'; % Change to your test image directory if different

% Read the test image
test_image_original = im2double(imread(fullfile(testImageFolder, 'original/image_00101.png'))); % Change the filename as needed
test_image_watermarked = im2double(imread(fullfile(testImageFolder, 'watermarked/image_00101.png'))); % Change the filename as needed

% Convert to grayscale if the images are RGB
if size(test_image_original, 3) == 3
    test_image_original = rgb2gray(test_image_original);
end
if size(test_image_watermarked, 3) == 3
    test_image_watermarked = rgb2gray(test_image_watermarked);
end

% Resize the test images to 512x512
test_image_original = imresize(test_image_original, [512, 512]);
test_image_watermarked = imresize(test_image_watermarked, [512, 512]);

% Normalize the test images
test_image_original = mat2gray(test_image_original);
test_image_watermarked = mat2gray(test_image_watermarked);

% Combine the original and watermarked images
input_image = cat(3, test_image_original, test_image_watermarked);

%% Predict the Watermark

% Use the trained network to predict the watermark
predicted_watermark = predict(net, input_image);
predicted_watermark = round(predicted_watermark)

%% Load the Ground Truth Watermark

% Load the ground truth watermark
ground_truth = load(fullfile(testImageFolder, 'watermarks/watermark_00101.mat')); % Change the filename as needed
ground_truth_watermark = ground_truth.robust_watermark;
% Assuming you have the ground_truth_watermark and predicted_watermark

disp(predicted_watermark)