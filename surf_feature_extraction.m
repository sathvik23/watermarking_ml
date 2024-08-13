clc;
clear;

% Add VLFeat to MATLAB path
run('vlfeat-0.9.21/toolbox/vl_setup');

% Load and preprocess the image
image = imread('office_1.jpg'); % Change path to your image as needed
if size(image, 3) == 3
    image = rgb2gray(image);
end

% Detect SIFT keypoints and compute descriptors
imageSingle = single(image);
[keypoints, descriptors] = vl_sift(imageSingle);

% Select top N keypoints based on their scale (strength)
N = 10;
[~, sortedIndices] = sort(keypoints(3, :), 'descend');
numValidPoints = min(N, length(sortedIndices));
topNDescriptors = descriptors(:, sortedIndices(1:numValidPoints));

% Concatenate descriptors into a single feature vector
featureVector = reshape(topNDescriptors, [], 1);

% Ensure the feature vector is at least 512 dimensions
if length(featureVector) < 512
    error('The feature vector has fewer dimensions than 512.');
end

% Select the first 512 elements from the feature vector
selectedFeatureVector = featureVector(1:512);

% Normalize the selected feature vector to the range [0, 1]
selectedFeatureVector = (selectedFeatureVector - min(selectedFeatureVector)) / (max(selectedFeatureVector) - min(selectedFeatureVector));

% Quantize the normalized feature vector to obtain a 512-bit sequence
threshold = median(selectedFeatureVector);  % Use the median value as the threshold
bitSequence = selectedFeatureVector > threshold;

% Print the 512-bit sequence
fprintf('512-bit sequence: \n');
disp(bitSequence');
