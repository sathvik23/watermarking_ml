%% Data Preparation

% Define the paths to the directories
originalImageFolder = 'dataset/original';
watermarkedImageFolder = 'dataset/watermarked';
watermarkFolder = 'dataset/watermarks';

% Get list of image files
imageFiles = dir(fullfile(originalImageFolder, '*.png'));
num_images = length(imageFiles);

% Initialize arrays to store images and watermarks
originalImages = zeros(512, 512, 1, num_images);
watermarkedImages = zeros(512, 512, 1, num_images);
watermarks = zeros(num_images, 512); % Assuming each watermark is 512 bits

for i = 2:num_images
    % Generate filenames based on naming convention
    originalFile = fullfile(originalImageFolder, sprintf('image_%05d.png', i));
    watermarkedFile = fullfile(watermarkedImageFolder, sprintf('image_%05d.png', i));
    watermarkFile = fullfile(watermarkFolder, sprintf('watermark_%05d.mat', i));
    
    % Read images and watermark
    originalImage = im2double(imread(originalFile));
    watermarkedImage = im2double(imread(watermarkedFile));
    watermarkData = load(watermarkFile);
    watermark = watermarkData.robust_watermark; % Extract the watermark data from the struct
    
    % Convert to grayscale if the images are RGB
    if size(originalImage, 3) == 3
        originalImage = rgb2gray(originalImage);
    end
    if size(watermarkedImage, 3) == 3
        watermarkedImage = rgb2gray(watermarkedImage);
    end
    
    % Store images and watermark
    originalImages(:,:,1,i) = originalImage;
    watermarkedImages(:,:,1,i) = watermarkedImage;
    watermarks(i,:) = watermark;
end

% Combine original and watermarked images
combinedImages = cat(3, originalImages, watermarkedImages);

%% Network Architecture

inputSize = [512 512 2];
numHiddenUnits = 512;

layers = [
    imageInputLayer(inputSize, 'Name', 'input')
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1')
    
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2')
    
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool3')
    
    convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv4')
    batchNormalizationLayer('Name', 'bn4')
    reluLayer('Name', 'relu4')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool4')
    
    convolution2dLayer(3, 512, 'Padding', 'same', 'Name', 'conv5')
    batchNormalizationLayer('Name', 'bn5')
    reluLayer('Name', 'relu5')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool5')
    
    globalAveragePooling2dLayer('Name', 'gap')
    
    fullyConnectedLayer(512, 'Name', 'fc1')
    dropoutLayer(0.5, 'Name', 'dropout1')
    reluLayer('Name', 'relu6')
    fullyConnectedLayer(numHiddenUnits, 'Name', 'fc2')
    dropoutLayer(0.5, 'Name', 'dropout2')
    fullyConnectedLayer(numHiddenUnits, 'Name', 'fc3')
    regressionLayer('Name', 'output')
];

%% Training

% Split data into training and validation sets
numTrainImages = round(0.9 * num_images);
trainImages = combinedImages(:,:,:,1:numTrainImages);
trainLabels = watermarks(1:numTrainImages,:);

valImages = combinedImages(:,:,:,numTrainImages+1:end);
valLabels = watermarks(numTrainImages+1:end,:);

% Define training options
options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...        % Increased number of epochs
    'InitialLearnRate', 1e-4, ... % Reduced learning rate
    'MiniBatchSize', 8, ...     % Reduced batch size
    'Shuffle', 'every-epoch', ...
    'ValidationData', {valImages, valLabels}, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train the network
net = trainNetwork(trainImages, trainLabels, layers, options);



% Save the trained network
save('trainedNet.mat', 'net');