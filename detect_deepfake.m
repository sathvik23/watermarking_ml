% Define paths for real and fake image directories
realImageDir = 'rfinal';
fakeImageDir = 'ffinal';

% Check if directories exist
if ~exist(realImageDir, 'dir')
    disp(['Real image directory does not exist: ', realImageDir]);
else
    disp(['Real image directory exists: ', realImageDir]);
end

if ~exist(fakeImageDir, 'dir')
    disp(['Fake image directory does not exist: ', fakeImageDir]);
else
    disp(['Fake image directory exists: ', fakeImageDir]);
end

% Get a list of all images in the directories
realImages = dir(fullfile(realImageDir, '*.jpg'));
fakeImages = dir(fullfile(fakeImageDir, '*.jpg'));

% Display the number of images found and the first few file names
disp(['Number of real images: ', num2str(length(realImages))]);
if length(realImages) > 0
    disp('First few real images:');
    disp({realImages(1:min(5,end)).name}');
end

disp(['Number of fake images: ', num2str(length(fakeImages))]);
if length(fakeImages) > 0
    disp('First few fake images:');
    disp({fakeImages(1:min(5,end)).name}');
end

% Initialize feature and label arrays
features = [];
labels = [];

% Set a fixed length for the radialProfile arrays
fixedLength = 100;

% Process real images
for i = 1:length(realImages)
    tic; % Start timer for processing each image
    image = imread(fullfile(realImageDir, realImages(i).name));
    grayImage = rgb2gray(image); % Convert to grayscale

    % Apply Discrete Fourier Transform (DFT)
    dftImage = fft2(double(grayImage));
    dftShifted = fftshift(dftImage); % Shift zero frequency component to center

    % Compute power spectrum
    powerSpectrum = abs(dftShifted).^2;

    % Apply azimuthal averaging to get 1D power spectrum
    [numRows, numCols] = size(powerSpectrum);
    centerX = round(numRows / 2);
    centerY = round(numCols / 2);
    maxRadius = min(centerX, centerY);
    radialProfile = zeros(1, maxRadius);
    for r = 1:maxRadius
        mask = createCircularMask(numRows, numCols, centerX, centerY, r);
        radialProfile(r) = mean(powerSpectrum(mask));
    end

    % Normalize the radial profile
    radialProfile = radialProfile / max(radialProfile);

    % Ensure radialProfile has consistent length
    radialProfile = ensureFixedLength(radialProfile, fixedLength);

    % Append features and labels
    features = [features; radialProfile];
    labels = [labels; 1]; % Label for real images

    elapsedTime = toc; % End timer for processing each image
    disp(['Processed real image ', num2str(i), '/', num2str(length(realImages)), ' in ', num2str(elapsedTime), ' seconds']);
end

% Process fake images
for i = 1:length(fakeImages)
    tic; % Start timer for processing each image
    image = imread(fullfile(fakeImageDir, fakeImages(i).name));
    grayImage = rgb2gray(image); % Convert to grayscale

    % Apply Discrete Fourier Transform (DFT)
    dftImage = fft2(double(grayImage));
    dftShifted = fftshift(dftImage); % Shift zero frequency component to center

    % Compute power spectrum
    powerSpectrum = abs(dftShifted).^2;

    % Apply azimuthal averaging to get 1D power spectrum
    [numRows, numCols] = size(powerSpectrum);
    centerX = round(numRows / 2);
    centerY = round(numCols / 2);
    maxRadius = min(centerX, centerY);
    radialProfile = zeros(1, maxRadius);
    for r = 1:maxRadius
        mask = createCircularMask(numRows, numCols, centerX, centerY, r);
        radialProfile(r) = mean(powerSpectrum(mask));
    end

    % Normalize the radial profile
    radialProfile = radialProfile / max(radialProfile);

    % Ensure radialProfile has consistent length
    radialProfile = ensureFixedLength(radialProfile, fixedLength);

    % Append features and labels
    features = [features; radialProfile];
    labels = [labels; 0]; % Label for fake images

    elapsedTime = toc; % End timer for processing each image
    disp(['Processed fake image ', num2str(i), '/', num2str(length(fakeImages)), ' in ', num2str(elapsedTime), ' seconds']);
end

% Display the size of the features and labels arrays
disp(['Size of features: ', num2str(size(features))]);
disp(['Size of labels: ', num2str(size(labels))]);

% Save features and labels
save('features.mat', 'features', 'labels');

% Step 2: Train the SVM Model
% Load extracted features and labels
load('features.mat', 'features', 'labels');

% Ensure labels are in a numeric vector format
labels = double(labels); % Convert to double if necessary

% Check the size and type of labels
disp(['Labels are of type: ', class(labels)]);
disp(['Size of labels: ', num2str(size(labels))]);

% Split the data into training and testing sets
cv = cvpartition(labels, 'HoldOut', 0.2);
XTrain = features(training(cv), :);
YTrain = labels(training(cv), :);
XTest = features(test(cv), :);
YTest = labels(test(cv), :);

% Train the SVM model
svmModel = fitcsvm(XTrain, YTrain, 'KernelFunction', 'rbf', 'Standardize', true);

% Save the trained model
save('trainedSVMModel.mat', 'svmModel');

% Test the model
YPred = predict(svmModel, XTest);

% Calculate accuracy
accuracy = sum(YPred == YTest) / length(YTest);
disp(['Test Accuracy: ', num2str(accuracy * 100), '%']);

% Load the trained SVM model
load('trainedSVMModel.mat', 'svmModel');

% Load a new image to classify (example image loading)
% Replace this with the actual image you want to classify
newImage = imread('path/to/new/image.jpg'); % Or directly load your image variable

% Classify the new image
isFake = classifyImageFromFile(newImage, svmModel);

if isFake
    disp('The image is classified as a deep fake.');
else
    disp('The image is classified as real.');
end

% Function definitions at the end of the file

% Helper function to create a circular mask
function mask = createCircularMask(numRows, numCols, centerX, centerY, radius)
    [X, Y] = meshgrid(1:numCols, 1:numRows);
    distance = sqrt((X - centerX).^2 + (Y - centerY).^2);
    mask = distance <= radius;
end

% Function to ensure radialProfile has consistent length
function radialProfile = ensureFixedLength(radialProfile, fixedLength)
    if length(radialProfile) < fixedLength
        radialProfile = [radialProfile, zeros(1, fixedLength - length(radialProfile))];
    elseif length(radialProfile) > fixedLength
        radialProfile = radialProfile(1:fixedLength);
    end
end

% Function to classify a new image directly
function isFake = classifyImageFromFile(imageFile, svmModel)
    % Convert the image to grayscale if it is not already
    if size(imageFile, 3) == 3
        grayImage = rgb2gray(imageFile);
    else
        grayImage = imageFile;
    end

    % Apply Discrete Fourier Transform (DFT)
    dftImage = fft2(double(grayImage));
    dftShifted = fftshift(dftImage); % Shift zero frequency component to center

    % Compute power spectrum
    powerSpectrum = abs(dftShifted).^2;

    % Apply azimuthal averaging to get 1D power spectrum
    [numRows, numCols] = size(powerSpectrum);
    centerX = round(numRows / 2);
    centerY =round(numCols / 2);
maxRadius = min(centerX, centerY);
radialProfile = zeros(1, maxRadius);
for r = 1:maxRadius
mask = createCircularMask(numRows, numCols, centerX, centerY, r);
radialProfile(r) = mean(powerSpectrum(mask));
end

% Normalize the radial profile
radialProfile = radialProfile / max(radialProfile);

% Ensure radialProfile has consistent length
radialProfile = ensureFixedLength(radialProfile, 100); % Ensure fixed length

% Predict the class (real or fake)
label = predict(svmModel, radialProfile);

% Return whether the image is fake
isFake = label == 0;

end 