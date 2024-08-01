% Read the image
imagePath = 'cameraman.tif'; % Replace with your image file path
img = imread(imagePath);

% Check if the image is RGB and convert to grayscale if necessary
if size(img, 3) == 3
    grayImg = im2gray(img);
else
    grayImg = img;
end

% Convert the grayscale image to binary
bwImg = imbinarize(grayImg);

% Display the binary image
figure;
imshow(bwImg);
title('Select ROI');

% Allow the user to select the ROI
roi = drawrectangle('Label','ROI');
addlistener(roi, 'ROIMoved', @(src, evt) title('ROI Selected - Press Enter to Confirm'));

% Wait for user to press Enter to confirm ROI
uiwait(msgbox('Adjust the ROI and press Enter in the command window to confirm.'));

% Capture Enter key press to continue
pause;  % This will pause until any key is pressed

% Get the ROI mask
roiMask = createMask(roi, bwImg);

% Confirm the selected ROI by displaying it
figure;
imshow(bwImg);
hold on;
h = imshow(roiMask);
set(h, 'AlphaData', 0.5); % Set transparency for the ROI overlay
title('Selected ROI');

% Complement the ROI mask to get the RONI mask
roniMask = ~roiMask;

% Extract the ROI segment
roiSegment = bwImg(roiMask);

% Extract the RONI segment
roniSegment = bwImg(roniMask);

% Flatten the segments to 1D arrays
roiSegment = roiSegment(:);
roniSegment = roniSegment(:);

% Convert the segments to uint8 for hashing
roiSegment = uint8(roiSegment * 255);
roniSegment = uint8(roniSegment * 255);

% Compute SHA-512 hash for ROI
roiHash = DataHash(roiSegment, struct('Method', 'SHA-512', 'Format', 'hex'));

% Compute SHA-512 hash for RONI
roniHash = DataHash(roniSegment, struct('Method', 'SHA-512', 'Format', 'hex'));

% Display the SHA-512 hashes
disp(['SHA-512 Hash for ROI: ', roiHash]);
disp(['SHA-512 Hash for RONI: ', roniHash]);

% DataHash function for computing SHA-512
function Hash = DataHash(Data, Opt)
    % Validate inputs
    if nargin < 1
        error('DataHash:BadInput', 'Missing input data.');
    end
    if nargin < 2
        Opt.Method = 'SHA-512';
        Opt.Format = 'hex';
    end
    
    % Convert data to uint8 array
    if ischar(Data)
        Data = uint8(Data);
    elseif isnumeric(Data) || islogical(Data)
        Data = typecast(Data(:), 'uint8');
    elseif iscell(Data)
        Data = cellfun(@DataHash, Data, 'UniformOutput', false);
        Data = [Data{:}];
    else
        error('DataHash:BadInput', 'Unsupported data type.');
    end
    
    % Compute hash
    Engine = java.security.MessageDigest.getInstance(Opt.Method);
    Engine.update(Data);
    Hash = typecast(Engine.digest, 'uint8');
    
    % Format hash
    if strcmp(Opt.Format, 'hex')
        Hash = sprintf('%.2x', Hash);
    elseif strcmp(Opt.Format, 'base64')
        Hash = matlab.net.base64encode(Hash);
    else
        error('DataHash:BadInput', 'Unsupported hash format.');
    end
end
