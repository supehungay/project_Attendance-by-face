
% Load the RGB image
rgb_image = imread('./lab_week4/homework/picture.jpg');

% Convert RGB to HSV
hsv_image = rgb2hsv(rgb_image);

% Extract individual channels
redChannel = rgb_image(:, :, 1);
greenChannel = rgb_image(:, :, 2);
blueChannel = rgb_image(:, :, 3);

hueChannel = hsv_image(:, :, 1);
saturationChannel = hsv_image(:, :, 2);
valueChannel = hsv_image(:, :, 3);

% Display the grayscale images
figure;

subplot(2, 3, 1);
imshow(redChannel);
title('Red Channel');

subplot(2, 3, 2);
imshow(greenChannel);
title('Green Channel');

subplot(2, 3, 3);
imshow(blueChannel);
title('Blue Channel');

subplot(2, 3, 4);
imshow(hueChannel);
title('Hue Channel');

subplot(2, 3, 5);
imshow(saturationChannel);
title('Saturation Channel');

subplot(2, 3, 6);
imshow(valueChannel);
title('Value Channel');


% Convert RGB to normalized RGB coordinates
R = double(rgb_image(:, :, 1)) / 255;
G = double(rgb_image(:, :, 2)) / 255;
B = double(rgb_image(:, :, 3)) / 255;

% Calculate the normalized RGB coordinates
r = R ./ (R + G + B + 1e-10);
g = G ./ (R + G + B + 1e-10);
b = B ./ (R + G + B + 1e-10);

% Plot g versus r
figure;
scatter(r(:), g(:), 1); % Scatter plot for r and g

mask = f(r, g) >= 0.08;

% Create an output image by applying the mask
outputImage = rgb_image;
outputImage(repmat(~mask, [1, 1, 3])) = 0;
% Display the output image
imshow(outputImage);


% Extract hue and saturation channels
H = hsv_image(:, :, 1);
S = hsv_image(:, :, 2);

% Define the rules for determining face pixels
a = 0;
b = 0.6; 
c = 0.2; 
d = 0.9;

% Create a mask based on the rules
mask2 = (H >= a) & (H <= b) & (S >= c) & (S <= d);

% Create an output image by applying the mask
outputImage2 = rgb_image;
outputImage2(repmat(~mask2, [1, 1, 3])) = 0;

% Display the output image
imshow(outputImage2);

function mask = f(r, g)
    mask = r - g;
end


