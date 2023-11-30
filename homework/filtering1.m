% Load the image
originalImage = imread('./lab_week4/homework/peppers.png');

% Convert to grayscale
grayImage = im2gray(originalImage);

% Add salt and pepper noise
noisyImageSP = imnoise(grayImage, 'salt & pepper', 0.03); % You can adjust noise density

% Filter with a 3x3 mean filter
meanFilteredSP = filter2(fspecial('average', [3, 3]), noisyImageSP);

% Filter with a 3x3 median filter
medianFilteredSP = medfilt2(noisyImageSP, [3, 3]);

% Filter with a Gaussian filter
gaussianFilteredSP = imfilter(noisyImageSP, fspecial('gaussian', [3, 3], 0.5)); % You can adjust the standard deviation

% Display the salt and pepper noisy and filtered images
figure;
subplot(2, 2, 1);
imshow(noisyImageSP);
title('Salt & Pepper Noisy Image');

subplot(2, 2, 2);
imshow(meanFilteredSP);
title('Mean Filtered Image');

subplot(2, 2, 3);
imshow(medianFilteredSP);
title('Median Filtered Image');

subplot(2, 2, 4);
imshow(gaussianFilteredSP);
title('Gaussian Filtered Image');

% Add Gaussian noise
noisyImageGaussian = imnoise(grayImage, 'gaussian', 0, 1/256); % Adjust as needed

% Filter with a 3x3 mean filter
meanFilteredGaussian = filter2(fspecial('average', [3, 3]), noisyImageGaussian);

% Filter with a 3x3 median filter
medianFilteredGaussian = medfilt2(noisyImageGaussian, [3, 3]);

% Filter with a Gaussian filter
gaussianFilteredGaussian = imfilter(noisyImageGaussian, fspecial('gaussian', [3, 3], 0.5)); % Adjust the standard deviation

% Display the Gaussian noisy and filtered images
figure;
subplot(2, 2, 1);
imshow(noisyImageGaussian);
title('Gaussian Noisy Image');

subplot(2, 2, 2);
imshow(meanFilteredGaussian);
title('Mean Filtered Image');

subplot(2, 2, 3);
imshow(medianFilteredGaussian);
title('Median Filtered Image');

subplot(2, 2, 4);
imshow(gaussianFilteredGaussian);
title('Gaussian Filtered Image');
