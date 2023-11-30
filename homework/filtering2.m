% Load the image
originalImage = imread('./lab_week4/homework/peppers.png');

% Convert to grayscale
grayImage = im2gray(originalImage);

% Find edges using different edge detection methods
sobelEdges = edge(grayImage, 'sobel');
prewittEdges = edge(grayImage, 'prewitt');
robertsEdges = edge(grayImage, 'roberts');
logEdges = edge(grayImage, 'log');
cannyEdges = edge(grayImage, 'canny');

% Display the results
figure;
subplot(2, 3, 1);
imshow(sobelEdges);
title('Sobel');

subplot(2, 3, 2);
imshow(prewittEdges);
title('Prewitt');

subplot(2, 3, 3);
imshow(robertsEdges);
title('Roberts');

subplot(2, 3, 4);
imshow(logEdges);
title('Laplacian of Gaussian');

subplot(2, 3, 5);
imshow(cannyEdges);
title('Canny');
