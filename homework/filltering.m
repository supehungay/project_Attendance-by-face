% Load hình ảnh
original_image = imread('./lab_week4/homework/peppers.png');

% Hiển thị hình ảnh
figure;
subplot(3, 2, 1);
imshow(image_original);
title('Ảnh gốc');

% Brighten 
% brighten_image = brighten(original_image, 0.5);
brighten_image = imadjust(original_image, [0.1, 0.7], [0, 1]);
subplot(3, 2, 2);
imshow(brighten_image);
title("Làm sáng");

% Contrast
contrast_image = imadjust(original_image, [0.2, 0.8], [0, 1]);
subplot(3, 2, 3);
imshow(contrast_image);
colormap gray;
title('Contrast');

% Histogram Equalization
equalizedImage = histeq(original_image);
subplot(3, 2, 4);
imshow(equalizedImage);
title('Histogram Equalized Image');

%  Image contrast
subplot(3, 2, 5);
a = imshow(original_image);
imcontrast(a);


% imadjust
adjustedImage = imadjust(original_image, [0.2, 0.8], []);
subplot(3, 2, 6);
imshow(adjustedImage);
title('Adjusted Image');

set(gcf, 'Position', get(0, 'Screensize'));