clc; clear;
%% load images
origin = imread('input.png');
cyclegan_0 = imread('cyclegan_0.png');
cyclegan_5 = imread('cyclegan_5.png');
bayes_0 = imread('bayes_0.png');
bayes_5 = imread('bayes_5.png');

%% claculate GDL
alpha = 1;
origin = int16(origin);
cyclegan_0 = int16(cyclegan_0);
cyclegan_5 = int16(cyclegan_5);
bayes_0 = int16(bayes_0);
bayes_5 = int16(bayes_5);
origin_diff_row = abs(origin(2:128, 2:255, :) - origin(1:127, 2:255, :));
origin_diff_column = abs(origin(2:128, 2:255, :) - origin(2:128, 1:254, :));
cyclegan_0_diff_row = abs(cyclegan_0(2:128, 2:255, :) - cyclegan_0(1:127, 2:255, :));
cyclegan_0_diff_column = abs(cyclegan_0(2:128, 2:255, :) - cyclegan_0(2:128, 1:254, :));
cyclegan_5_diff_row = abs(cyclegan_5(2:128, 2:255, :) - cyclegan_5(1:127, 2:255, :));
cyclegan_5_diff_column = abs(cyclegan_5(2:128, 2:255, :) - cyclegan_5(2:128, 1:254, :));
bayes_0_diff_row = abs(bayes_0(2:128, 2:255, :) - bayes_0(1:127, 2:255, :));
bayes_0_diff_column = abs(bayes_0(2:128, 2:255, :) - bayes_0(2:128, 1:254, :));
bayes_5_diff_row = abs(bayes_5(2:128, 2:255, :) - bayes_5(1:127, 2:255, :));
bayes_5_diff_column = abs(bayes_5(2:128, 2:255, :) - bayes_5(2:128, 1:254, :));
gdl_cyclegan_0 = sum(sum((abs(origin_diff_row - cyclegan_0_diff_row)).^alpha + (abs(origin_diff_column - cyclegan_0_diff_column)).^alpha));
gdl_cyclegan_5 = sum(sum((abs(origin_diff_row - cyclegan_5_diff_row)).^alpha + (abs(origin_diff_column - cyclegan_5_diff_column)).^alpha));
gdl_bayes_0 = sum(sum((abs(origin_diff_row - bayes_0_diff_row)).^alpha + (abs(origin_diff_column - bayes_0_diff_column)).^alpha));
gdl_bayes_5 = sum(sum((abs(origin_diff_row - bayes_5_diff_row)).^alpha + (abs(origin_diff_column - bayes_5_diff_column)).^alpha));

fprintf('gdl of cyclegan_0: %f \n', gdl_cyclegan_0 / 127 / 254);
fprintf('gdl of cyclegan_5: %f \n', gdl_cyclegan_5 / 127 / 254);
fprintf('gdl of bayes_0: %f \n', gdl_bayes_0 / 127 / 254);
fprintf('gdl of bayes_5: %f \n', gdl_bayes_5 / 127 / 254);