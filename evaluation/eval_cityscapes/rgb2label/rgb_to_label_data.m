clear; clc;
%% maps
load map_full.mat
map = map / 255;
%% file name
fileFolder = fullfile('~/disk/Bayesian-CycleGAN/results/cityscapes_bayes_L1_lsgan/test_latest/images');
file_png = dir(fullfile(fileFolder,'*fake_B.png'));
file_png = {file_png.name}';
[file_png,INDEX] = sort_nat(file_png);
num_file = length(file_png);
%% rgb to label
cd ~/disk/Bayesian-CycleGAN/evaluate/eval_cityscapes/fake_B
for i = 1 : num_file
    i
    cd ~/disk/Bayesian-CycleGAN/results/cityscapes_bayes_L1_lsgan/test_latest/images
    file_name = file_png{i};
    file = imread(file_name);           % rgb
    file = rgb2ind(file, map);          % label
    file(find(file == 0)) = 3;
    cd ~/disk/Bayesian-CycleGAN/evaluate/eval_cityscapes/fake_B
    % file_name = file_name(1:length(file_name)-16);
    % file_name = strcat(file_name,'.png');
    imwrite(file, file_name);
    % fprintf(file_name)
end

