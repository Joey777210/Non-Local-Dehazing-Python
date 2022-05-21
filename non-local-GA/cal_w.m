function [w, img_dehazed] = cal_w(chromosome, file_path, params_path)

img_hazy = imread(file_path);
fid = fopen(params_path,'r');
[C] = textscan(fid,'%s %f');
fclose(fid);
gamma = C{2}(1);

A = reshape(estimate_airlight(im2double(img_hazy).^(gamma)),1,1,3);
[img_dehazed, trans_refined] = non_local_dehazing(img_hazy, A, gamma, chromosome);
w = get_haze_factor(img_dehazed);
