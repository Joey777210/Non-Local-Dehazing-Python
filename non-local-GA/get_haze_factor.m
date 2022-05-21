function [w] = get_haze_factor(img_dehazed)
mu = 5.1;
nu = 2.9;
sigma = 0.2461;
img_dehazed = double(img_dehazed);
dI = min(img_dehazed, [], 3);

bI = max(img_dehazed, [], 3);

cI = bI - dI;

d = mean(dI(:));

b = mean(bI(:));

c = b - d;

lamda = 1/3;

bmax = max(bI(:));

A0 = lamda * bmax + (1 - lamda) * b;

x1 = (A0 - d)/A0;

x2 = c/A0;

w = exp(-0.5 * (mu * x1 + nu * x2) + sigma);

