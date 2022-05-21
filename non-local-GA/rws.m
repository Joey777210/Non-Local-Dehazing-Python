function [i] = rws(selection_probability)

sum = 0;
rand_num = rand;

for i = 1:length(selection_probability(:))
    sum = sum + selection_probability(i);
    if sum >= rand_num
        break
    end
end
