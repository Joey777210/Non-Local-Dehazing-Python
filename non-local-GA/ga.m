% GA相关参数定义
% 基因数量 = 分组数量
gen_size = 1000;
% 迭代次数
iterator_num = 10;
% 一代染色体种群大小;
generation_size = 5;
% 适应度数组
adaptability = [];
% 染色体复制的比例
cp = 0.2;
% 复制的染色体数量
copy_num = generation_size * cp;
% 交叉生成的新染色体数量
crossover_num = generation_size - copy_num;
% 轮盘赌选择概率
selection_probability = [];
% GA迭代得到的最好染色体
advance_chromosome = [];
% 提前退出的条件
break_point = 0.95;
file_path = 'img/0559_haze_input.jpg';
params_path = 'img/0559_haze_params.txt';
out_img = '0559.jpg';

% 去雾相关参数定义
% 补偿项取值[p_min_range, p_max_range]
p_max_range = 1;
p_min_range = -1;

% 初始化第一代染色体
generation = p_min_range + (p_max_range - p_min_range)*rand(generation_size, gen_size);
%% 迭代开始
for i = 1 : iterator_num
    disp(['第', num2str(i), '次迭代开始'])
    
    %% 计算本代适应度
    disp('计算本代适应度')
    adaptability = [];
    for index = 1 : generation_size
        chromosome = generation(index, :);
        [w, img_dehazed] = cal_w(chromosome, file_path, params_path);
        
        adapt = 1-w;
        adaptability = [adaptability adapt];
        if adapt > break_point
            advance_chromosome = chromosome;
            break
        end
    end
    
    disp(adaptability)

    if ~isempty(advance_chromosome) 
        disp('提前退出')
        return;
    end
    
    %% 计算自然选择概率
    disp('计算自然选择概率')
    selection_probability = [];
    sum_adaptability = 0;
    for index = 1:generation_size
        sum_adaptability = sum_adaptability + adaptability(index);
    end
    
    for index = 1:generation_size
        selection_probability = [selection_probability adaptability(index) / sum_adaptability];
    end
    
   %% 生成新一代染色体
   disp('生成新一代染色体')
   new_generation = [];
   
   % 交叉
   disp('交叉')

   for index = 1:crossover_num 
        baba = generation(rws(selection_probability), :);
        mama = generation(rws(selection_probability), :);
        cross_index = ceil(rand * gen_size);
                % 假如crossover_point=3, number_of_variables=5
        % mask1 = 1     1     1     0     0
        % mask2 = 0     0     0     1     1
        mask1 = [ones(1, cross_index), zeros(1, gen_size - cross_index)];
        mask2 = not(mask1);
        
        % 获取分开的4段染色体
        % 注意是 .*
        baba1 = mask1 .* baba;
        baba2 = mask2 .* baba;
        mama1 = mask1 .* mama;
        mama2 = mask2 .* mama;
        
        % 得到下一代
        new_generation = [new_generation;mama1+baba2];

    end % 染色体交叉结束
    
    % 变异
    disp('变异')

    %随机找一条染色体
    chromosome_index = ceil(rand * generation_size);
    %随机找一个基因
    gen_index = ceil(rand * gen_size);
    %重新生成一个变异后的
    new_gen = p_min_range + (p_max_range - p_min_range)*rand;
    
    new_generation(chromosome_index, gen_index) = new_gen;

    % 复制
    disp('复制')

    % 寻找适应度最高的N条染色体的下标(N=染色体数量*复制比例)
    [num, val] = sort(adaptability);

    for j = copy_num : generation_size
        new_generation = [new_generation; generation(j, :)];
    end
    generation = [];
    generation = new_generation;
    
    disp(['第', num2str(i), '次迭代结束']);
end
disp('迭代结束')
%% 

[max_value, max_value_index] = max(adaptability);
disp('适应度')
disp(adaptability)
disp('最大适应度')
disp(max_value_index)
disp('染色体长度')
disp(size(generation(max_value_index, :)))
[w, img_dehazed] = cal_w(chromosome, file_path, params_path);
imshow(img_dehazed)
imwrite(img_dehazed, out_img)

