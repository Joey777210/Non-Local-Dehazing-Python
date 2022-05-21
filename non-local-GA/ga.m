% GA��ز�������
% �������� = ��������
gen_size = 1000;
% ��������
iterator_num = 10;
% һ��Ⱦɫ����Ⱥ��С;
generation_size = 5;
% ��Ӧ������
adaptability = [];
% Ⱦɫ�帴�Ƶı���
cp = 0.2;
% ���Ƶ�Ⱦɫ������
copy_num = generation_size * cp;
% �������ɵ���Ⱦɫ������
crossover_num = generation_size - copy_num;
% ���̶�ѡ�����
selection_probability = [];
% GA�����õ������Ⱦɫ��
advance_chromosome = [];
% ��ǰ�˳�������
break_point = 0.95;
file_path = 'img/0559_haze_input.jpg';
params_path = 'img/0559_haze_params.txt';
out_img = '0559.jpg';

% ȥ����ز�������
% ������ȡֵ[p_min_range, p_max_range]
p_max_range = 1;
p_min_range = -1;

% ��ʼ����һ��Ⱦɫ��
generation = p_min_range + (p_max_range - p_min_range)*rand(generation_size, gen_size);
%% ������ʼ
for i = 1 : iterator_num
    disp(['��', num2str(i), '�ε�����ʼ'])
    
    %% ���㱾����Ӧ��
    disp('���㱾����Ӧ��')
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
        disp('��ǰ�˳�')
        return;
    end
    
    %% ������Ȼѡ�����
    disp('������Ȼѡ�����')
    selection_probability = [];
    sum_adaptability = 0;
    for index = 1:generation_size
        sum_adaptability = sum_adaptability + adaptability(index);
    end
    
    for index = 1:generation_size
        selection_probability = [selection_probability adaptability(index) / sum_adaptability];
    end
    
   %% ������һ��Ⱦɫ��
   disp('������һ��Ⱦɫ��')
   new_generation = [];
   
   % ����
   disp('����')

   for index = 1:crossover_num 
        baba = generation(rws(selection_probability), :);
        mama = generation(rws(selection_probability), :);
        cross_index = ceil(rand * gen_size);
                % ����crossover_point=3, number_of_variables=5
        % mask1 = 1     1     1     0     0
        % mask2 = 0     0     0     1     1
        mask1 = [ones(1, cross_index), zeros(1, gen_size - cross_index)];
        mask2 = not(mask1);
        
        % ��ȡ�ֿ���4��Ⱦɫ��
        % ע���� .*
        baba1 = mask1 .* baba;
        baba2 = mask2 .* baba;
        mama1 = mask1 .* mama;
        mama2 = mask2 .* mama;
        
        % �õ���һ��
        new_generation = [new_generation;mama1+baba2];

    end % Ⱦɫ�彻�����
    
    % ����
    disp('����')

    %�����һ��Ⱦɫ��
    chromosome_index = ceil(rand * generation_size);
    %�����һ������
    gen_index = ceil(rand * gen_size);
    %��������һ��������
    new_gen = p_min_range + (p_max_range - p_min_range)*rand;
    
    new_generation(chromosome_index, gen_index) = new_gen;

    % ����
    disp('����')

    % Ѱ����Ӧ����ߵ�N��Ⱦɫ����±�(N=Ⱦɫ������*���Ʊ���)
    [num, val] = sort(adaptability);

    for j = copy_num : generation_size
        new_generation = [new_generation; generation(j, :)];
    end
    generation = [];
    generation = new_generation;
    
    disp(['��', num2str(i), '�ε�������']);
end
disp('��������')
%% 

[max_value, max_value_index] = max(adaptability);
disp('��Ӧ��')
disp(adaptability)
disp('�����Ӧ��')
disp(max_value_index)
disp('Ⱦɫ�峤��')
disp(size(generation(max_value_index, :)))
[w, img_dehazed] = cal_w(chromosome, file_path, params_path);
imshow(img_dehazed)
imwrite(img_dehazed, out_img)

