import random
import non_local

# GA相关参数定义
# 基因数量 = 分组数量
gen_size = 1000
# 迭代次数
iterator_num = 50
# 一代染色体种群大小
generation_size = 10
# 适应度数组
adaptability = []
# 染色体复制的比例
_cp = 0.2
# 复制的染色体数量
copy_num = int(generation_size * _cp)
# 交叉生成的新染色体数量
crossover_num = generation_size - copy_num
# 轮盘赌选择概率
selection_probability = []
# GA迭代得到的最好染色体
advance_chromosome = []
# 提前退出的条件
break_point = 0.95

# 去雾相关参数定义
# 补偿项取值[p_min_range, p_max_range]
p_max_range = 1000
p_min_range = -1000


def create_new_gen():
    gen = random.randint(p_min_range, p_max_range - 1) / 1000
    return gen


# 轮盘赌算法
# param selectionProbability 概率数组(下标：元素编号、值：该元素对应的概率)
# returns {number} 返回概率数组中某一元素的下标
def RWS(selection_probability):
    sum = 0
    rand = random.random()
    for i in range(len(selection_probability)):
        sum += selection_probability[i]
        if sum >= rand:
            return i


# 交叉生成新的染色体
def cross(generation):
    new_generation = []
    for index in range(int(crossover_num)):
        # 采用轮盘赌选择父母染色体
        baba = generation[RWS(selection_probability)]
        mama = generation[RWS(selection_probability)]
        # 交叉
        cross_index = random.randint(0, gen_size - 1)
        baba = baba[:cross_index]
        mama = mama[cross_index:]
        baba = baba + mama
        # debugger
        new_generation.append(baba)

    return new_generation


def mutation(new_generation):
    # 随机找一条染色体
    chromosome_index = random.randint(0, crossover_num - 1)

    # 随机找一个基因
    gen_index = random.randint(0, gen_size - 1)

    # 重新生成一个变异后的
    new_gen = create_new_gen()

    new_generation[chromosome_index][gen_index] = new_gen

    return new_generation


# 复制(复制上一代中优良的染色体)
# @param chromosomeMatrix 上一代染色体矩阵
# @param newChromosomeMatrix 新一代染色体矩阵
def copy(pre_generation, new_generation):
    # 寻找适应度最高的N条染色体的下标(N=染色体数量*复制比例)
    chromosome_index_arr = maxN(adaptability, copy_num)

    # 复制
    for i in range(len(chromosome_index_arr)):
        chromosome = pre_generation[chromosome_index_arr[i][0]]
        new_generation.append(chromosome)

    return new_generation


# 从数组中寻找最大的n个元素
def maxN(array, n):
    # 将一切数组升级成二维数组，二维数组的每一行都有两个元素构成[原一位数组的下标,值]
    matrix = []
    for i in range(len(array)):
        matrix.append([i, array[i]])

    # 对二维数组排序
    for i in range(n):
        for j in range(len(matrix)):
            if matrix[j - 1][1] > matrix[j][1]:
                temp = matrix[j - 1]
                matrix[j - 1] = matrix[j]
                matrix[j] = temp

    # 取最大的n个元素, 从后往前取n个
    max_index_arr = []
    for i in range(n):
        max_index_arr.insert(0, matrix[len(matrix) - i - 1])

    return max_index_arr


# 生成新一代染色体
def createGeneration(pre_generation=None):
    if pre_generation is None:
        # 初始化种群
        new_generation = []
        for i in range(generation_size):
            chromosome_i = []
            for gen in range(gen_size):
                chromosome_i.append(create_new_gen())
            new_generation.append(chromosome_i)

        # TODO: 记录每一代种群

        return new_generation

    # 交叉生成{crossoverMutationNum}条染色体
    new_generation = cross(pre_generation)

    # 变异
    new_generation = mutation(new_generation)

    # 复制
    new_generation = copy(pre_generation, new_generation)

    # TODO: 记录每一代种群

    return new_generation


# 计算本代各染色体的适应度
# 根据去雾效果来定义适应度
def calAdaptability(generation):
    adaptability.clear()

    # 计算每条染色体去雾得到的图片对应计算的haze degree
    for index in range(generation_size):
        chromosome = generation[index]
        w = non_local.cal_w(chromosome)

        # 适应度 = 1 - w. w越小，雾霾程度越低，适应度越大.
        adapt = 1 - w
        adaptability.append(adapt)

        # 如果中途出现了很优秀的种群，则退出计算
        if adapt > break_point:
            global advance_chromosome
            advance_chromosome = chromosome
            break


# 计算自然选择概率
def calSelectionProbability():
    selection_probability.clear()

    # 计算适应度总和
    sum_adaptability = 0
    for i in range(generation_size):
        sum_adaptability += adaptability[i]

    # 计算每条染色体的选择概率
    for i in range(generation_size):
        selection_probability.append(adaptability[i] / sum_adaptability)


def ga_search():
    # 初始化第一代染色体
    generation = createGeneration()

    # 迭代繁衍
    for itIndex in range(iterator_num):
        # 计算上一代各条染色体的适应度
        calAdaptability(generation)
        if advance_chromosome is not None and len(advance_chromosome) > 0:
            # 提前退出
            print("提前退出, 迭代次数：", itIndex)
            return advance_chromosome

        # 根据适应度计算自然选择概率
        calSelectionProbability()

        # 生成新一代染色体
        generation = createGeneration(generation)
        print("迭代次数： ", itIndex)
    max_adaptability_index = maxN(adaptability, 1)

    return generation[max_adaptability_index[0][0]]


def main():
    chromosome = ga_search()
    print(chromosome)
    w = non_local.cal_w(chromosome)
    print("调整后的雾霾参数: ", w)
    w = non_local.cal_w([])
    print("调整前的雾霾参数: ", w)

    non_local.get_pic(chromosome)


if __name__ == '__main__':
    main()

