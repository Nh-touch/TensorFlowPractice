import numpy as np
import matplotlib.pyplot as plt


# 利用MGA算法，计算函数f(x) = sin(10x) * x + cos(2x)*x 在区间[0, 5]之间最大值点
DNA_SIZE        = 10                # DNA长度
POP_SIZE        = 20                # 种群大小     
BIRTH_RATE      = 0.6               # 交叉配对率   
VARIATION_RATE  = 0.01              # 变异率
NUM_GENERATIONS = 200               # 主循环次数
X_BOUND         = [0, 5]            # 自变量区间

# 函数表达式
def F(x): 
    return np.sin(10 * x) * x + np.cos(2 * x) * x

class MGA(object):
    # 默认函数
    def __init__(self, dna_size, pop_size, birth_rate, variation_rate, dna_bound, x_bound):
        self._dna_size        = dna_size
        self._pop_size        = pop_size
        self._birth_rate      = birth_rate
        self._variation_rate  = variation_rate
        self._dna_bound       = dna_bound
        self._x_bound         = x_bound

        # 生成目标DNA

        # 生成随机种群
        self._pop = np.random.randint(*self._dna_bound, size=(1, self._dna_size)).repeat(self._pop_size, axis=0)

    # 内部函数
    # 翻译DNA
    def _translate_dna(self, pop):
        return pop.dot(2 ** np.arange(self._dna_size)[::-1]) / float(2 ** self._dna_size - 1) * self._x_bound[1]

    # 获取适应度
    def _get_fitness(self, product):
        return product

    # 交叉配对
    def _generation(self, loser_winner):
        cross_idx = np.empty((self._dna_size,)).astype(np.bool)
        for i in range(self._dna_size):
            cross_idx[i] = True if np.random.rand() < self._birth_rate else False
        loser_winner[0, cross_idx] = loser_winner[1, cross_idx]  
        return loser_winner

    # 变异
    def _variation(self, loser_winner):
        mutation_idx = np.empty((self._dna_size,)).astype(np.bool)
        for i in range(self._dna_size):
            mutation_idx[i] = True if np.random.rand() < self._variation_rate else False 

        loser_winner[0, mutation_idx] = ~loser_winner[0, mutation_idx].astype(np.bool)
        return loser_winner

    # 自然选择
    # 外部接口
    def evolve(self, n):
        for _ in range(n):
            # 每次从种群中随机选取两个DNA比较，选择loser进行交叉配对和变异（对loser尝试改良）
            sub_pop_idx = np.random.choice(np.arange(0, self._pop_size), size=2, replace=False)
            sub_pop     = self._pop[sub_pop_idx]
            product     = F(self._translate_dna(sub_pop))
            fitness     = self._get_fitness(product)
            loser_winner_idx = np.argsort(fitness)
            loser_winner = sub_pop[loser_winner_idx]
            loser_winner = self._generation(loser_winner)
            loser_winner = self._variation(loser_winner)
            self._pop[sub_pop_idx] = loser_winner

        dna_prod = self._translate_dna(self._pop)
        pred     = F(dna_prod)
        return dna_prod, pred

if __name__ == '__main__':
    ga = MGA(dna_size        = DNA_SIZE 
           , pop_size        = POP_SIZE
           , birth_rate      = BIRTH_RATE
           , variation_rate  = VARIATION_RATE
           , dna_bound       = [0, 1]
           , x_bound         = X_BOUND)

    # 作函数对应图像
    plt.ion()       
    x = np.linspace(*X_BOUND, 200)
    plt.plot(x, F(x))

    for generation in range(NUM_GENERATIONS):
        dna_prod, pred = ga.evolve(5)

        # 绘图实时显示各点的位置
        if 'sca' in globals(): sca.remove()
        sca = plt.scatter(dna_prod, pred, s = 200, lw = 0, c ='red', alpha = 0.5)
        plt.pause(0.05)

    plt.ioff()
    plt.show()

