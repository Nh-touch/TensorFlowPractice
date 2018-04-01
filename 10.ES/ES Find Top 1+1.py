import numpy as np
import matplotlib.pyplot as plt


# 利用ES算法，计算函数f(x) = sin(10x) * x + cos(2x)*x 在区间[0, 5]之间最大值点
DNA_SIZE            = 1             # DNA长度
NUM_GENERATIONS     = 200           # 主循环次数
DNA_BOUND           = [0, 5]        # 自变量区间
VARIATION_STRENGTH  = 5.            # 变异强度

# 函数表达式
def F(x): 
    return np.sin(10 * x) * x + np.cos(2 * x) * x

class ES(object):
    # 默认函数
    def __init__(self, dna_size, variation_strength, dna_bound):
        self._dna_size              = dna_size
        self._variation_strength    = variation_strength
        self._dna_bound             = dna_bound

        # 生成目标DNA
        # 生成随机种群
        self._parent = 5 * np.random.rand(self._dna_size)

        # 生成绘图：
        plt.ion()

    def __del__(self):
        plt.ioff()
        plt.show()

    # 内部函数
    # 翻译DNA
    def _translate_dna(self, pop):
        return F(pop)

    # 获取适应度
    def _get_fitness(self, product):
        return product.flatten()

    # 变异
    def _variation(self):
        k = self._parent + self._variation_strength * np.random.randn(self._dna_size)
        k = np.clip(k, *self._dna_bound)
        return k

    # 自然选择
    def _select(self, kid, slow_down_flag):
        fp = self._get_fitness(F(self._parent))[0]
        fk = self._get_fitness(F(kid))[0]

        # if 4/5 of the samples are not better than now then reduce the strength
        p_target = 1/5

        if fp < fk:     # kid better than parent
            self._parent = kid
            ps = 1.     # kid win -> ps = 1 (successful offspring)
        else:
            ps = 0.

        # adjust global mutation strength
        if slow_down_flag:
            self._variation_strength *= np.exp(1/np.sqrt(self._dna_size + 1) * (ps - p_target)/(1 - p_target))

    # 外部接口
    def evolve(self, slow_down_flag):
        kid = self._variation()
        self._select(kid, slow_down_flag)

        plt.cla()
        plt.scatter(self._parent, F(self._parent), s=200, lw=0, c='red', alpha=0.5,)
        plt.scatter(kid, F(kid), s=200, lw=0, c='blue', alpha=0.5)
        plt.text(0, -7, 'Mutation strength=%.2f' % self._variation_strength)
        x = np.linspace(*self._dna_bound, 200)
        plt.plot(x, F(x))
        plt.pause(0.05)

    # dump
    def dump(self):
        pass

if __name__ == '__main__':
    es = ES(dna_size            = DNA_SIZE 
          , variation_strength  = VARIATION_STRENGTH
          , dna_bound           = DNA_BOUND)

    for generation in range(NUM_GENERATIONS):
        if generation < 50:
            es.evolve(False)
        else:
            es.evolve(True)

        #es.dump()


