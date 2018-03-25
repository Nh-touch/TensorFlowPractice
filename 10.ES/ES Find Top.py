import numpy as np
import matplotlib.pyplot as plt


# 利用ES算法，计算函数f(x) = sin(10x) * x + cos(2x)*x 在区间[0, 5]之间最大值点
DNA_SIZE        = 1                 # DNA长度
POP_SIZE        = 100               # 种群大小     
NUM_KIDS        = 50                # 每次繁衍的后代数量
NUM_GENERATIONS = 200               # 主循环次数
DNA_BOUND       = [0, 5]            # 自变量区间
BIRTH_RATE      = 0.6               # 交叉配对率   
VARIATION_RATE  = 0.01              # 变异率

# 函数表达式
def F(x): 
    return np.sin(10 * x) * x + np.cos(2 * x) * x

class ES(object):
    # 默认函数
    def __init__(self, dna_size, pop_size, birth_rate, variation_rate, dna_bound, num_kids):
        self._dna_size        = dna_size
        self._pop_size        = pop_size
        self._birth_rate      = birth_rate
        self._variation_rate  = variation_rate
        self._dna_bound       = dna_bound
        self._num_kids        = num_kids

        # 生成目标DNA
        # 生成随机种群
        self._pop = dict(dna                = self._dna_bound[1] * np.random.rand(1, self._dna_size).repeat(self._pop_size, axis=0),
                         viriation_strength = np.random.rand(self._pop_size, self._dna_size))

        # 生成绘图：
        plt.ion()
        x = np.linspace(*self._dna_bound, 200)
        plt.plot(x, F(x))

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

    # 交叉配对
    def _generation(self):
        kids = {'dna': np.empty((self._num_kids, self._dna_size))}
        kids['viriation_strength'] = np.empty_like(kids['dna'])
        for kv, ks in zip(kids['dna'], kids['viriation_strength']):
            p1, p2  = np.random.choice(np.arange(self._pop_size), size=2, replace=False)
            cp      = np.random.randint(0, 2, self._dna_size, dtype = np.bool) 
            kv[cp]  = self._pop['dna'][p1, cp]
            kv[~cp] = self._pop['dna'][p2, ~cp]
            ks[cp]  = self._pop['viriation_strength'][p1, cp]
            ks[~cp] = self._pop['viriation_strength'][p2, ~cp]

        return kids

    # 变异
    def _variation(self, kids):
        for kv, ks in zip(kids['dna'], kids['viriation_strength']):
            ks[:] = np.maximum(ks + (np.random.rand(*ks.shape)-0.5), 0.)
            kv   += ks * np.random.randn(*kv.shape)
            kv[:] = np.clip(kv, *self._dna_bound)

    # 自然选择
    def _select(self, kids):
        for key in ['dna', 'viriation_strength']:
            self._pop[key] = np.vstack((self._pop[key], kids[key]))

        fitness = self._get_fitness(F(self._pop['dna']))
        idx = np.arange(self._pop['dna'].shape[0])
        good_idx = idx[fitness.argsort()][-self._pop_size:]
        for key in ['dna', 'viriation_strength']:
            self._pop[key] = self._pop[key][good_idx]

    # 外部接口
    def evolve(self):
        kids = self._generation()
        self._variation(kids)
        self._select(kids)

    # dump
    def dump(self):
        pass

if __name__ == '__main__':
    es = ES(dna_size        = DNA_SIZE 
          , pop_size        = POP_SIZE
          , birth_rate      = BIRTH_RATE
          , variation_rate  = VARIATION_RATE
          , dna_bound       = DNA_BOUND
          , num_kids        = NUM_KIDS)

    for generation in range(NUM_GENERATIONS):
        es.evolve()

        if 'sca' in globals(): 
            sca.remove()

        sca = plt.scatter(es._pop['dna']
                        , F(es._pop['dna'])
                        , s     = 200
                        , lw    = 0
                        , c     = 'red'
                        , alpha = 0.5)

        plt.pause(0.05)

        #es.dump()


