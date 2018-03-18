import numpy as np
import matplotlib.pyplot as plt
     
POP_SIZE = 500                     
BIRTH_RATE = 0.1                  
VARIATION_RATE = 0.02            
NUM_GENERATIONS = 500
NUM_CITIES = 20  # DNA size

class TravelSalesProblemGA(object):
    # 默认函数
    def __init__(self, dna_size, pop_size, birth_rate, variation_rate):
        self._dna_size        = dna_size
        self._pop_size        = pop_size
        self._birth_rate      = birth_rate
        self._variation_rate  = variation_rate

        # 生成随机种群
        self._pop             =  np.vstack([np.random.permutation(self._dna_size) for _ in range(self._pop_size)])

    # 内部函数
    # 翻译DNA
    def _translate_dna(self, city_position):
        x = np.empty_like(self._pop, dtype=np.float64)
        y = np.empty_like(self._pop, dtype=np.float64)
        for i, d in enumerate(self._pop):
            city_coord = city_position[d]
            x[i, :]    = city_coord[:, 0]
            y[i, :]    = city_coord[:, 1]
            
        return x, y


    # 获取适应度
    def _get_fitness(self, x, y):
        total_distance = np.empty((x.shape[0],), dtype=np.float64)
        for i, (xs, ys) in enumerate(zip(x, y)):
            total_distance[i] = np.sum(np.sqrt(np.square(np.diff(xs)) + np.square(np.diff(ys))))
        fitness = np.exp(self._dna_size * 2 / total_distance)
        
        return fitness, total_distance

    # 交叉配对
    def _generation(self, parent, pop):
        if np.random.rand() < self._birth_rate:
            i_ = np.random.randint(0, self._pop_size, size = 1)                       
            cross_points = np.random.randint(0, 2, self._dna_size).astype(np.bool)
            keep_city = parent[~cross_points]
            swap_city = pop[i_, np.isin(pop[i_].ravel(), keep_city, invert = True)]
            parent[:] = np.concatenate((keep_city, swap_city))
            
        return parent

    # 变异
    def _variation(self, child):
        for point in range(self._dna_size):
            if np.random.rand() < self._variation_rate:
                swap_point = np.random.randint(0, self._dna_size)
                swapA, swapB = child[point], child[swap_point]
                child[point], child[swap_point] = swapB, swapA
        return child

    # 自然选择
    def _select(self, city_position):
        lx, ly = ga._translate_dna(city_position)
        fitness, _ = self._get_fitness(lx, ly)
        
        idx = np.random.choice(np.arange(self._pop_size), 
                               size    = self._pop_size, 
                               replace = True, 
                               p       = fitness/fitness.sum())
        return self._pop[idx]

    # 外部接口

    
    def evolve(self, data):
        pop = self._select(data._city_position)
        pop_copy = pop.copy()
        for parent in pop:
            child = self._generation(parent, pop_copy)
            child = self._variation(child)
            parent[:] = child
        self._pop = pop

    def is_perfect(self):
        fitness = self._get_fitness()
        best_dna = self._pop[np.argmax(fitness)]
        best_phrase = self._translate_dna(best_dna)
        return (best_phrase == self._target_sentence)

    def dump(self, data, generation):
        lx, ly = ga._translate_dna(data._city_position)
        fitness, total_distance = self._get_fitness(lx, ly)

        best_idx = np.argmax(fitness)
        print('Gen:', generation, '| best fit: %.2f' % fitness[best_idx],)
        data.plotting(lx[best_idx], ly[best_idx], total_distance[best_idx])


class DataCollection(object):
    def __init__(self, n_cities):
        self._city_position = np.random.rand(n_cities, 2)
        plt.ion()

    def plotting(self, lx, ly, total_d):
        plt.cla()
        plt.scatter(self._city_position[:, 0].T, self._city_position[:, 1].T, s=100, c='k')
        plt.plot(lx.T, ly.T, 'r-')
        plt.text(-0.05, -0.05, "Total distance=%.2f" % total_d, fontdict={'size': 20, 'color': 'red'})
        plt.xlim((-0.1, 1.1))
        plt.ylim((-0.1, 1.1))
        plt.pause(0.01)

if __name__ == '__main__':
    ga = TravelSalesProblemGA(dna_size        = NUM_CITIES
                            , pop_size        = POP_SIZE
                            , birth_rate      = BIRTH_RATE
                            , variation_rate  = VARIATION_RATE)
    
    data = DataCollection(NUM_CITIES)
    
    for generation in range(NUM_GENERATIONS):
        ga.evolve(data)
        ga.dump(data, generation)
        
    plt.ioff()
    plt.show()

