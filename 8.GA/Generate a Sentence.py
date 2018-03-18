import numpy as np
TARGET_PHRASE = 'You get it!'       
POP_SIZE = 300                     
BIRTH_RATE = 0.4                  
VARIATION_RATE = 0.01            
NUM_GENERATIONS = 1000
DNA_BOUND = [32, 126]

class SentenceGA(object):
    # 默认函数
    def __init__(self, target_sentence, dna_bound, pop_size, birth_rate, variation_rate):
        self._target_sentence = target_sentence
        self._dna_size        = len(target_sentence)
        self._dna_bound       = dna_bound
        self._pop_size        = pop_size
        self._birth_rate      = birth_rate
        self._variation_rate  = variation_rate

        # 生成目标DNA
        self._target_dna      = np.fromstring(self._target_sentence, dtype=np.uint8)

        # 生成随机种群
        self._pop = np.random.randint(*self._dna_bound, 
                                      size = (self._pop_size, self._dna_size)).astype(np.int8)

    # 内部函数
    # 翻译DNA
    def _translate_dna(self, dna):
        return dna.tostring().decode('ascii')

    # 获取适应度
    def _get_fitness(self):
        return (self._pop == self._target_dna).sum(axis=1)

    # 交叉配对
    def _generation(self, parent, pop):
        if np.random.rand() < self._birth_rate:
            i_ = np.random.randint(0, self._pop_size, size = 1)                        
            cross_points = np.random.randint(0, 2, self._dna_size).astype(np.bool)  
            parent[cross_points] = pop[i_, cross_points]                          
        return parent

    # 变异
    def _variation(self, child):
        for point in range(self._dna_size):
            if np.random.rand() < self._variation_rate:
                child[point] = np.random.randint(*self._dna_bound) 
        return child

    # 自然选择
    def _select(self):
        fitness = self._get_fitness() + 1e-4
        idx = np.random.choice(np.arange(self._pop_size), 
                               size    = self._pop_size, 
                               replace = True, 
                               p       = fitness/fitness.sum())
        return self._pop[idx]

    # 外部接口
    def evolve(self):
        pop = self._select()
        pop_copy = pop.copy()
        for parent in pop:
            child = self._generation(parent, pop_copy)
            child = self._variation(child)
            parent = child
        self._pop = pop

    def is_perfect(self):
        fitness = self._get_fitness()
        best_dna = self._pop[np.argmax(fitness)]
        best_phrase = self._translate_dna(best_dna)
        return (best_phrase == self._target_sentence)

    def dump(self):
        fitness = self._get_fitness()
        best_dna = self._pop[np.argmax(fitness)]
        best_phrase = self._translate_dna(best_dna)
        print('Gen', generation, ': ', best_phrase)

if __name__ == '__main__':
    ga = SentenceGA(target_sentence = TARGET_PHRASE
                  , dna_bound       = DNA_BOUND
                  , pop_size        = POP_SIZE
                  , birth_rate      = BIRTH_RATE
                  , variation_rate  = VARIATION_RATE)

    for generation in range(NUM_GENERATIONS):
        if ga.is_perfect():
            break

        ga.evolve()
        ga.dump()
