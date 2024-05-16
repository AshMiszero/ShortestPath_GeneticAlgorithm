import numpy as np
import matplotlib.pyplot as plt
import random
from deap import creator, tools, base


creator.create('FitnessMin', base.Fitness, weights=(-1.0, 1.0))
creator.create('Individual', list, fitness=creator.FitnessMin)

gen_size = 54
toolbox = base.Toolbox()
toolbox.register('Sequence', np.random.permutation, gen_size)
toolbox.register('Individual', tools.initIterate,
                 creator.Individual, toolbox.Sequence)

toolbox.register('Population', tools.initRepeat, list, toolbox.Individual)

nodeDict = {}
with open("node.txt", 'r') as f:
    tmp = f.readlines()
    for t in tmp:
        t = t.split()
        nodeDict[t[0]] = list(map(int, t[1:]))

nodeLine = {}
with open("line.txt", 'r') as f:
    tmp = f.readlines()
    for t in tmp:
        t = t.split()
        nodeLine[t[0]] = list(map(int, t[1:]))

edgeTime = {}
with open("time.txt", 'r') as f:
    tmp = f.readlines()
    for t in tmp:
        t = t.split()
        edgeTime[t[0]+t[1]] = float(t[2])
        edgeTime[t[1]+t[0]] = float(t[2])

edgeCost = {}
with open("dis.txt", 'r') as f:
    tmp = f.readlines()
    for t in tmp:
        t = t.split()
        edgeCost[t[0]+t[1]] = float(t[2])
        edgeCost[t[1]+t[0]] = float(t[2])

T = 5


def dfs(ind, source, target):
    path = []
    visited = []
    stack = [source]
    lens = []
    while stack:
        cur = stack[-1]
        if cur not in visited:
            path.append(cur)
            visited.append(cur)
            allow = np.array(nodeDict[str(cur)])
            priority_ls = np.asarray(ind)[np.asarray(allow)-1]
            index = np.argsort(priority_ls)
            stack.extend(allow[index])
            lens.append(len(allow[index]))
            if target in stack:
                path.append(target)
                break
        else:
            stack.pop()
            lens[-1] -= 1
            if lens[-1] == 0:
                lens.pop()
                path.pop()
    return path


def eval(ind):
    path = dfs(ind, source, target)
    time = TIME(path)
    cost = COST(path)
    return (time), (cost)


def TIME(path):
    # 若路径上只有两个结点，直接计算路径第一段的时间
    time = edgeTime[str(path[0])+str(path[1])]
    # 当路径上的结点大于2时，进行一般性的时间计算
    if len(path) > 2:
        # 通过前三个结点可以唯一确定出发时的线路
        # 对三个结点的线路做交集，即可得到出发线路
        pre = set(nodeLine[str(path[0])]) & set(
            nodeLine[str(path[1])]) & set(nodeLine[str(path[2])])
        # 加上第二段的时间
        time += edgeTime[str(path[1])+str(path[2])]
        for i in range(2, len(path)-1):
            time += edgeTime[str(path[i])+str(path[i+1])]
            # 通过对当前结点线路和下一结点线路做交集，得到当前线路
            now = set(nodeLine[str(path[i])]) & set(nodeLine[str(path[i+1])])
            # 如果之前线路与当前线路不同，说明进行了换乘
            if not pre & now:
                pre = now
                time += T
    return time


def COST(path):
    dis = 0
    # 计算总距离
    for i in range(len(path)-1):
        dis += edgeCost[str(path[i])+str(path[i+1])]
    price = [2, 3, 4, 5, 6, 7]  # 价目表
    rank = [6, 11, 17, 24, 32]  # 分段标准
    # 判断总距离属于什么级别的收费
    rank.append(dis)
    rank.sort()
    cost = price[rank.index(dis)]
    return cost


for source in [54]:
    for target in range(53, 54):
        if target == source:
            continue
        toolbox.register('evaluate', eval)
        toolbox.register('select', tools.selTournament, tournsize=2)
        toolbox.register('mate', tools.cxOrdered)
        toolbox.register('mutate', tools.mutShuffleIndexes, indpb=0.5)
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register('avg', np.mean)
        stats.register('std', np.std)
        stats.register('min', np.min)
        stats.register('max', np.max)
        logbook = tools.Logbook()
        logbook.header = 'gen', 'avg', 'std', 'min', 'max'

        pop_size = 100
        N_GEN = 20
        CXPB = 0.8
        MUTPB = 0.2

        pop = toolbox.Population(n=pop_size)
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        record = stats.compile(pop)
        logbook.record(gen=0, **record)
        for gen in range(1+N_GEN):
            selectTour = toolbox.select(pop, pop_size)
            selectInd = list(map(toolbox.clone, selectTour))
            for child1, child2 in zip(selectInd[::2], selectInd[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            for ind in selectInd:
                if random.random() < MUTPB:
                    toolbox.mutate(ind)
                    del ind.fitness.values
            invalid_ind = [ind for ind in selectInd if not ind.fitness.valid]
            fitnesses = list(map(toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            combinedPop = pop + selectInd
            pop = tools.selBest(combinedPop, pop_size)
            record = stats.compile(pop)
            logbook.record(gen=gen, **record)
        print(logbook)
        bestInd = tools.selBest(pop, 1)[0]
        bestFit = bestInd.fitness.values
        print(f'{source} to {target}')
        print('最短耗时为:', bestFit[0])
        print('对应费用为:', bestFit[1])
        print('对应路径为:', dfs(bestInd, source, target))


min = logbook.select('min')
avg = logbook.select('avg')
gen = logbook.select('gen')
plt.plot(gen, min, 'b-', label='MIN_FITNESS')
plt.plot(gen, avg, 'r-', label='AVG_FITNESS')
plt.xlabel('gen')
plt.ylabel('fitness')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
