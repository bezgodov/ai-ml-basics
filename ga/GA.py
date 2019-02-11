import numpy as np
from pathlib import Path
class GA:
    def __init__(self):
        script_location = Path(__file__).absolute().parent
        file_location = script_location / 'input.txt'
        input_file = file_location.open()

        # input_file = open('input.txt')

        # count of tasks
        self.N = int(input_file.readline())
        # task's difficulty
        self.tasks = np.array([int(val) for val in input_file.readline().split()], int)
        # hours per one task
        self.hours = np.array([float(val) for val in input_file.readline().split()], float)
        # count of developers
        self.M = int(input_file.readline())
        #coefficients
        coefs = []
        for _ in range(self.M):
            coefs.append(list(map(float, input_file.readline().split())))
        self.coefs = np.array(coefs)

        self.init()
        self.firstPopulation()

        input_file.close
    def init(self):
        # beginning amount of entities
        self.countEntities = 400
        # beginning amount of 
        self.countCrossOver = 100
        # maximum count of 
        self.numberOfTasks = len(self.tasks)
        # current generation
        self.generation = 1
        # max score
        self.maxScore = 0.0
        # number of iterations with no improvements of score
        self.countStaticScore = 0
        # crossing over different way chance
        self.crossOverHalfRate = 0.4
    def firstPopulation(self):
        self.entities = []
        for i in range(self.countEntities):
            self.addEntity(self.generation, np.random.choice(self.M, size = self.numberOfTasks), 'Developer_' + str(i))
        
        while True:
            self.naturalSelection()
            self.beginMutations(np.random.randint(0, len(self.entities), size = int((len(self.entities)) / 1), dtype = int))
            self.crossOver()
    
    def byScore_key(self, entity):
        return entity.score

    def naturalSelection(self):
        self.entities = sorted(self.entities, key = self.byScore_key, reverse = True)[:self.countCrossOver]
        print('BEST (G = ' + str(self.generation) + ', EG = ' + str(self.entities[0].generation) + '): ' + str(self.maxScore) + '; CURRENT: ' + str(self.entities[0].score))
        # print(' '.join(map(str, [val.score for val in self.entities[:5]])))
        self.printResult(self.entities[0].score, self.entities[0].genes)
        #countMutations = np.random.randint(1, int((len(self.entities)) / 4))
        self.entities = sorted(self.entities, key = self.byScore_key, reverse = True)
    
    def crossOver(self):
        self.generation += 1
        entities = np.array(self.entities.copy())
        # entitiesToChange = np.concatenate((np.random.choice(5, size = 50), np.arange(0, len(self.entities) - 50, dtype = int)))
        # entitiesToChange = np.random.randint(0, len(self.entities) - 1, size = len(self.entities), dtype = int)
        entitiesToChange = np.arange(0, len(entities), dtype = int)
        np.random.shuffle(entitiesToChange)
        while(len(self.entities) < self.countEntities):
            length = len(entities)
            # mutant chance
            if (len(entitiesToChange) <= 0):
                entitiesToChange = np.arange(0, len(entities), dtype = int)
                np.random.shuffle(entitiesToChange)
            #np.random.shuffle(entities)
            #first, second = np.random.randint(length - 1, size = 2, dtype = int)
            # print(entitiesToChange)
            
            first = entitiesToChange[-1:][0]
            entitiesToChange = np.delete(entitiesToChange, -1)
            second = entitiesToChange[-1:][0]
            entitiesToChange = np.delete(entitiesToChange, -1)

            genes = []
            if np.random.sample(1) < self.crossOverHalfRate:
                offset = int(np.random.randint(10, self.N - 10, dtype = int))
                genes = np.concatenate((entities[first].genes[:offset], entities[second].genes[offset:]))
            else:
                # switch = np.random.randint(4, 40, dtype = int)
                flag = True
                for i in range(0, self.N):
                    if flag:
                        genes.append(entities[first].genes[i])
                    else:
                        genes.append(entities[second].genes[i])
                    
                    if np.random.sample(1) < 0.5:
                        flag = not flag

                # genes = np.concatenate([self.entities[first].genes[:int(self.N / 2)], self.entities[second].genes[int(self.N / 2):]])
                # np.random.shuffle(genes)
            self.addEntity(self.generation, genes, 'Developer_' + str(length + 1))

        # if self.countStaticScore > 50:
        #     self.outOfLocalMinimum()

    def beginMutations(self, mutationsTo):
        if (self.maxScore > 1600.0) and (self.generation > 300) and (self.generation % 75 == 0):
            # if self.countStaticScore > round(500 / self.generation):
            for index in mutationsTo:
                self.mutation(index)
            # for indexEntity in range(len(self.entities / 4)):
            #     self.mutation(indexEntity)
                # self.mutation(self.entities[np.random.randint(len(self.entities) - 1, dtype = int)])
            # self.countStaticScore = 0
            print('MUTATED ' + str(len(mutationsTo)) + ' ENTITIES')

    def mutation(self, indexEntity):
        # entity = self.entities[indexEntity]
        size = round(np.random.randint(1, 2) * (indexEntity + 1), 0)
        if size > self.N / 4:
            size = self.N / 4
        genesToChange = np.random.choice(self.N, size = int(size), replace = False)
                    # distance = np.random.randint(15, 20)
                    # for i, _ in enumerate(self.entities[indexEntity].genes):
                    #     if i % distance == 0:
                    #         self.entities[indexEntity].genes[i] = np.random.randint(0, self.M, dtype = int)
                    # self.entities[indexEntity].score = self.countScore(self.entities[indexEntity].genes)
        for gene in genesToChange:
            self.entities[indexEntity].genes[gene] = np.random.randint(0, self.M, dtype = int)
            
        self.entities[indexEntity].score = self.countScore(self.entities[indexEntity].genes)
        # print('MUTATION (GE = ' + str(entity.generation) + '): ' + str(np.array_equal(before, entity.genes)))

    def outOfLocalMinimum(self):
        self.countStaticScore = 0
        for index in range(len(self.entities)):
            size = round(np.random.randint(5, 10, dtype = int) * (index + 1), 0)
            if size >= self.N / 2:
                size = self.N / 2
            genes = np.random.choice(self.N, size = int(size), replace = False)
            for gene in genes:
                self.entities[index].genes[gene] = np.random.randint(0, self.M, dtype = int)
            self.entities[index].score = self.countScore(self.entities[index].genes)
        print('OUT OF LOCAL MINIMUM')
    
    def addEntity(self, generation, genes, name = 'Developer'):
        entity = Entity(generation, genes, name)
        entity.score = self.countScore(entity.genes)
        self.entities.append(entity)
        return entity

    def countMaxTime(self, answer):
        res = [0] * self.M

        for i, developer in enumerate(answer):
            res[developer] += self.hours[i] * self.coefs[developer][self.tasks[i] - 1]
        return res
        
    def countScore(self, answer):
        return self.scoreFunc(self.countMaxTime(answer))

    def scoreFunc(self, values):
        return (1e6 / np.amax(values))

    def printResult(self, score, genes):
        if abs(float(score) - float(self.maxScore)) < 0.025:
            self.countStaticScore += 1
        else:
            self.countStaticScore = 0

        if (float(score) > float(self.maxScore)):
            self.maxScore = score

            if (float(score) > 1600.0):
                output_file = open('results/' + str(score) + '.txt', 'w')
                # output_file = open('' + str(score) + '.txt', 'w')
                output_file.write(' '.join(map(str, [x + 1 for x in genes])))
                output_file.close


class Entity:
    def __init__(self, generation, genes, name = 'Developer'):
        self.generation = generation
        self.genes = genes
        self.name = name
        self.score = 0.0

instance = GA()
# print(
#     instance.countScore(
#         np.array(list(map(int, '10 9 3 8 8 7 5 9 6 3 8 10 5 5 9 9 3 4 1 1 1 2 5 2 8 5 9 10 1 5 7 5 10 9 8 5 5 2 8 6 4 8 10 4 9 7 8 8 6 2 2 4 6 1 9 5 2 2 5 8 1 8 7 6 7 10 10 5 10 4 3 2 6 7 5 1 2 6 6 1 5 9 4 5 4 8 10 9 2 3 2 4 2 10 5 5 8 2 9 3 6 6 5 4 6 10 9 6 4 9 6 2 6 5 10 5 3 2 9 3 4 2 3 2 10 1 2 4 8 6 7 5 1 9 7 2 8 3 7 8 2 6 3 7 9 6 2 5 9 3 9 6 10 8 7 2 6 3 3 8 4 9 5 4 8 4 1 3 7 3 2 3 3 7 10 4 1 10 4 2 5 9 2 6 8 6 3 10 5 2 3 9 10 5 1 10 2 4 9 7 10 10 6 7 2 2 4 5 5 1 6 10 9 2 7 10 3 2 4 4 2 5 5 4 2 3 10 8 1 10 10 1 7 10 6 7 5 4 9 10 1 7 2 9 8 2 3 4 3 10 10 8 2 3 10 1 2 5 10 6 2 5 6 3 4 8 10 5 8 4 2 3 1 3 6 8 10 3 1 10 9 4 3 6 9 9 3 4 2 2 4 1 10 5 4 2 2 6 10 4 5 8 8 10 5 8 7 1 2 3 7 2 9 3 3 1 9 2 8 4 2 3 7 6 3 9 1 7 2 9 2 9 2 4 3 8 4 8 4 8 2 6 8 6 9 2 2 7 9 2 6 8 2 1 6 9 2 3 7 5 7 5 2 6 3 9 1 7 10 6 2 9 2 2 5 2 10 8 1 9 10 8 7 9 6 10 4 2 1 7 4 2 3 3 1 2 8 2 9 3 1 9 6 2 6 3 3 9 10 8 10 3 2 5 4 6 1 9 1 2 8 10 5 4 2 6 7 7 4 8 2 2 1 10 8 2 2 5 3 2 9 7 10 6 10 7 8 8 8 10 7 1 2 4 10 1 4 7 2 2 2 1 6 2 7 1 9 5 8 6 8 10 4 4 8 4 6 5 10 8 1 8 9 2 3 6 7 2 6 6 2 2 8 9 3 6 10 6 3 3 3 2 4 7 1 2 6 10 6 10 8 6 8 10 7 10 6 5 3 5 2 7 10 2 2 9 10 1 1 3 6 1 10 10 2 7 1 9 7 1 1 8 9 3 7 1 2 3 4 2 9 7 10 10 4 2 4 10 6 10 5 8 9 10 9 6 7 10 5 7 10 1 2 6 1 7 7 4 2 3 9 6 2 4 5 7 10 7 10 10 7 1 10 9 8 10 6 10 10 1 5 4 3 10 8 4 10 8 6 3 8 2 3 5 6 1 5 1 9 2 2 3 7 6 10 9 4 1 10 4 2 2 10 9 4 9 2 8 4 3 1 10 6 5 9 3 9 8 8 2 6 1 6 5 1 8 5 9 1 2 6 4 4 6 2 4 2 2 9 2 3 3 3 10 10 10 5 4 2 2 1 8 5 9 10 4 1 10 6 2 9 6 3 8 7 3 8 6 10 3 3 10 6 6 7 8 7 4 8 9 6 7 2 10 5 10 3 8 6 5 4 1 5 6 9 2 5 7 2 5 7 1 2 4 2 4 1 10 10 4 7 2 6 4 4 4 7 6 2 9 7 10 2 3 1 7 3 1 4 1 7 5 8 4 1 3 3 3 7 10 1 2 1 3 2 5 10 8 10 3 4 4 7 2 5 8 10 9 1 7 6 6 4 5 1 6 10 9 1 4 1 2 7 1 7 3 1 8 5 5 5 2 4 6 7 5 8 7 1 10 9 6 4 10 8 8 2 4 9 7 1 10 6 3 10 7 3 5 10 2 4 2 3 3 7 2 9 6 8 3 8 10 3 5 4 10 2 10 2 3 6 7 6 5 7 2 2 1 7 1 8 8 2 10 6 2 9 2 3 3 4 2 5 5 10 5 5 2 6 7 5 7 6 6 2 4 4 5 7 3 2 5 6 3 8 4 6 1 10 2 3 3 3 9 3 1 4 10 5 4 2 1 2 10 9 1 8 10 8 6 4 9 10 5 3 7 2 5 10 2 6 7 4 2 10 10 7 6 10 8 4 10 7 3 3 2 5 4 8 1 2 1 5 8 1 3 6 9 2 2 9 2 5 8 6 7 6 5 1 7 6 3 10 1 4 9 7 6 4 9 6 2 3 1 10 7 6 8 3 5'.split()))) - 1
#     )
# )
