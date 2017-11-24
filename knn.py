from collections import Counter

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


colours = ("r", "g", "y")
X = []


iris = datasets.load_iris()
iris_data = iris.data
iris_labels = iris.target


np.random.seed(42)
indices = np.random.permutation(len(iris_data))
n_training_samples = 12

learnset_data = iris_data[indices[:-n_training_samples]]
learnset_labels = iris_labels[indices[:-n_training_samples]]
testset_data = iris_data[indices[-n_training_samples:]]
testset_labels = iris_labels[indices[-n_training_samples:]]

# for iclass in range(3):
#     X.append([[],[],[]])
#     for i in range(len(learnset_data)):
#         if learnset_labels[i] == iclass:
#             X[iclass][0].append(learnset_data[i][0])
#             X[iclass][1].append(learnset_data[i][1])
#             X[iclass][2].append(sum(learnset_data[i][2:]))
# fig = plt.figure()
# ax = fig.add_subplot(111, projection ='3d')

# for iclass in range(3):
#     ax.scatter(X[iclass][0], X[iclass][1], X[iclass][2], c=colours[iclass])
# plt.show()


def distance(instance1, instance2):
    # calculating Euclidean distance
    instance1 = np.array(instance1)
    instance2 = np.array(instance2)
    return np.linalg.norm(instance1 - instance2)

def vote(neighbors):
    class_counter = Counter()
    for neighbor in neighbors:
        class_counter[neighbor[2]] += 1
        winner = class_counter.n
        return class_counter.most_common(1)[0][0]

def vote_prob(neighbors):
    class_counter = Counter()
    for neighbor in neighbors:
        class_counter[neighbor[2]] += 1
        labels, votes = zip(*class_counter.most_common())
        winner = class_counter.most_common(1)[0][0]
        votes4winner = class_counter.most_common(1)[0][1]
        return winner, votes4winner/sum(votes)

def vote_harmonic_weights(neighbors, all_results=True):
    class_counter = Counter()
    number_of_neighbours = len(neighbors)
    for i in range(number_of_neighbours):
        class_counter[neighbors[index][2]] += 1/(index+1)
    labels, votes = zip(*class_counter.most_common())
    winner = class_counter.most_common(1)[0][0]
    votes4winner = class_counter.most_common(1)[0][1]
    if all_results:
        total = sum(class_counter.values(), 0.0)
        for key in class_counter:
             class_counter[key] /= total
        return winner, class_counter.most_common()
    else:
        return winner, votes4winner / sum(votes)


def vote_distance_weights(neighbors, all_results=True):
    class_counter = Counter()
    number_of_neighbours = len(neighbors)
    for i in range(number_of_neighbours):
        dist = neighbors[index][1]
        class_counter[neighbors[index][2]] += 1/(dist**2+1)
    labels, votes = zip(*class_counter.most_common())
    winner = class_counter.most_common(1)[0][0]
    votes4winner = class_counter.most_common(1)[0][1]
    if all_results:
        total = sum(class_counter.values(), 0.0)
        for key in class_counter:
             class_counter[key] /= total
        return winner, class_counter.most_common()
    else:
        return winner, votes4winner / sum(votes)


def get_neighbors(training_set, labels, test_instance, k, distance = distance):
    """
    get_neighbors calculates a list of the k nearest neighbors
    of an instance 'test_instance'.
    it return a list of neighbours contaning 3 tuples,
    neighbours =  [(index, dist, label)]
    index,  is the index from training_set,
    dist, is the distance between the test_instance and the 
    instance training_set[index]
    distance, is the refrence to a distance function used to calculate the distance
    """
    distances = []
    for index in range(len(training_set)):
        dist = distance(test_instance, training_set[index])
        distances.append((training_set[index], dist, labels[index]))
    distances.sort(key=lambda x: x[1])
    neighbors = distances[:k]
    return neighbors

for i in range(n_training_samples):
    """
    learnset_labeles are pre tagged distance from a testset_data
    """
    neighbors = get_neighbors(learnset_data, learnset_labels,testset_data[i], 5, distance=distance)


    
    print(
        "index", i,
        ", vote_prob: ", vote_prob(neighbors),
        ", label: ", testset_labels[i],
        ", data:", testset_data[i]
        )
    