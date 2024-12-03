import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import numpy as np
import matplotlib.pyplot as plt
import time
import joblib
import mlrose_hiive
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn import metrics
from sklearn.metrics import precision_score

np.random.seed(2415)


#----------------------------------------------------------4-Peaks--------------------------------------------------------------------------------------------
length = [40, 70, 100]
rhc_time = []
sa_time = []
ga_time = []
mimic_time = []


#------------------------------Length 40---------------------------------------------------------
#--------------RHC---------------

fitness = mlrose.FourPeaks(t_pct=.5)
problem = mlrose.DiscreteOpt(length=40, fitness_fn=fitness, maximize=True)

start=time.time()
rhc_best_sol, rhc_best_fitness, rhc_fitness_curve = mlrose.random_hill_climb(problem, max_attempts=100, max_iters=100, restarts=0, init_state=None, curve=True, random_state=None)
end=time.time()
rhc_time.append(end-start)

# print(rhc_best_sol)
# print(rhc_best_fitness)
# print(rhc_fitness_curve)

#--------------SA---------------

fitness = mlrose.FourPeaks(t_pct=.5)
problem = mlrose.DiscreteOpt(length=40, fitness_fn=fitness, maximize=True)

start=time.time()
sa_best_sol, sa_best_fitness, sa_fitness_curve = mlrose.simulated_annealing(problem, max_attempts=100, max_iters=100, init_state=None, curve=True, random_state=None)
end=time.time()
sa_time.append(end-start)

# print(sa_best_sol)
# print(sa_best_fitness)
# print(sa_fitness_curve)

#--------------GA---------------

fitness = mlrose.FourPeaks(t_pct=.5)
problem = mlrose.DiscreteOpt(length=40, fitness_fn=fitness, maximize=True)

start=time.time()
ga_best_sol, ga_best_fitness, ga_fitness_curve = mlrose.genetic_alg(problem, pop_size=200, mutation_prob=0.1, max_attempts=100, max_iters=100, curve=True, random_state=None)
end=time.time()
ga_time.append(end-start)

# print(ga_best_sol)
# print(ga_best_fitness)
# print(ga_fitness_curve)

#--------------MIMIC---------------

fitness = mlrose.FourPeaks(t_pct=.5)
problem = mlrose.DiscreteOpt(length=40, fitness_fn=fitness, maximize=True)

start=time.time()
mimic_best_sol, mimic_best_fitness, mimic_fitness_curve = mlrose.mimic(problem, pop_size=200, keep_pct=0.2, max_attempts=100, max_iters=100, curve=True, random_state=None, fast_mimic=False)
end=time.time()
mimic_time.append(end-start)

# print(mimic_best_sol)
# print(mimic_best_fitness)
# print(mimic_fitness_curve)

plt.clf()
plt.plot(rhc_fitness_curve)
plt.plot(sa_fitness_curve)
plt.plot(ga_fitness_curve)
plt.plot(mimic_fitness_curve)
plt.legend(["RHC", "SA", "GA", "MIMIC"])
plt.title("4 Peaks: Iteration - Length: 40")
plt.xlabel("Iterations")
plt.ylabel("Fitness")
plt.savefig("images2/4peaksIter40.png")

#------------------------------Length 70---------------------------------------------------------
#--------------RHC---------------

fitness = mlrose.FourPeaks(t_pct=.5)
problem = mlrose.DiscreteOpt(length=70, fitness_fn=fitness, maximize=True)

start=time.time()
rhc_best_sol, rhc_best_fitness, rhc_fitness_curve = mlrose.random_hill_climb(problem, max_attempts=100, max_iters=100, restarts=0, init_state=None, curve=True, random_state=None)
end=time.time()
rhc_time.append(end-start)

# print(rhc_best_sol)
# print(rhc_best_fitness)
# print(rhc_fitness_curve)

#--------------SA---------------

fitness = mlrose.FourPeaks(t_pct=.5)
problem = mlrose.DiscreteOpt(length=70, fitness_fn=fitness, maximize=True)

start=time.time()
sa_best_sol, sa_best_fitness, sa_fitness_curve = mlrose.simulated_annealing(problem, max_attempts=100, max_iters=100, init_state=None, curve=True, random_state=None)
end=time.time()
sa_time.append(end-start)

# print(sa_best_sol)
# print(sa_best_fitness)
# print(sa_fitness_curve)

#--------------GA---------------

fitness = mlrose.FourPeaks(t_pct=.5)
problem = mlrose.DiscreteOpt(length=70, fitness_fn=fitness, maximize=True)

start=time.time()
ga_best_sol, ga_best_fitness, ga_fitness_curve = mlrose.genetic_alg(problem, pop_size=200, mutation_prob=0.1, max_attempts=100, max_iters=100, curve=True, random_state=None)
end=time.time()
ga_time.append(end-start)

# print(ga_best_sol)
# print(ga_best_fitness)
# print(ga_fitness_curve)

#--------------MIMIC---------------

fitness = mlrose.FourPeaks(t_pct=.5)
problem = mlrose.DiscreteOpt(length=70, fitness_fn=fitness, maximize=True)

start=time.time()
mimic_best_sol, mimic_best_fitness, mimic_fitness_curve = mlrose.mimic(problem, pop_size=200, keep_pct=0.2, max_attempts=100, max_iters=100, curve=True, random_state=None, fast_mimic=False)
end=time.time()
mimic_time.append(end-start)

# print(mimic_best_sol)
# print(mimic_best_fitness)
# print(mimic_fitness_curve)

plt.clf()
plt.plot(rhc_fitness_curve)
plt.plot(sa_fitness_curve)
plt.plot(ga_fitness_curve)
plt.plot(mimic_fitness_curve)
plt.legend(["RHC", "SA", "GA", "MIMIC"])
plt.title("4 Peaks: Iteration - Length: 70")
plt.xlabel("Iterations")
plt.ylabel("Fitness")
plt.savefig("images2/4peaksIter70.png")

#------------------------------Length 100---------------------------------------------------------
#--------------RHC---------------

fitness = mlrose.FourPeaks(t_pct=.5)
problem = mlrose.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True)

start=time.time()
rhc_best_sol, rhc_best_fitness, rhc_fitness_curve = mlrose.random_hill_climb(problem, max_attempts=100, max_iters=100, restarts=0, init_state=None, curve=True, random_state=None)
end=time.time()
rhc_time.append(end-start)

# print(rhc_best_sol)
# print(rhc_best_fitness)
# print(rhc_fitness_curve)

#--------------SA---------------

fitness = mlrose.FourPeaks(t_pct=.5)
problem = mlrose.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True)

start=time.time()
sa_best_sol, sa_best_fitness, sa_fitness_curve = mlrose.simulated_annealing(problem, max_attempts=100, max_iters=100, init_state=None, curve=True, random_state=None)
end=time.time()
sa_time.append(end-start)

# print(sa_best_sol)
# print(sa_best_fitness)
# print(sa_fitness_curve)

#--------------GA---------------

fitness = mlrose.FourPeaks(t_pct=.5)
problem = mlrose.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True)

start=time.time()
ga_best_sol, ga_best_fitness, ga_fitness_curve = mlrose.genetic_alg(problem, pop_size=200, mutation_prob=0.1, max_attempts=100, max_iters=100, curve=True, random_state=None)
end=time.time()
ga_time.append(end-start)

# print(ga_best_sol)
# print(ga_best_fitness)
# print(ga_fitness_curve)

#--------------MIMIC---------------

fitness = mlrose.FourPeaks(t_pct=.5)
problem = mlrose.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True)

start=time.time()
mimic_best_sol, mimic_best_fitness, mimic_fitness_curve = mlrose.mimic(problem, pop_size=200, keep_pct=0.2, max_attempts=100, max_iters=100, curve=True, random_state=None, fast_mimic=False)
end=time.time()
mimic_time.append(end-start)

# print(mimic_best_sol)
# print(mimic_best_fitness)
# print(mimic_fitness_curve)

plt.clf()
plt.plot(rhc_fitness_curve)
plt.plot(sa_fitness_curve)
plt.plot(ga_fitness_curve)
plt.plot(mimic_fitness_curve)
plt.legend(["RHC", "SA", "GA", "MIMIC"])
plt.title("4 Peaks: Iteration - Length: 100")
plt.xlabel("Iterations")
plt.ylabel("Fitness")
plt.savefig("images2/4peaksIter100.png")


plt.clf()
plt.plot(length, rhc_time)
plt.plot(length, sa_time)
plt.plot(length, ga_time)
plt.plot(length, mimic_time)
plt.legend(["RHC", "SA", "GA", "MIMIC"])
plt.title("4 Peaks: Length vs Time")
plt.xlabel("Length")
plt.ylabel("Time")
plt.savefig("images2/4peaksTimevsLength.png")

# ---------------------------------------------------------End---------------------------------------------------------------------------------------------------

# -------------------------------------------------------Knapsack--------------------------------------------------------------------------------------------
length = [40, 70, 100]
rhc_time = []
sa_time = []
ga_time = []
mimic_time = []

#------------------------------Length 40---------------------------------------------------------

weights= np.random.uniform(low=0.1, high=1, size=(40,))
values= np.random.uniform(low=1, high=40, size=(40,))

#--------------RHC---------------

fitness = mlrose.Knapsack(weights, values)
problem = mlrose.DiscreteOpt(length=40, fitness_fn=fitness, maximize=True)

start=time.time()
rhc_best_sol, rhc_best_fitness, rhc_fitness_curve = mlrose.random_hill_climb(problem, max_attempts=100, max_iters=100, restarts=0, init_state=None, curve=True, random_state=None)
end=time.time()
rhc_time.append(end-start)

# print(rhc_best_sol)
# print(rhc_best_fitness)
# print(rhc_fitness_curve)

#--------------SA---------------

fitness = mlrose.Knapsack(weights, values)
problem = mlrose.DiscreteOpt(length=40, fitness_fn=fitness, maximize=True)

start=time.time()
sa_best_sol, sa_best_fitness, sa_fitness_curve = mlrose.simulated_annealing(problem, max_attempts=100, max_iters=100, init_state=None, curve=True, random_state=None)
end=time.time()
sa_time.append(end-start)

# print(sa_best_sol)
# print(sa_best_fitness)
# print(sa_fitness_curve)

#--------------GA---------------

fitness = mlrose.Knapsack(weights, values)
problem = mlrose.DiscreteOpt(length=40, fitness_fn=fitness, maximize=True)

start=time.time()
ga_best_sol, ga_best_fitness, ga_fitness_curve = mlrose.genetic_alg(problem, pop_size=200, mutation_prob=0.1, max_attempts=100, max_iters=100, curve=True, random_state=None)
end=time.time()
ga_time.append(end-start)

# print(ga_best_sol)
# print(ga_best_fitness)
# print(ga_fitness_curve)

#--------------MIMIC---------------

fitness = mlrose.Knapsack(weights, values)
problem = mlrose.DiscreteOpt(length=40, fitness_fn=fitness, maximize=True)

start=time.time()
mimic_best_sol, mimic_best_fitness, mimic_fitness_curve = mlrose.mimic(problem, pop_size=200, keep_pct=0.2, max_attempts=100, max_iters=100, curve=True, random_state=None, fast_mimic=False)
end=time.time()
mimic_time.append(end-start)

# print(mimic_best_sol)
# print(mimic_best_fitness)
# print(mimic_fitness_curve)

plt.clf()
plt.plot(rhc_fitness_curve)
plt.plot(sa_fitness_curve)
plt.plot(ga_fitness_curve)
plt.plot(mimic_fitness_curve)
plt.legend(["RHC", "SA", "GA", "MIMIC"])
plt.title("Knapsack: Iteration - Length: 40")
plt.xlabel("Iterations")
plt.ylabel("Fitness")
plt.savefig("images2/KnapsackIter40.png")

#------------------------------Length 70---------------------------------------------------------

weights= np.random.uniform(low=0.1, high=1, size=(70,))
values= np.random.uniform(low=1, high=70, size=(70,))

#--------------RHC---------------

fitness = mlrose.Knapsack(weights, values)
problem = mlrose.DiscreteOpt(length=70, fitness_fn=fitness, maximize=True)

start=time.time()
rhc_best_sol, rhc_best_fitness, rhc_fitness_curve = mlrose.random_hill_climb(problem, max_attempts=100, max_iters=100, restarts=0, init_state=None, curve=True, random_state=None)
end=time.time()
rhc_time.append(end-start)

# print(rhc_best_sol)
# print(rhc_best_fitness)
# print(rhc_fitness_curve)

#--------------SA---------------

fitness = mlrose.Knapsack(weights, values)
problem = mlrose.DiscreteOpt(length=70, fitness_fn=fitness, maximize=True)

start=time.time()
sa_best_sol, sa_best_fitness, sa_fitness_curve = mlrose.simulated_annealing(problem, max_attempts=100, max_iters=100, init_state=None, curve=True, random_state=None)
end=time.time()
sa_time.append(end-start)

# print(sa_best_sol)
# print(sa_best_fitness)
# print(sa_fitness_curve)

#--------------GA---------------

fitness = mlrose.Knapsack(weights, values)
problem = mlrose.DiscreteOpt(length=70, fitness_fn=fitness, maximize=True)

start=time.time()
ga_best_sol, ga_best_fitness, ga_fitness_curve = mlrose.genetic_alg(problem, pop_size=200, mutation_prob=0.1, max_attempts=100, max_iters=100, curve=True, random_state=None)
end=time.time()
ga_time.append(end-start)

# print(ga_best_sol)
# print(ga_best_fitness)
# print(ga_fitness_curve)

#--------------MIMIC---------------

fitness = mlrose.Knapsack(weights, values)
problem = mlrose.DiscreteOpt(length=70, fitness_fn=fitness, maximize=True)

start=time.time()
mimic_best_sol, mimic_best_fitness, mimic_fitness_curve = mlrose.mimic(problem, pop_size=200, keep_pct=0.2, max_attempts=100, max_iters=100, curve=True, random_state=None, fast_mimic=False)
end=time.time()
mimic_time.append(end-start)

# print(mimic_best_sol)
# print(mimic_best_fitness)
# print(mimic_fitness_curve)

plt.clf()
plt.plot(rhc_fitness_curve)
plt.plot(sa_fitness_curve)
plt.plot(ga_fitness_curve)
plt.plot(mimic_fitness_curve)
plt.legend(["RHC", "SA", "GA", "MIMIC"])
plt.title("Knapsack: Iteration - Length: 70")
plt.xlabel("Iterations")
plt.ylabel("Fitness")
plt.savefig("images2/KnapsackIter70.png")

#------------------------------Length 100---------------------------------------------------------

weights= np.random.uniform(low=0.1, high=1, size=(100,))
values= np.random.uniform(low=1, high=100, size=(100,))

#--------------RHC---------------

fitness = mlrose.Knapsack(weights, values)
problem = mlrose.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True)

start=time.time()
rhc_best_sol, rhc_best_fitness, rhc_fitness_curve = mlrose.random_hill_climb(problem, max_attempts=100, max_iters=100, restarts=0, init_state=None, curve=True, random_state=None)
end=time.time()
rhc_time.append(end-start)

# print(rhc_best_sol)
# print(rhc_best_fitness)
# print(rhc_fitness_curve)

#--------------SA---------------

fitness = mlrose.Knapsack(weights, values)
problem = mlrose.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True)

start=time.time()
sa_best_sol, sa_best_fitness, sa_fitness_curve = mlrose.simulated_annealing(problem, max_attempts=100, max_iters=100, init_state=None, curve=True, random_state=None)
end=time.time()
sa_time.append(end-start)

# print(sa_best_sol)
# print(sa_best_fitness)
# print(sa_fitness_curve)

#--------------GA---------------

fitness = mlrose.Knapsack(weights, values)
problem = mlrose.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True)

start=time.time()
ga_best_sol, ga_best_fitness, ga_fitness_curve = mlrose.genetic_alg(problem, pop_size=200, mutation_prob=0.1, max_attempts=100, max_iters=100, curve=True, random_state=None)
end=time.time()
ga_time.append(end-start)

# print(ga_best_sol)
# print(ga_best_fitness)
# print(ga_fitness_curve)

#--------------MIMIC---------------

fitness = mlrose.Knapsack(weights, values)
problem = mlrose.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True)

start=time.time()
mimic_best_sol, mimic_best_fitness, mimic_fitness_curve = mlrose.mimic(problem, pop_size=200, keep_pct=0.2, max_attempts=100, max_iters=100, curve=True, random_state=None, fast_mimic=False)
end=time.time()
mimic_time.append(end-start)

# print(mimic_best_sol)
# print(mimic_best_fitness)
# print(mimic_fitness_curve)

plt.clf()
plt.plot(rhc_fitness_curve)
plt.plot(sa_fitness_curve)
plt.plot(ga_fitness_curve)
plt.plot(mimic_fitness_curve)
plt.legend(["RHC", "SA", "GA", "MIMIC"])
plt.title("Knapsack: Iteration - Length: 100")
plt.xlabel("Iterations")
plt.ylabel("Fitness")
plt.savefig("images2/KnapsackIter100.png")


plt.clf()
plt.plot(length, rhc_time)
plt.plot(length, sa_time)
plt.plot(length, ga_time)
plt.plot(length, mimic_time)
plt.legend(["RHC", "SA", "GA", "MIMIC"])
plt.title("Knapsack: Length vs Time")
plt.xlabel("Length")
plt.ylabel("Time")
plt.savefig("images2/KnapsackTimevsLength.png")

# ---------------------------------------------------------End---------------------------------------------------------------------------------------------------

# --------------------------------------------------------Queens-------------------------------------------------------------------------------------------------
length = [40, 70, 100]
rhc_time = []
sa_time = []
ga_time = []
mimic_time = []


#------------------------------Length 40---------------------------------------------------------
#--------------RHC---------------

fitness = mlrose.Queens()
problem = mlrose.DiscreteOpt(length=40, fitness_fn=fitness, maximize=True)

start=time.time()
rhc_best_sol, rhc_best_fitness, rhc_fitness_curve = mlrose.random_hill_climb(problem, max_attempts=100, max_iters=100, restarts=0, init_state=None, curve=True, random_state=None)
end=time.time()
rhc_time.append(end-start)

# print(rhc_best_sol)
# print(rhc_best_fitness)
# print(rhc_fitness_curve)

#--------------SA---------------

fitness = mlrose.Queens()
problem = mlrose.DiscreteOpt(length=40, fitness_fn=fitness, maximize=True)

start=time.time()
sa_best_sol, sa_best_fitness, sa_fitness_curve = mlrose.simulated_annealing(problem, max_attempts=100, max_iters=100, init_state=None, curve=True, random_state=None)
end=time.time()
sa_time.append(end-start)

# print(sa_best_sol)
# print(sa_best_fitness)
# print(sa_fitness_curve)

#--------------GA---------------

fitness = mlrose.Queens()
problem = mlrose.DiscreteOpt(length=40, fitness_fn=fitness, maximize=True)

start=time.time()
ga_best_sol, ga_best_fitness, ga_fitness_curve = mlrose.genetic_alg(problem, pop_size=200, mutation_prob=0.1, max_attempts=100, max_iters=100, curve=True, random_state=None)
end=time.time()
ga_time.append(end-start)

# print(ga_best_sol)
# print(ga_best_fitness)
# print(ga_fitness_curve)

#--------------MIMIC---------------

fitness = mlrose.Queens()
problem = mlrose.DiscreteOpt(length=40, fitness_fn=fitness, maximize=True)

start=time.time()
mimic_best_sol, mimic_best_fitness, mimic_fitness_curve = mlrose.mimic(problem, pop_size=200, keep_pct=0.2, max_attempts=100, max_iters=100, curve=True, random_state=None, fast_mimic=False)
end=time.time()
mimic_time.append(end-start)

# print(mimic_best_sol)
# print(mimic_best_fitness)
# print(mimic_fitness_curve)

plt.clf()
plt.plot(rhc_fitness_curve)
plt.plot(sa_fitness_curve)
plt.plot(ga_fitness_curve)
plt.plot(mimic_fitness_curve)
plt.legend(["RHC", "SA", "GA", "MIMIC"])
plt.title("Queens: Iteration - Length: 40")
plt.xlabel("Iterations")
plt.ylabel("Fitness")
plt.savefig("images2/QueensIter40.png")

#------------------------------Length 70---------------------------------------------------------
#--------------RHC---------------

fitness = mlrose.Queens()
problem = mlrose.DiscreteOpt(length=70, fitness_fn=fitness, maximize=True)

start=time.time()
rhc_best_sol, rhc_best_fitness, rhc_fitness_curve = mlrose.random_hill_climb(problem, max_attempts=100, max_iters=100, restarts=0, init_state=None, curve=True, random_state=None)
end=time.time()
rhc_time.append(end-start)

# print(rhc_best_sol)
# print(rhc_best_fitness)
# print(rhc_fitness_curve)

#--------------SA---------------

fitness = mlrose.Queens()
problem = mlrose.DiscreteOpt(length=70, fitness_fn=fitness, maximize=True)

start=time.time()
sa_best_sol, sa_best_fitness, sa_fitness_curve = mlrose.simulated_annealing(problem, max_attempts=100, max_iters=100, init_state=None, curve=True, random_state=None)
end=time.time()
sa_time.append(end-start)

# print(sa_best_sol)
# print(sa_best_fitness)
# print(sa_fitness_curve)

#--------------GA---------------

fitness = mlrose.Queens()
problem = mlrose.DiscreteOpt(length=70, fitness_fn=fitness, maximize=True)

start=time.time()
ga_best_sol, ga_best_fitness, ga_fitness_curve = mlrose.genetic_alg(problem, pop_size=200, mutation_prob=0.1, max_attempts=100, max_iters=100, curve=True, random_state=None)
end=time.time()
ga_time.append(end-start)

# print(ga_best_sol)
# print(ga_best_fitness)
# print(ga_fitness_curve)

#--------------MIMIC---------------

fitness = mlrose.Queens()
problem = mlrose.DiscreteOpt(length=70, fitness_fn=fitness, maximize=True)

start=time.time()
mimic_best_sol, mimic_best_fitness, mimic_fitness_curve = mlrose.mimic(problem, pop_size=200, keep_pct=0.2, max_attempts=100, max_iters=100, curve=True, random_state=None, fast_mimic=False)
end=time.time()
mimic_time.append(end-start)

# print(mimic_best_sol)
# print(mimic_best_fitness)
# print(mimic_fitness_curve)

plt.clf()
plt.plot(rhc_fitness_curve)
plt.plot(sa_fitness_curve)
plt.plot(ga_fitness_curve)
plt.plot(mimic_fitness_curve)
plt.legend(["RHC", "SA", "GA", "MIMIC"])
plt.title("Queens: Iteration - Length: 70")
plt.xlabel("Iterations")
plt.ylabel("Fitness")
plt.savefig("images2/QueensIter70.png")

#------------------------------Length 100---------------------------------------------------------
#--------------RHC---------------

fitness = mlrose.Queens()
problem = mlrose.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True)

start=time.time()
rhc_best_sol, rhc_best_fitness, rhc_fitness_curve = mlrose.random_hill_climb(problem, max_attempts=100, max_iters=100, restarts=0, init_state=None, curve=True, random_state=None)
end=time.time()
rhc_time.append(end-start)

# print(rhc_best_sol)
# print(rhc_best_fitness)
# print(rhc_fitness_curve)

#--------------SA---------------

fitness = mlrose.Queens()
problem = mlrose.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True)

start=time.time()
sa_best_sol, sa_best_fitness, sa_fitness_curve = mlrose.simulated_annealing(problem, max_attempts=100, max_iters=100, init_state=None, curve=True, random_state=None)
end=time.time()
sa_time.append(end-start)

# print(sa_best_sol)
# print(sa_best_fitness)
# print(sa_fitness_curve)

#--------------GA---------------

fitness = mlrose.Queens()
problem = mlrose.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True)

start=time.time()
ga_best_sol, ga_best_fitness, ga_fitness_curve = mlrose.genetic_alg(problem, pop_size=200, mutation_prob=0.1, max_attempts=100, max_iters=100, curve=True, random_state=None)
end=time.time()
ga_time.append(end-start)

# print(ga_best_sol)
# print(ga_best_fitness)
# print(ga_fitness_curve)

#--------------MIMIC---------------

fitness = mlrose.Queens()
problem = mlrose.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True)

start=time.time()
mimic_best_sol, mimic_best_fitness, mimic_fitness_curve = mlrose.mimic(problem, pop_size=200, keep_pct=0.2, max_attempts=100, max_iters=100, curve=True, random_state=None, fast_mimic=False)
end=time.time()
mimic_time.append(end-start)

# print(mimic_best_sol)
# print(mimic_best_fitness)
# print(mimic_fitness_curve)

plt.clf()
plt.plot(rhc_fitness_curve)
plt.plot(sa_fitness_curve)
plt.plot(ga_fitness_curve)
plt.plot(mimic_fitness_curve)
plt.legend(["RHC", "SA", "GA", "MIMIC"])
plt.title("Queens: Iteration - Length: 100")
plt.xlabel("Iterations")
plt.ylabel("Fitness")
plt.savefig("images2/QueensIter100.png")


plt.clf()
plt.plot(length, rhc_time)
plt.plot(length, sa_time)
plt.plot(length, ga_time)
plt.plot(length, mimic_time)
plt.legend(["RHC", "SA", "GA", "MIMIC"])
plt.title("Queens: Length vs Time")
plt.xlabel("Length")
plt.ylabel("Time")
plt.savefig("images2/QueensTimevsLength.png")

#---------------------------------------------------------End---------------------------------------------------------------------------------------------------

#---------------------------------------------------------Neural-Networks---------------------------------------------------------------------------------------------------

# load dataset diabetes
pima = pd.read_csv("diabetes.csv", header=0)

#split dataset in features and target variable
feature_cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
X = pima[feature_cols] # Features
y = pima.Outcome # Target variable


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#------------- Learning Curve

epoch = []
trainScoreA = []
valScoreA = []
trainScoreP = []
valScoreP = []

for i in range(50, 500, 10):
    # Split dataset into training set and test set
    epoch.append(i)

    model = Sequential()
    model.add(Dense(15, input_shape=(8,), activation='sigmoid'))
    model.add(Dense(12, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=i, batch_size=10, verbose=0)


    y_pred = (model.predict(X_train) > 0.5).astype(int)
    trainScoreA.append(metrics.accuracy_score(y_train, y_pred))
    trainScoreP.append(precision_score(y_train, y_pred, average='binary'))

    y_pred = (model.predict(X_test) > 0.5).astype(int)
    valScoreA.append(metrics.accuracy_score(y_test, y_pred))
    valScoreP.append(precision_score(y_test, y_pred, average='binary'))

plt.clf()
plt.plot(epoch,trainScoreA)
plt.plot(epoch,valScoreA)
plt.plot(epoch,trainScoreP)
plt.plot(epoch, valScoreP)
plt.legend(["Training - Accuracy", "Validation - Accuracy", "Training - Precision", "Validation - Precision"])
plt.title("Neural Networks: Learning Curve - Pima Indians Diabetes Database")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.savefig("images2/NNEpochPIDD.png")


#------------RHC----------------------------------------

accuracy = []
precision = []

NeuralNetworkM = mlrose_hiive.NeuralNetwork(hidden_nodes = [15,12,1], activation = 'sigmoid', 
                                algorithm = 'random_hill_climb', 
                                max_iters=100, bias = True, is_classifier = True, 
                                learning_rate = 0.1, early_stopping = False, clip_max = 1e+10, 
                                max_attempts = 100,curve=True)

NeuralNetworkM.fit(X_train,y_train)
plt.clf()
plt.plot(NeuralNetworkM.fitness_curve[:,0])
plt.title("RHC Fitness for NN")
plt.xlabel("Iterations")
plt.ylabel("Fitness")
plt.savefig("images2/NNFitRHC.png")

y_pred = NeuralNetworkM.predict(X_train)
accuracy.append(metrics.accuracy_score(y_train, y_pred))
precision.append(precision_score(y_train, y_pred, average='binary'))

y_pred = NeuralNetworkM.predict(X_test)
accuracy.append(metrics.accuracy_score(y_test, y_pred))
precision.append(precision_score(y_test, y_pred, average='binary'))

#------------SA----------------------------------------

NeuralNetworkM = mlrose_hiive.NeuralNetwork(hidden_nodes = [15,12,1], activation = 'sigmoid', 
                                algorithm = 'simulated_annealing', 
                                max_iters=100, bias = True, is_classifier = True, 
                                learning_rate = 0.1, early_stopping = False, clip_max = 1e+10, 
                                max_attempts = 100,curve=True)

NeuralNetworkM.fit(X_train,y_train)
plt.clf()
plt.plot(NeuralNetworkM.fitness_curve[:,0])
plt.title("SA Fitness for NN")
plt.xlabel("Iterations")
plt.ylabel("Fitness")
plt.savefig("images2/NNFitSA.png")

y_pred = NeuralNetworkM.predict(X_train)
accuracy.append(metrics.accuracy_score(y_train, y_pred))
precision.append(precision_score(y_train, y_pred, average='binary'))

y_pred = NeuralNetworkM.predict(X_test)
accuracy.append(metrics.accuracy_score(y_test, y_pred))
precision.append(precision_score(y_test, y_pred, average='binary'))

#------------GA----------------------------------------

NeuralNetworkM = mlrose_hiive.NeuralNetwork(hidden_nodes = [15,12,1], activation = 'sigmoid', 
                                algorithm = 'genetic_alg', 
                                max_iters=100, bias = True, is_classifier = True, 
                                learning_rate = 0.1, early_stopping = False, clip_max = 1e+10, 
                                max_attempts = 100,curve=True)

NeuralNetworkM.fit(X_train,y_train)
plt.clf()
plt.plot(NeuralNetworkM.fitness_curve[:,0])
plt.title("GA Fitness for NN")
plt.xlabel("Iterations")
plt.ylabel("Fitness")
plt.savefig("images2/NNFitGA.png")

y_pred = NeuralNetworkM.predict(X_train)
accuracy.append(metrics.accuracy_score(y_train, y_pred))
precision.append(precision_score(y_train, y_pred, average='binary'))

y_pred = NeuralNetworkM.predict(X_test)
accuracy.append(metrics.accuracy_score(y_test, y_pred))
precision.append(precision_score(y_test, y_pred, average='binary'))



Names = ["Training Accuracy - RHC", "Test Accuracy - RHC",
         "Training Accuracy - SA", "Test Accuracy - SA",
         "Training Accuracy - GA", "Test Accuracy - GA"]

plt.clf()
plt.figure(figsize=(20, 10))
plt.xticks(rotation=45, ha='right')
plt.bar(Names, accuracy)
plt.title("Traing and Testing Accuracy - Pima Indians Diabetes Database")
plt.xlabel("Test Ran - Algorithm")
plt.ylabel("Score")
plt.savefig("images2/NNAccuracy.png", bbox_inches="tight")


Names = ["Training Precision - RHC", "Test Precision - RHC",
         "Training Precision - SA", "Test Precision - SA",
         "Training Precision - GA", "Test Precision - GA"]

plt.clf()
plt.figure(figsize=(20, 10))
plt.xticks(rotation=45, ha='right')
plt.bar(Names, precision)
plt.title("Traing and Testing Precision - Pima Indians Diabetes Database")
plt.xlabel("Test Ran - Algorithm")
plt.ylabel("Score")
plt.savefig("images2/NNPrecision.png", bbox_inches="tight")
"""
-------------------------------------------Sources----------------------------------------------
Dataset
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

Code
https://mlrose.readthedocs.io/en/stable/source/fitness.html
https://mlrose.readthedocs.io/en/stable/source/algorithms.html
https://mlrose.readthedocs.io/en/stable/source/opt_probs.html
https://github.com/hiive/mlrose/blob/master/mlrose_hiive/neural/neural_network.py
"""