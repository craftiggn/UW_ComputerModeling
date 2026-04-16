import numpy as np
import matplotlib.pyplot as plt
import chromosome_module as chr

# global variables
SIZE = 32
STEPS = 100
EVAL_REP = 3
MUTATION_PROBABILITY = 0.005

# NOTE
# need to make it run with a varied survival rate
# need to make it save best chromosome after each epoch
# make final evaluation 500x500
# vary mutation probability? 

def  sin_generations(min_max, length_total, length_cycle):
    """
    This function will be used to determine the ammount of survivors for each generations.
    """
    min_val, max_val = min_max

    omega = 2 * np.pi / length_cycle
    time = np.arange(length_total)

    sine = np.sin(omega * time)
    generations = (sine + 1) / 2 * (max_val - min_val) + min_val

    return generations.astype(int)


def advanced_evolution(generations, clone_to_regular, perf_clone_count):
    """
    This function will run the genetic algorthm
    """

generations = sin_generations((10,50), 1024, 256)
plt.plot(generations)
plt.show()
