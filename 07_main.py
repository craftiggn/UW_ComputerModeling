import numpy as np
import matplotlib.pyplot as plt
import chromosome_module as chr
from itertools import cycle

# global variables
SIZE = 32
STEPS = 100
EVAL_REP = 3
MUTATION_PROBABILITY = 0.005

# NOTE
# need to make it run with a varied survival rate
# need to make it save best chromosome after each epoch
# make final evaluation 500x500
# vary mutation probability (decay)?
# reduce the evaluation time? as chromosomes get better? (to increase efficiency)

def  cos_survival(min_max, length_total, length_cycle):
    """
    This function will be used to determine the ammount of survivors for each generations.
    """
    min_val, max_val = min_max

    omega = 2 * np.pi / length_cycle
    time = np.arange(length_total)

    sine = np.cos(omega * time)
    generations = (sine + 1) / 2 * (max_val - min_val) + min_val

    return generations.astype(int)


def advanced_evolution(population, generations, survival_rate, clone_to_procreation, perf_clone_count=3):
    """
    This function will run the genetic algorthm, following the
    survival function to determine how many member survive each generation.
    Each generation will reproduce via the cloning or procreation.
    Last generation will be evaluated via the 500x500 matrix.
    Some member will be perfect clones from the last generation.
    """
    assert(clone_to_procreation <= 1.0 and clone_to_procreation >= 0.0)
    history = []
    speedrun_history = []
    pop = [chr.gen_chromosome() for _ in range(population)]

    best_chromosome = None # might be useless
    best_score = -np.inf

    eval_rep = 3
    eval_size = 64
    steps = 64

    for gen in range(generations):
        # --- reproduction counts ---
        num_survivors = survival_rate[gen]
        num_offspring = population - perf_clone_count
        num_clones = int(num_offspring * clone_to_procreation)
        num_children = num_offspring - num_clones

        # --- evaluate population ---
        scores = np.array([
            chr.evaluate_chromosome(c, eval_rep, eval_size, steps)
            for c in pop
        ])



        # --- select survivors ---
        survivor_idx = np.argsort(scores)[-num_survivors:]
        survivors = [pop[i] for i in survivor_idx]

        # --- track best ---
        best_idx = np.argmax(scores)
        if scores[best_idx] > best_score:
            best_score = scores[best_idx]
            best_chromosome = pop[best_idx].copy()

        history.append(best_score)

        # --- next generation ---
        next_pop = []

        # === 1. ELITISM (best individuals cloned perfectly) ===
        best_indices = np.argsort(scores)[-perf_clone_count:]
        for i in best_indices:
            next_pop.append(chr.clone_perfect(pop[i]))

        # === 2. CLONE + MUTATE ===
        # ensure every survivor participates at least once
        clone_parents = list(survivor_idx)

        remaining_clones = num_clones - len(clone_parents)
        if remaining_clones > 0:
            extra = np.random.choice(survivor_idx, size=remaining_clones, replace=True)
            clone_parents.extend(extra)

        clone_parents = np.array(clone_parents)
        np.random.shuffle(clone_parents)

        for i in range(num_clones):
            parent_idx = clone_parents[i]
            next_pop.append(chr.clone_mutate(pop[parent_idx]))

        # === 3. PROCREATION (UNBIASED RANDOM PAIRING) ===
        total_parents_needed = 2 * num_children

        # ensure every survivor appears at least once
        parent_pool = list(survivor_idx)

        remaining = total_parents_needed - len(parent_pool)
        if remaining > 0:
            extra = np.random.choice(survivor_idx, size=remaining, replace=True)
            parent_pool.extend(extra)

        parent_pool = np.array(parent_pool)
        np.random.shuffle(parent_pool)

        for i in range(num_children):
            p1_idx = parent_pool[2 * i]
            p2_idx = parent_pool[2 * i + 1]

            child = chr.procreate(pop[p1_idx], pop[p2_idx])
            next_pop.append(child)

        if gen % 32 == 0:
            speedrun_scores = np.array([
                chr.evaluate_chromosome_speedrun(c, eval_rep*2, eval_size*2, steps*2)
                for c in pop
            ])
            speedrun_history.append(np.max(speedrun_scores))
            print(f"Gen {gen:04d}: best = {best_score:.3f},\tsurvivors = {num_survivors}/{population}\t spedrun score = {np.max(speedrun_scores)}")
        else:
            print(f"Gen {gen:04d}: best = {best_score:.3f},\tsurvivors = {num_survivors}/{population}")

        # --- update population ---
        pop = next_pop
        assert(len(pop) == population)

    # save to json in a given directory:
    # score, chromosome, evolution parameters

    return best_chromosome, history, speedrun_history




survival_rate = cos_survival((10,50), 1024, 64)
# plt.plot(survival_rate)
# plt.show()
print("min:", survival_rate.min())
print("max:", survival_rate.max())
print("mean:", survival_rate.mean())

best, hist, speed_hist = advanced_evolution(100, 2048, survival_rate, 0.3)
chr.animate(best, "_test")

plt.plot(hist)
plt.show()

plt.plot(speed_hist)
plt.show()

print(best)