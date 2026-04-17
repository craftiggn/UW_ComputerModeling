import sys
import select
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import chromosome_module as chr
from itertools import cycle

# ------------------------------
# GLOBAL VARIABLES

# normal evaluation constants
SIZE = 32
STEPS = 100
REPS = 3

# final evaluation constatnt
FINAL_SIZE = 256
FINAL_STEPS = 256 * 2
FINAL_REPS = 3

# evolution constants
MUTATION_PROBABILITY = 0.005
EVOLUTION_LENGTH = 1024
POPULATION_SIZE = 100
# ------------------------------

def chromosome_key(c):
    return tuple(c)

def add_unique(next_pop, seen, generator_fn):
    while True:
        candidate = generator_fn()
        key = chromosome_key(candidate)
        if key not in seen:
            seen.add(key)
            next_pop.append(candidate)
            return

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
        # if scores[best_idx] > best_score:
        #     best_score = scores[best_idx]
        #     best_chromosome = pop[best_idx].copy()

        history.append(scores[best_idx])

        # --- next generation ---
        next_pop = []
        seen = set()

        # === 1. ELITISM (best individuals cloned perfectly) ===
        best_indices = np.argsort(scores)[-perf_clone_count:]
        for i in best_indices:
            add_unique(next_pop, seen, lambda i=i: chr.clone_perfect(pop[i]))

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
            add_unique(next_pop, seen, lambda idx=parent_idx: chr.clone_mutate(pop[idx]))

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
            add_unique(
                next_pop,
                seen,
                lambda p1=p1_idx, p2=p2_idx: chr.procreate(pop[p1], pop[p2])
            )

        print(f"Gen {gen:04d}/{generations:04d}: best = {scores[best_idx]:.3f},\tsurvivors = {num_survivors}/{population}")

        # --- update population ---
        pop = next_pop
        assert(len(pop) == population)

    return pop, history

def run_evolution(min_survival, max_survival, period_count, clone_to_procreation, mutation_probability_multiplyer=1.0):
    """
    This function will run the whole process.
    """
    period_length = (EVOLUTION_LENGTH / (2 * period_count + 1)) * 2
    
    # --- THE FIX ---
    # We take the base MUTATION_PROBABILITY defined at the top of your main script,
    # multiply it, and directly inject it into the chromosome_module (chr).
    # This ensures chr.clone_mutate actually uses the updated rate.
    chr.MUTATION_PROBABILITY = MUTATION_PROBABILITY * mutation_probability_multiplyer 
    
    survival_rate = cos_survival((min_survival, max_survival), EVOLUTION_LENGTH, period_length)
    final_pop, hist = advanced_evolution(POPULATION_SIZE, EVOLUTION_LENGTH, survival_rate, clone_to_procreation)

    scores = np.array([
            chr.evaluate_chromosome(c, FINAL_REPS, FINAL_SIZE, FINAL_STEPS)
            for c in final_pop
        ])
    
    ranking = np.argsort(scores)
    
    # np.argsort is ascending, so the best score is at the end [-1]
    best_chromosome = final_pop[ranking[-1]] 
    best_score = scores[ranking[-1]]

    # --- SAVE TO DIRECTORY LOGIC ---
    save_dir = "chromosomes"
    os.makedirs(save_dir, exist_ok=True)
        
    clean_chromosome_str = "".join(best_chromosome.astype(str))
    
    written = False
    attempt_idx = 1
    
    # We loop until we successfully claim a unique filename
    while not written:
        # NOTE change here PC/LT to avoid collision later
        filename = f"chromosome_LT_{attempt_idx:04d}.json"
        filepath = os.path.join(save_dir, filename)
        
        try:
            # 'x' mode fails if the file already exists, making it "atomic"
            with open(filepath, 'x') as f:
                json.dump([float(best_score), clean_chromosome_str], f, indent=4)
            written = True
        except FileExistsError:
            # If someone else grabbed this index, increment and try again
            attempt_idx += 1

    print(f"Saved best chromosome (Score: {best_score:.3f}) to {filename}")
    return best_chromosome, best_score

# run_evolution(10, 50, 10, 0.2)
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

def main():
    # PARAMETER LOOKUP:
    # 0: min_survival (int)    -> Minimum survivors per generation
    # 1: max_survival (int)    -> Maximum survivors per generation
    # 2: period_count (int)    -> Number of survival waves over total evolution length
    # 3: clone_to_proc (float) -> Ratio of clones to sexual children (0.0 to 1.0)
    # 4: mut_mult (float)      -> Multiplier for the base MUTATION_PROBABILITY
    param_list = [
        (10, 50, 2, 0.2, 1.0),
        (10, 50, 4, 0.2, 1.0),
        (10, 50, 8, 0.2, 1.0),
        (10, 50, 16, 0.2, 1.0),
        (10, 50, 32, 0.2, 1.0),
        (10, 50, 64, 0.2, 1.0),

        (10, 50, 8, 0.2, 1.0),
        (10, 50, 8, 0.4, 1.0),
        (10, 50, 8, 0.8, 1.0),

        (5, 50, 8, 0.2, 1.0),
        (15, 50, 8, 0.2, 1.0),
        (30, 60, 8, 0.2, 1.0),
        (45, 50, 8, 0.2, 1.0), 
        
        (10, 50, 16, 0.2, 1.0),
        (10, 50, 16, 0.2, 2.0),
        (10, 50, 16, 0.2, 4.0),
        (10, 50, 16, 0.2, 8.0),
    ]
    param_generator = cycle(param_list)
    # NOTE change here for lower cpu utilisation
    max_workers = max(1, os.cpu_count() - 4)
    # max_workers = 8
    
    print(f"--- Linux Evolution Engine Started ---")
    print(f"Parallel Workers: {max_workers} | Type 'stop' + Enter to exit.\n")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = set()
        keep_running = True
        
        try:
            while keep_running or futures:
                # 1. Fill the queue
                while len(futures) < max_workers and keep_running:
                    params = next(param_generator)
                    futures.add(executor.submit(run_evolution, *params))

                # 2. Non-blocking wait for at least one to finish
                # We wait 0.5s to give the CPU room to breathe
                if futures:
                    done, futures = wait(futures, timeout=0.5, return_when=FIRST_COMPLETED)
                    
                    for f in done:
                        try:
                            f.result() # Catch internal errors if any
                        except Exception as e:
                            print(f"Evolution process failed: {e}")

                # 3. Linux-specific Stop Check
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    line = sys.stdin.readline().strip().lower()
                    if line == "stop":
                        print("\n[STOP SIGNAL] Finishing current runs...")
                        keep_running = False

        except KeyboardInterrupt:
            print("\n[FORCED EXIT]")
            executor.shutdown(wait=False, cancel_futures=True)
            sys.exit(0)

    print("\nAll tasks finished.")


if __name__ == "__main__":
    # Multiprocessing requires the main guard on Windows
    main()