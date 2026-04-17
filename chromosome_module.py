import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import random
import os
import json
import glob

MUTATION_PROBABILITY = 0.005

def calc_neighborhood_deprecated(matrix):
    # from ____ side:
    left = np.roll(matrix, 1, 1) 
    right = np.roll(matrix, -1, 1)

    top = np.roll(matrix, 1, 0)
    top_left = np.roll(top, 1, 1)
    top_right = np.roll(top, -1, 1)

    bottom = np.roll(matrix, -1, 0)
    bottom_left = np.roll(bottom, 1, 1) 
    bottom_right = np.roll(bottom, -1, 1)

    a = [
        top_left, top, top_right,
        left, matrix, right,
        bottom_left, bottom, bottom_right
        ]
    
    powers = 2 ** np.arange(9)

    return sum(neighbor * power for neighbor, power in zip(a, powers))

def calc_neighborhood(matrix):
    return (
        np.roll(matrix, 1, 1)  * 1 +
        np.roll(matrix, -1, 1) * 2 +
        np.roll(matrix, 1, 0)  * 4 +
        np.roll(np.roll(matrix, 1, 0), 1, 1)  * 8 +
        np.roll(np.roll(matrix, 1, 0), -1, 1) * 16 +
        np.roll(matrix, -1, 0) * 32 +
        np.roll(np.roll(matrix, -1, 0), 1, 1) * 64 +
        np.roll(np.roll(matrix, -1, 0), -1, 1) * 128 +
        matrix * 256
    )

def gen_matrix(size):
    return np.random.randint(0, 2, size=(size, size))

def gen_chromosome():
    return np.random.randint(0,2, 512)

def apply_ruleset(neighboorhood, chromosome):
    return (chromosome[neighboorhood]).astype(np.int32)

def calc_score(matrix):
    # Get forward-looking neighbors to match instructions and avoid double-counting
    right = np.roll(matrix, -1, 1)        # (x+1, y)
    bottom = np.roll(matrix, -1, 0)       # (x, y+1)
    
    bottom_right = np.roll(bottom, -1, 1) # (x+1, y+1)
    
    top = np.roll(matrix, 1, 0)
    top_right = np.roll(top, -1, 1)       # (x+1, y-1)

    # "Subtract 3 points if cell (x+1,y) or (x,y+1) is in the same state as (x,y)"
    # Using bitwise OR (|) for the condition
    cond1 = (matrix == right) | (matrix == bottom)

    # "Otherwise..." 
    # add 8 points if cell (x+1, y + 1) is same, subtract 5 if not
    diag1_score = np.where(matrix == bottom_right, 8, -5)
    # likewise add 8 points if cell (x+1,y-1) is same, subtract 5 if not
    diag2_score = np.where(matrix == top_right, 8, -5)

    # Apply the if/else logic across the entire grid
    scores = np.where(cond1, -3, diag1_score + diag2_score)

    return np.sum(scores)

def evaluate_chromosome(chromosome, eval_rep, size, steps):
    total_repetition_scores = []
    
    for _ in range(eval_rep):
        matrix = gen_matrix(size)
        step_scores = []
        
        for step in range(steps):
            neigh = calc_neighborhood(matrix)
            matrix = apply_ruleset(neigh, chromosome)
            
            # Calculate score at EVERY step
            current_score = calc_score(matrix)
            step_scores.append(current_score)
                
        # The score for this run is the average of all steps.
        # Quick convergence = more high-score steps in the list = higher average.
        total_repetition_scores.append(np.mean(step_scores))

    return np.mean(total_repetition_scores)

def evaluate_chromosome_speedrun(chromosome, eval_rep, size, max_steps):
    """
    Evaluates how many steps it takes to reach a stable checkerboard.
    Returns the average steps (lower is better, 0 is perfect).
    """
    # Pre-generate the two possible checkerboard patterns for comparison
    coords = np.indices((size, size))
    checker_a = (coords[0] + coords[1]) % 2
    checker_b = 1 - checker_a
    
    total_steps_to_success = []

    for _ in range(eval_rep):
        matrix = gen_matrix(size)
        stable_count = 0
        first_reached_at = max_steps
        
        for step in range(1, max_steps + 1):
            # Advance the CA
            neigh = calc_neighborhood(matrix)
            matrix = apply_ruleset(neigh, chromosome)
            
            # Check if current state is a checkerboard
            # np.array_equal is fast, but for large matrices np.all(matrix == target) is often better
            is_checker = np.array_equal(matrix, checker_a) or np.array_equal(matrix, checker_b)
            
            if is_checker:
                if stable_count == 0:
                    first_reached_at = step
                stable_count += 1
            else:
                stable_count = 0
                first_reached_at = max_steps
            
            # If it has stayed a checkerboard for 3 consecutive steps, we stop
            if stable_count >= 3:
                break
        
        total_steps_to_success.append(first_reached_at)

    return - np.mean(total_steps_to_success)

def clone_perfect(chromosome):
    return chromosome.copy()

def clone_mutate(chromosome):
    return mutate(chromosome.copy())

def mutate(chromosome):
    mask = np.random.random(chromosome.shape) < MUTATION_PROBABILITY
    return chromosome ^ mask.astype(np.int8)
    
def procreate(parent1, parent2):
    mask = np.random.random(parent1.shape) < 0.5
    child = np.where(mask, parent1, parent2)
    return mutate(child)

def run_evolution(generations, population, survival_rate):
    """
    This function runs the genetic algoritm, after each generation
    the survivors randomly procrete with each other.
    """

    history = []

    # initial population
    pop = [gen_chromosome() for _ in range(population)]

    best_chromosome = None
    best_score = -np.inf

    for gen in range(generations):
        # evaluate the population
        scores = np.array([evaluate_chromosome(c) for c in pop])

        # track the best score
        idx_best = np.argmax(scores)
        if scores[idx_best] > best_score:
            best_score = scores[idx_best]
            best_chromosome = pop[idx_best].copy()

        history.append(best_score)

        # select survivors
        num_survivors = max(2, int(population * survival_rate))
        survivor_idx = np.argsort(scores)[-num_survivors:]
        survivors = [pop[i] for i in survivor_idx]

        # reproduction
        new_pop = []

        # keep best survivor
        new_pop.append(clone_perfect(pop[idx_best]))

        # breeding
        while len(new_pop) < population:
            p1, p2 = random.choice(survivors), random.choice(survivors)
            child = procreate(p1,p2)
            new_pop.append(child)
        
        pop = new_pop

        print(f"Gen {gen}: best={best_score:03f}")
    
    return best_chromosome, history

def animate(chromosome, filename):
    # Use a slightly larger size for better visibility in the GIF
    matrix = gen_matrix(256)
    frames = []

    for i in range(256*3):
        neigh = calc_neighborhood(matrix)
        matrix = apply_ruleset(neigh, chromosome)

        # Convert the 0/1 matrix to a 0-255 uint8 image
        # We repeat the pixels (zoom) so the GIF isn't tiny
        frame = (matrix * 255).astype(np.uint8)
        
        # Optional: Upscale the image so it's not a tiny 64x64 GIF
        # This repeats each pixel 4 times in both directions
        # frame_large = np.repeat(np.repeat(frame, 4, axis=0), 4, axis=1)
        
        frames.append(frame)

    # Save using imageio
    imageio.mimsave(f"{filename}.gif", frames, fps=20)
    print(f"Animation saved as {filename}.gif")

def get_best_saved_chromosome(directory="chromosomes"):
    """
    Searches the given directory for chromosome JSON files and returns 
    the one with the highest evaluated score.
    """
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return None, None

    best_score = -np.inf
    best_chromosome_str = None
    best_file = None

    # Iterate through all chromosome JSON files
    for file_path in glob.glob(os.path.join(directory, "chromosome*.json")):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # data[0] is the score, data[1] is the bitstring
                score = float(data[0])
                chrom_str = data[1]

                if score > best_score:
                    best_score = score
                    best_chromosome_str = chrom_str
                    best_file = file_path
                    
        except (json.JSONDecodeError, IndexError, ValueError) as e:
            print(f"Skipping {file_path} due to read error: {e}")

    if best_file:
        print(f"--- Best Found in {os.path.basename(best_file)} ---")
        print(f"Score: {best_score:.3f}")
        print(f"Raw Chromosome: {best_chromosome_str}")
        print("-" * (len(os.path.basename(best_file)) + 20))
    else:
        print("No valid chromosome files found in the directory.")

    return best_score, best_chromosome_str