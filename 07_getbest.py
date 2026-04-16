import numpy as np
import chromosome_module as chr

# 1. Fetch the best score and the bitstring from the directory
score, bitstring = chr.get_best_saved_chromosome(directory="chromosomes")

if bitstring is not None:
    # 2. Convert the "0101..." string back into a NumPy array (the "chromosome object")
    # We iterate through the string, convert each char to an int, and wrap it in a numpy array
    best_chromosome_obj = np.array([int(bit) for bit in bitstring], dtype=np.int8)

    # 3. Run the animation using the converted object
    print(f"Generating animation for chromosome with score: {score:.3f}")
    chr.animate(best_chromosome_obj, "best_chromosome_animation")
    
    print("Success! Check your folder for best_chromosome_animation.gif")
else:
    print("Operation cancelled: No chromosome files were found.")