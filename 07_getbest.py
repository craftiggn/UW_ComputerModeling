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

"""
--- Best Found in chromosome_PC0426.json ---
Score: 1035682.118
Raw Chromosome: 01001000100001001100000000101100000000000000000000001000100010000010000011000000101011101110100010000000110000001100100010001000111000001110000011000100111011010000100010000000110001001110010110101111111110110111111011111010111110001000000011000000111110011110000011001111111111111110101000000000010000000000101000001111110010001100000011111110111111100101000011001001111110001111101011011100100011001111110011111101000010001010000011011010111011001110111011111110111111111111111111101000111001001110101010101010
------------------------------------------
Generating animation for chromosome with score: 1035682.118
Animation saved as best_chromosome_animation.gif
Success! Check your folder for best_chromosome_animation.gif

"""