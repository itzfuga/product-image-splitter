#!/usr/bin/env python3
"""Test script for 1-9.jpg images only"""

import shutil
import os
from puzzle_reconstructor import PuzzleReconstructor

# Create a temp directory with only 1-9.jpg
temp_dir = "temp_1_9"
os.makedirs(temp_dir, exist_ok=True)

# Copy only the 1-9.jpg files
for i in range(1, 10):
    src = f"results/{i}.jpg" 
    dst = f"{temp_dir}/{i}.jpg"
    if os.path.exists(src):
        shutil.copy2(src, dst)

# Run the reconstructor
output_dir = "test_1_9_output"
reconstructor = PuzzleReconstructor(output_dir)

print("Loading 1-9.jpg images only...")
images = reconstructor.load_images(temp_dir)
print(f"Loaded {len(images)} images")

print("Building image chains...")
chains = reconstructor.build_image_chains(images, [])

print("Processing chains...")
results = reconstructor.process_chains(images, chains)

print(f"\nResults for 1-9.jpg:")
for i, result in enumerate(results):
    if result['success']:
        print(f"Product {i+1}: {' + '.join(result['source_images'])}")
    else:
        print(f"Product {i+1}: ERROR - {result.get('error', 'Unknown')}")

# Cleanup
shutil.rmtree(temp_dir)