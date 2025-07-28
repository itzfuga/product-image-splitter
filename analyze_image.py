#!/usr/bin/env python3
import cv2
import numpy as np
import sys

# Load first image to analyze
img = cv2.imread('1.jpg')
if img is None:
    print("Could not load 1.jpg")
    sys.exit(1)

height, width = img.shape[:2]
print(f"Image dimensions: {width}x{height}")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Analyze rows to find white areas
print("\nAnalyzing white areas in image:")
print("Row | White % | Type")
print("-" * 30)

white_regions = []
in_white = False
start_y = 0

for y in range(0, height, 10):  # Check every 10 pixels
    row = gray[y, :]
    white_pixels = np.sum(row > 240)
    white_percent = (white_pixels / width) * 100
    
    if white_percent > 80:
        if not in_white:
            in_white = True
            start_y = y
    else:
        if in_white:
            end_y = y
            if end_y - start_y > 20:  # Significant white region
                white_regions.append((start_y, end_y))
                print(f"{start_y:4d} - {end_y:4d} | WHITE REGION ({end_y-start_y}px high)")
            in_white = False
    
    if y % 100 == 0:  # Print every 100 pixels
        region_type = "WHITE" if white_percent > 80 else "CONTENT"
        print(f"{y:4d} | {white_percent:5.1f}% | {region_type}")

print(f"\nFound {len(white_regions)} significant white regions:")
for i, (start, end) in enumerate(white_regions):
    mid = (start + end) // 2
    print(f"Region {i+1}: y={start}-{end} (middle at y={mid})")

# Check what's actually in the image at different positions
print("\nImage content analysis:")
print(f"Top 25% (0-{height//4}): Check what's here")
print(f"Middle (around {height//2}): Check what's here")
print(f"Bottom 25% ({3*height//4}-{height}): Check what's here")