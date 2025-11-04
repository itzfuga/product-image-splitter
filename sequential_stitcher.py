#!/usr/bin/env python3
"""
Sequential Stitcher for Taobao Products
Properly detects "START EXCEED END" separators and stitches sequential segments
"""

import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
import json
import re
from typing import Dict, List, Tuple, Optional


class SequentialStitcher:
    def __init__(self, result_dir, session_id=None):
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(exist_ok=True)
        self.session_id = session_id

    def load_images(self, image_dir):
        """Load all images from directory in chronological order"""
        image_dir = Path(image_dir)
        images = []

        supported_formats = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.avif']

        def natural_sort_key(path):
            s = path.name
            return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

        all_files = []
        for file_path in image_dir.glob('*'):
            if file_path.suffix.lower() in supported_formats:
                all_files.append(file_path)

        sorted_files = sorted(all_files, key=natural_sort_key)

        for file_path in sorted_files:
            try:
                img = cv2.imread(str(file_path))
                if img is not None:
                    images.append({
                        'path': str(file_path),
                        'name': file_path.name,
                        'image': img,
                        'index': len(images)
                    })
                    print(f"Loaded {len(images)}: {file_path.name}")
            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")

        print(f"Total loaded: {len(images)} images")
        return images

    def detect_white_rectangles(self, image):
        """Detect white rectangles containing model pictures"""
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        print(f"  Detecting white rectangles in image ({width}x{height})...")

        # Threshold to find white/light regions
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

        # Invert so white becomes black and content becomes white
        binary_inv = cv2.bitwise_not(binary)

        # Find contours
        contours, _ = cv2.findContours(binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rectangles = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by size - must be substantial (model picture)
            # At least 400px wide and 400px tall
            if w > 400 and h > 400:
                rectangles.append({
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'area': w * h
                })

        # Sort rectangles by Y position (top to bottom)
        rectangles.sort(key=lambda r: r['y'])

        print(f"  Found {len(rectangles)} model picture rectangles")
        for i, rect in enumerate(rectangles):
            print(f"    Rectangle {i+1}: pos=({rect['x']}, {rect['y']}) size={rect['width']}x{rect['height']}")

        return rectangles

    def extract_model_pictures(self, image, rectangles):
        """Extract model pictures from rectangles"""
        model_pictures = []

        for i, rect in enumerate(rectangles):
            # Extract rectangle region
            x, y, w, h = rect['x'], rect['y'], rect['width'], rect['height']
            model_img = image[y:y+h, x:x+w]

            # Auto-crop to remove any remaining white space
            cropped = self.auto_crop(model_img)

            model_pictures.append({
                'image': cropped,
                'original_rect': rect,
                'index': i
            })
            print(f"    Extracted model picture {i+1}: {cropped.shape[1]}x{cropped.shape[0]}")

        return model_pictures

    def remove_header(self, image):
        """Remove header text and borders from top of image"""
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Check top 15% for header
        scan_height = min(int(height * 0.15), 250)

        for y in range(0, scan_height, 5):
            # Check for horizontal line
            row = gray[y, :]
            dark_pixels = np.sum(row < 100)

            # If >50% of pixels are dark, this is likely a border line
            if dark_pixels / width > 0.5:
                # Return image starting 10px below this line
                return image[y+10:, :]

        # No header found
        return image

    def auto_crop(self, image):
        """Auto-crop to remove excess white space"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find content bounds (non-white pixels)
        content_mask = gray < 250

        # Find bounding box
        coords = cv2.findNonZero(content_mask.astype(np.uint8))
        if coords is None:
            return image

        x, y, w, h = cv2.boundingRect(coords)

        # Add small padding
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + padding * 2)
        h = min(image.shape[0] - y, h + padding * 2)

        return image[y:y+h, x:x+w]

    def process_images_chronologically(self, images):
        """Process images and extract all model pictures"""
        all_model_pictures = []

        for i, img_data in enumerate(images):
            print(f"\nProcessing image {i+1}/{len(images)}: {img_data['name']}")
            image = img_data['image']

            # Remove header from first image
            if i == 0:
                image = self.remove_header(image)
                print("  Removed header from first image")

            # Detect white rectangles containing model pictures
            rectangles = self.detect_white_rectangles(image)

            # Extract model pictures from rectangles
            model_pictures = self.extract_model_pictures(image, rectangles)

            # Store with source info
            for pic in model_pictures:
                all_model_pictures.append({
                    'image': pic['image'],
                    'source_image': img_data['name'],
                    'source_index': i,
                    'rect_index': pic['index']
                })

        return all_model_pictures

    def combine_sequential_model_pictures(self, model_pictures):
        """Combine: bottom of model picture N + top of model picture N+1 = Product"""
        products = []
        product_id = 1

        print(f"\n=== Combining {len(model_pictures)} model pictures into products ===")

        # Combine pairs: pic[i] bottom + pic[i+1] top
        for i in range(len(model_pictures) - 1):
            current_pic = model_pictures[i]
            next_pic = model_pictures[i + 1]

            print(f"\nProduct {product_id}:")
            print(f"  Bottom: {current_pic['source_image']} (model {current_pic['rect_index']+1})")
            print(f"  Top: {next_pic['source_image']} (model {next_pic['rect_index']+1})")

            img1 = current_pic['image']
            img2 = next_pic['image']

            # Ensure same width
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            target_width = max(w1, w2)

            if w1 != target_width:
                img1 = cv2.resize(img1, (target_width, h1))
            if w2 != target_width:
                img2 = cv2.resize(img2, (target_width, h2))

            # Combine vertically
            combined = np.vstack([img1, img2])

            # Final crop to remove any remaining white space
            final = self.auto_crop(combined)

            # Save
            filename = f"product_{product_id}.jpg"
            output_path = self.result_dir / filename
            cv2.imwrite(str(output_path), final)

            products.append({
                'product_id': product_id,
                'filename': filename,
                'path': str(output_path),
                'source_bottom': current_pic['source_image'],
                'source_top': next_pic['source_image'],
                'dimensions': f"{final.shape[1]}x{final.shape[0]}"
            })

            print(f"  Saved: {filename} ({final.shape[1]}x{final.shape[0]})")
            product_id += 1

        return products


def main():
    """Test the Sequential Stitcher"""
    import sys

    if len(sys.argv) != 3:
        print("Usage: python sequential_stitcher.py <input_dir> <output_dir>")
        return

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    stitcher = SequentialStitcher(output_dir)

    print("Loading images...")
    images = stitcher.load_images(input_dir)

    if not images:
        print("No images found!")
        return

    print("\nProcessing images...")
    model_pictures = stitcher.process_images_chronologically(images)

    print("\nCombining model pictures...")
    products = stitcher.combine_sequential_model_pictures(model_pictures)

    print(f"\n=== Complete! Created {len(products)} products ===")
    for p in products:
        print(f"  {p['filename']}: {p['dimensions']}")


if __name__ == '__main__':
    main()
