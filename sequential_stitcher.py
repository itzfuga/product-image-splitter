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

    def detect_separator_position(self, image):
        """Detect the Y position of 'START EXCEED END' separator"""
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        print(f"  Detecting separator in image ({width}x{height})...")

        # Scan for separator band - look for horizontal white/light gray area
        # Skip the top 20% to avoid detecting headers like "MODEL INFORMATION"
        start_y = int(height * 0.2)
        scan_step = 10
        separator_height = 80

        for y in range(start_y, height - separator_height, scan_step):
            strip = gray[y:y+separator_height, :]
            mean_val = np.mean(strip)
            std_val = np.std(strip)

            # Separator: mostly white/light gray (200-250) with some text variation (10-80)
            # Make it more strict - require higher brightness for white separators
            if 210 < mean_val < 250 and 10 < std_val < 80:
                # Check for horizontal uniformity (rows should be similar)
                row_means = np.mean(strip, axis=1)
                row_uniformity = np.std(row_means)

                # Check central region has high brightness (white separator characteristic)
                central_region = strip[20:60, int(width*0.3):int(width*0.7)]
                central_brightness = np.mean(central_region)

                if row_uniformity < 30 and central_brightness > 220:
                    print(f"    Found separator at y={y}")
                    return y + (separator_height // 2)  # Return middle of separator

        print(f"    No separator found")
        return None

    def split_at_separator(self, image, separator_y):
        """Split image into top and bottom parts at separator"""
        height, width = image.shape[:2]

        if separator_y is None:
            # No separator - return whole image
            return [{'image': self.auto_crop(image), 'type': 'full'}]

        # Split into top and bottom
        # Use larger margin to ensure separator text is completely removed
        margin = 100  # Skip separator area (increased from 40 to 100)
        top_part = image[0:separator_y-margin, :]
        bottom_part = image[separator_y+margin:height, :]

        parts = []

        if top_part.shape[0] > 100:  # Must be substantial
            parts.append({
                'image': self.auto_crop(top_part),
                'type': 'top'
            })
            print(f"    Top part: {top_part.shape[1]}x{top_part.shape[0]}")

        if bottom_part.shape[0] > 100:  # Must be substantial
            parts.append({
                'image': self.auto_crop(bottom_part),
                'type': 'bottom'
            })
            print(f"    Bottom part: {bottom_part.shape[1]}x{bottom_part.shape[0]}")

        return parts

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
        """Process images and split at separators"""
        all_parts = []

        for i, img_data in enumerate(images):
            print(f"\nProcessing image {i+1}/{len(images)}: {img_data['name']}")
            image = img_data['image']

            # Remove header from first image
            if i == 0:
                image = self.remove_header(image)
                print("  Removed header from first image")

            # Detect separator position
            separator_y = self.detect_separator_position(image)

            # Split at separator
            parts = self.split_at_separator(image, separator_y)

            # Store with source info
            for part in parts:
                all_parts.append({
                    'image': part['image'],
                    'type': part['type'],
                    'source_image': img_data['name'],
                    'source_index': i
                })

        return all_parts

    def combine_sequential_model_pictures(self, parts):
        """
        Combine parts: Part N + Part N+1 = Product

        IMPORTANT: Only combine parts from DIFFERENT images!
        Skip combinations where both parts are from the same image.

        Structure:
        - Product 1 = original 1 bottom + original 2 top (DIFFERENT images)
        - Product 2 = original 2 bottom + original 3 top (DIFFERENT images)
        - Skip: original 1 top + original 1 bottom (SAME image - not a product)
        """
        products = []
        product_id = 1

        print(f"\n=== Combining {len(parts)} parts into products ===")
        print("Logic: Only combine parts from DIFFERENT source images")

        # Combine pairs: part[i] + part[i+1], but ONLY if from different images
        i = 0
        while i < len(parts) - 1:
            current_part = parts[i]
            next_part = parts[i + 1]

            # SKIP if both parts are from the same source image
            if current_part['source_index'] == next_part['source_index']:
                print(f"\nSkipping: {current_part['source_image']} ({current_part['type']}) + {next_part['source_image']} ({next_part['type']}) - same image, not a product")
                i += 1
                continue

            print(f"\nProduct {product_id}:")
            print(f"  Top: {current_part['source_image']} ({current_part['type']} part)")
            print(f"  Bottom: {next_part['source_image']} ({next_part['type']} part)")

            img_top = current_part['image']
            img_bottom = next_part['image']

            # Ensure same width
            h1, w1 = img_top.shape[:2]
            h2, w2 = img_bottom.shape[:2]
            target_width = max(w1, w2)

            if w1 != target_width:
                img_top = cv2.resize(img_top, (target_width, h1))
            if w2 != target_width:
                img_bottom = cv2.resize(img_bottom, (target_width, h2))

            # Combine vertically: top first, then bottom
            combined = np.vstack([img_top, img_bottom])

            # Save (no additional cropping - already cropped)
            filename = f"product_{product_id}.jpg"
            output_path = self.result_dir / filename
            cv2.imwrite(str(output_path), combined)

            products.append({
                'product_id': product_id,
                'filename': filename,
                'path': str(output_path),
                'source_top': current_part['source_image'],
                'source_bottom': next_part['source_image'],
                'dimensions': f"{combined.shape[1]}x{combined.shape[0]}"
            })

            print(f"  Saved: {filename} ({combined.shape[1]}x{combined.shape[0]})")
            product_id += 1
            i += 1  # Move to next pair (overlapping)

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
