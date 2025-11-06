import cv2
import numpy as np
import os
from pathlib import Path
import re

class AutoCrop:
    """
    Automatically crop product images to remove text, whitespace, and keep only the product/model.
    Uses variance-based content detection similar to the Taobao stitcher.
    """

    def __init__(self):
        self.images = []

    def natural_sort_key(self, s):
        """Sort strings containing numbers naturally"""
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split('([0-9]+)', str(s))]

    def load_images(self, input_dir):
        """Load all images from directory"""
        self.images = []
        image_files = []

        for file in os.listdir(input_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(file)

        image_files.sort(key=self.natural_sort_key)

        for file in image_files:
            path = os.path.join(input_dir, file)
            img = cv2.imread(path)
            if img is not None:
                self.images.append({
                    'filename': file,
                    'image': img,
                    'path': path
                })
                print(f"Loaded: {file} - Shape: {img.shape}")

        print(f"\nTotal images loaded: {len(self.images)}")
        return len(self.images)

    def detect_content_bounds(self, image):
        """
        Detect the bounds of actual content (product/model).
        Removes text/whitespace from all sides.
        Returns (top_y, bottom_y, left_x, right_x) or None if detection fails.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = image.shape[:2]

        # Calculate row variance (vertical analysis)
        row_variances = []
        for y in range(height):
            row = gray[y, :]
            variance = np.var(row)
            row_variances.append(variance)

        # Smooth variance to reduce noise
        from scipy.ndimage import uniform_filter1d
        smoothed_row_variance = uniform_filter1d(row_variances, size=20)

        # Content rows have high variance (>300 for these images)
        variance_threshold = 300
        is_content_row = smoothed_row_variance > variance_threshold

        # Find continuous content regions (exclude small isolated text blocks)
        content_regions = []
        start = None
        min_region_height = height * 0.20  # Region must be at least 20% of image height

        for y, is_content in enumerate(is_content_row):
            if is_content and start is None:
                start = y
            elif not is_content and start is not None:
                if y - start > min_region_height:
                    content_regions.append((start, y))
                start = None

        # Check last region
        if start is not None and height - start > min_region_height:
            content_regions.append((start, height))

        if len(content_regions) == 0:
            return None

        # Use the largest content region (the main product image)
        largest_region = max(content_regions, key=lambda r: r[1] - r[0])
        top_y, bottom_y = largest_region

        # Don't crop horizontally - only remove text from top/bottom
        # Keep full width to preserve image box and aspect ratio
        left_x = 0
        right_x = width - 1

        return (top_y, bottom_y, left_x, right_x)

    def crop_image(self, image, bounds, margin=10):
        """
        Crop image to content bounds, removing whitespace on all sides where detected.
        """
        if bounds is None:
            return image

        top_y, bottom_y, left_x, right_x = bounds
        height, width = image.shape[:2]

        # Add margin but keep within image bounds
        # Use no margin for bottom to avoid leaving whitespace
        top_y = max(0, top_y - margin)
        bottom_y = min(height, bottom_y)  # No bottom margin to avoid whitespace
        left_x = max(0, left_x - margin)
        right_x = min(width, right_x + margin)

        # Crop to detected bounds
        cropped = image[top_y:bottom_y, left_x:right_x].copy()
        return cropped

    def process(self, input_dir, output_dir):
        """
        Process all images: detect content and crop to remove text/whitespace.
        """
        os.makedirs(output_dir, exist_ok=True)

        if self.load_images(input_dir) == 0:
            print("No images found!")
            return []

        print("\n=== AUTO-CROPPING IMAGES ===")

        cropped_paths = []
        for idx, img_data in enumerate(self.images):
            print(f"\nProcessing {img_data['filename']}...")

            # Detect content bounds
            bounds = self.detect_content_bounds(img_data['image'])

            if bounds is None:
                print(f"  ⚠ Could not detect content, skipping...")
                continue

            top_y, bottom_y, left_x, right_x = bounds
            print(f"  Content detected: y={top_y}-{bottom_y}, x={left_x}-{right_x}")

            # Crop to content
            cropped = self.crop_image(img_data['image'], bounds)

            # Save cropped image
            output_filename = f"cropped_{img_data['filename']}"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, cropped)
            cropped_paths.append(output_path)

            print(f"  ✓ Cropped: {img_data['image'].shape} → {cropped.shape}")

        print(f"\n=== PROCESSED {len(cropped_paths)}/{len(self.images)} IMAGES ===")
        return cropped_paths


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python auto_crop.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    cropper = AutoCrop()
    results = cropper.process(input_dir, output_dir)

    print(f"\nDone! Processed {len(results)} images in {output_dir}")
