import cv2
import numpy as np
import os
from pathlib import Path
import re

class SimpleBoxStitcher:
    """
    Simple approach: Detect horizontal bands/regions in each image.
    Each image typically has:
    - Gray header area (MODEL INFORMATION)
    - White box with model picture
    - Gray separator (START EXCEED END)
    - White box with model picture

    Strategy: Find transitions from gray->white->gray to identify boxes.
    """

    def __init__(self):
        self.images = []

    def natural_sort_key(self, s):
        """Sort strings containing numbers naturally"""
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split('([0-9]+)', str(s))]

    def load_images(self, input_dir):
        """Load all images from directory in natural sort order"""
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

    def find_white_regions(self, image):
        """
        Find white box regions by analyzing horizontal brightness AND variance profiles.

        Strategy:
        - Model picture rows have HIGH variance (mix of dark clothing and white background)
        - Separator/header rows have LOW variance (uniform gray or white)

        Returns list of y-ranges: [(start_y, end_y), ...]
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = image.shape[:2]

        # Calculate variance for each row
        # High variance = content (model pictures)
        # Low variance = uniform areas (separators, headers)
        row_variances = []
        for y in range(height):
            row = gray[y, :]
            variance = np.var(row)
            row_variances.append(variance)

        # Smooth the variance to reduce noise
        from scipy.ndimage import uniform_filter1d
        smoothed_variance = uniform_filter1d(row_variances, size=20)

        # Content rows have high variance (>500), uniform rows have low variance (<500)
        variance_threshold = 500
        is_content_row = smoothed_variance > variance_threshold

        # Find continuous content regions
        regions = []
        start = None

        for y, is_content in enumerate(is_content_row):
            if is_content and start is None:
                start = y
            elif not is_content and start is not None:
                # End of content region
                if y - start > height * 0.08:  # Region must be at least 8% of image height
                    regions.append((start, y))
                start = None

        # Don't forget last region
        if start is not None and height - start > height * 0.08:
            regions.append((start, height))

        return regions

    def extract_region(self, image, y_start, y_end):
        """Extract a region from image, with small margin"""
        margin = 5
        y_start = max(0, y_start - margin)
        y_end = min(image.shape[0], y_end + margin)
        return image[y_start:y_end, :].copy()

    def process(self, input_dir, output_dir):
        """
        Main processing:
        1. Load all images
        2. Find white box regions in each image
        3. Stitch: last box from image N + first box from image N+1
        """
        os.makedirs(output_dir, exist_ok=True)

        if self.load_images(input_dir) == 0:
            print("No images found!")
            return []

        print("\n=== FINDING WHITE BOX REGIONS ===")

        # Find regions in all images and store per-image
        # Keep ALL regions because:
        # - First region = bottom part of previous product
        # - Last region = top part of next product
        images_with_regions = []
        for idx, img_data in enumerate(self.images):
            print(f"\nProcessing {img_data['filename']}...")
            regions = self.find_white_regions(img_data['image'])
            print(f"  Found {len(regions)} white box regions")

            if not regions:
                continue

            region_images = []
            for r_idx, (start_y, end_y) in enumerate(regions):
                print(f"    Region {r_idx + 1}: y={start_y} to y={end_y} (height={end_y - start_y}px)")
                region_img = self.extract_region(img_data['image'], start_y, end_y)
                region_images.append(region_img)

            images_with_regions.append({
                'filename': img_data['filename'],
                'regions': region_images
            })

        print(f"\n=== TOTAL IMAGES WITH REGIONS: {len(images_with_regions)} ===")

        if len(images_with_regions) < 2:
            print("Not enough images to create products!")
            return []

        print("\n=== STITCHING PRODUCTS ===")

        # Stitch: LAST region from image[i] + FIRST region from image[i+1]
        products = []
        for i in range(len(images_with_regions) - 1):
            current_img = images_with_regions[i]
            next_img = images_with_regions[i + 1]

            # Get LAST region from current image
            last_region = current_img['regions'][-1]
            # Get FIRST region from next image
            first_region = next_img['regions'][0]

            # Combine vertically
            combined = self.combine_images(last_region, first_region)

            product_num = i + 1
            output_filename = f"product_{product_num}.jpg"
            output_path = os.path.join(output_dir, output_filename)

            cv2.imwrite(output_path, combined)
            products.append(output_path)

            print(f"Product {product_num}: {current_img['filename']} (last region) + {next_img['filename']} (first region)")

        print(f"\n=== CREATED {len(products)} PRODUCTS ===")
        return products

    def combine_images(self, img1, img2):
        """Combine two images vertically"""
        # Make sure widths match
        width = max(img1.shape[1], img2.shape[1])

        if img1.shape[1] != width:
            img1 = cv2.resize(img1, (width, img1.shape[0]))
        if img2.shape[1] != width:
            img2 = cv2.resize(img2, (width, img2.shape[0]))

        # Stack vertically
        combined = np.vstack([img1, img2])
        return combined


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python simple_box_stitcher.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    stitcher = SimpleBoxStitcher()
    products = stitcher.process(input_dir, output_dir)

    print(f"\nDone! Created {len(products)} products in {output_dir}")
