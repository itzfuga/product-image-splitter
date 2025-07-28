#!/usr/bin/env python3
"""
Outfit-based Image Grouper
Analyzes fashion photos and groups them by outfit similarity
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import json

class OutfitGrouper:
    def __init__(self, output_dir="output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def load_images(self, image_dir):
        """Load all images from directory"""
        image_dir = Path(image_dir)
        image_files = (list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg")) + 
                      list(image_dir.glob("*.png")) + list(image_dir.glob("*.gif")) + 
                      list(image_dir.glob("*.webp")) + list(image_dir.glob("*.avif")))
        image_files.sort()  # Ensure consistent ordering
        
        images = []
        for img_path in image_files:
            try:
                # Try OpenCV first
                img = cv2.imread(str(img_path))
                
                # If OpenCV fails (e.g., with AVIF), try Pillow
                if img is None:
                    from PIL import Image
                    import numpy as np
                    
                    pil_img = Image.open(img_path)
                    # Convert to RGB if necessary
                    if pil_img.mode != 'RGB':
                        pil_img = pil_img.convert('RGB')
                    # Convert PIL to OpenCV format (RGB to BGR)
                    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                
                if img is not None:
                    images.append({
                        'path': img_path,
                        'image': img,
                        'name': img_path.name
                    })
                    print(f"Loaded: {img_path.name}")
                else:
                    print(f"Failed to load: {img_path}")
                    
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        return images
    
    def segment_person(self, image):
        """Segment person from background using color-based segmentation"""
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for background (light gray/white areas)
        # Background pixels are typically very light
        lower_bg = np.array([0, 0, 200])  # Light areas
        upper_bg = np.array([180, 50, 255])  # Very light areas
        bg_mask = cv2.inRange(hsv, lower_bg, upper_bg)
        
        # Person mask is inverse of background
        person_mask = cv2.bitwise_not(bg_mask)
        
        # Clean up the mask
        kernel = np.ones((5,5), np.uint8)
        person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_CLOSE, kernel)
        person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_OPEN, kernel)
        
        return person_mask
    
    def extract_clothing_region(self, image, person_mask):
        """Extract upper body clothing region"""
        h, w = image.shape[:2]
        
        # Focus on upper body region (torso area where jackets are)
        upper_start = int(h * 0.2)  # Start below head
        upper_end = int(h * 0.7)    # End at waist
        left_margin = int(w * 0.2)
        right_margin = int(w * 0.8)
        
        # Extract clothing region
        clothing_mask = person_mask[upper_start:upper_end, left_margin:right_margin]
        clothing_region = image[upper_start:upper_end, left_margin:right_margin]
        
        return clothing_region, clothing_mask
    
    def extract_features(self, image_data):
        """Extract clothing features from image"""
        image = image_data['image']
        
        # Segment person from background
        person_mask = self.segment_person(image)
        
        # Extract clothing region
        clothing_region, clothing_mask = self.extract_clothing_region(image, person_mask)
        
        # Apply mask to focus only on clothing pixels
        clothing_pixels = clothing_region[clothing_mask > 0]
        
        if len(clothing_pixels) == 0:
            # Fallback: use center region if segmentation fails
            h, w = image.shape[:2]
            center_region = image[int(h*0.2):int(h*0.7), int(w*0.25):int(w*0.75)]
            clothing_pixels = center_region.reshape(-1, 3)
        
        # Extract color features
        features = self.extract_color_features(clothing_pixels)
        
        # Add texture features
        texture_features = self.extract_texture_features(clothing_region, clothing_mask)
        features.update(texture_features)
        
        print(f"Extracted features for {image_data['name']}: "
              f"dark={features['dark_ratio']:.2f}, "
              f"light={features['light_ratio']:.2f}, "
              f"metallic={features['metallic_ratio']:.2f}")
        
        return features
    
    def extract_color_features(self, pixels):
        """Extract color-based features from clothing pixels"""
        if len(pixels) == 0:
            return {
                'dark_ratio': 0, 'light_ratio': 0, 'metallic_ratio': 0,
                'avg_brightness': 0, 'dominant_hue': 0, 'saturation_avg': 0
            }
        
        # Convert BGR to HSV for better color analysis
        pixels_hsv = cv2.cvtColor(pixels.reshape(1, -1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        
        # Calculate brightness (V channel in HSV)
        brightness = pixels_hsv[:, 2]
        avg_brightness = np.mean(brightness)
        
        # Calculate saturation
        saturation = pixels_hsv[:, 1]
        saturation_avg = np.mean(saturation)
        
        # Classify pixels by clothing type - more focused on jacket differences
        dark_pixels = np.sum(brightness < 100)  # Dark clothing (black leather) - more lenient
        light_pixels = np.sum((brightness > 140) & (saturation < 60))  # Light/metallic (silver) - more lenient  
        metallic_pixels = np.sum((brightness > 100) & (brightness < 180) & (saturation < 100))  # Metallic shine
        
        total_pixels = len(pixels)
        
        # Calculate dominant hue (most common hue bin)
        hue_hist, _ = np.histogram(pixels_hsv[:, 0], bins=18, range=(0, 180))
        dominant_hue = np.argmax(hue_hist) * 10  # Convert bin to hue value
        
        return {
            'dark_ratio': dark_pixels / total_pixels,
            'light_ratio': light_pixels / total_pixels,
            'metallic_ratio': metallic_pixels / total_pixels,
            'avg_brightness': avg_brightness / 255.0,
            'dominant_hue': dominant_hue / 180.0,  # Normalize
            'saturation_avg': saturation_avg / 255.0
        }
    
    def extract_texture_features(self, clothing_region, clothing_mask):
        """Extract texture features from clothing"""
        if clothing_region.size == 0:
            return {'texture_contrast': 0, 'texture_homogeneity': 0}
        
        # Convert to grayscale
        gray = cv2.cvtColor(clothing_region, cv2.COLOR_BGR2GRAY)
        
        # Apply clothing mask
        gray_masked = cv2.bitwise_and(gray, clothing_mask)
        
        # Calculate texture metrics
        # Standard deviation as texture measure
        texture_contrast = np.std(gray_masked[clothing_mask > 0]) if np.sum(clothing_mask) > 0 else 0
        
        # Calculate local homogeneity
        kernel = np.ones((5,5), np.float32) / 25
        smooth = cv2.filter2D(gray_masked, -1, kernel)
        texture_homogeneity = np.mean(np.abs(gray_masked - smooth)[clothing_mask > 0]) if np.sum(clothing_mask) > 0 else 0
        
        return {
            'texture_contrast': texture_contrast / 255.0,  # Normalize
            'texture_homogeneity': texture_homogeneity / 255.0
        }
    
    def group_by_similarity(self, images_with_features):
        """Group images by outfit similarity using clustering"""
        # Prepare feature matrix - include more features for better discrimination
        feature_names = ['dark_ratio', 'light_ratio', 'avg_brightness', 'dominant_hue', 'saturation_avg']
        
        features_matrix = []
        for img_data in images_with_features:
            feature_vector = [img_data['features'][name] for name in feature_names]
            features_matrix.append(feature_vector)
        
        features_matrix = np.array(features_matrix)
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_matrix)
        
        # Use DBSCAN clustering for automatic group detection
        # eps controls how similar images need to be to group together
        # Lower eps = more groups, higher eps = fewer groups
        clustering = DBSCAN(eps=0.6, min_samples=1, metric='euclidean')
        group_labels = clustering.fit_predict(features_scaled)
        
        # Handle noise points (label -1) by giving them unique groups
        max_label = max(group_labels) if len(group_labels) > 0 else -1
        for i, label in enumerate(group_labels):
            if label == -1:
                max_label += 1
                group_labels[i] = max_label
        
        # Assign groups to images
        groups = {}
        for i, (img_data, group_id) in enumerate(zip(images_with_features, group_labels)):
            group_id = int(group_id) + 1  # Start groups from 1
            if group_id not in groups:
                groups[group_id] = []
            groups[group_id].append(img_data)
            
            print(f"{img_data['name']} → Group {group_id}")
        
        return groups
    
    def create_group_summary(self, groups):
        """Create visual summary of groups"""
        summary_path = self.output_dir / "group_summary.jpg"
        
        # Calculate grid size
        total_images = sum(len(group) for group in groups.values())
        cols = min(6, total_images)
        rows = (total_images + cols - 1) // cols
        
        # Create summary image
        img_size = 200
        summary_width = cols * img_size
        summary_height = rows * (img_size + 50)  # Extra space for labels
        
        summary = Image.new('RGB', (summary_width, summary_height), 'white')
        draw = ImageDraw.Draw(summary)
        
        x, y = 0, 0
        for group_id, images in sorted(groups.items()):
            for img_data in images:
                # Load and resize image
                pil_img = Image.open(img_data['path'])
                pil_img = pil_img.resize((img_size, img_size), Image.Resampling.LANCZOS)
                
                # Paste image
                summary.paste(pil_img, (x, y))
                
                # Add group label
                draw.text((x + 5, y + img_size + 5), f"Group {group_id}", fill='black')
                draw.text((x + 5, y + img_size + 25), img_data['name'], fill='gray')
                
                x += img_size
                if x >= summary_width:
                    x = 0
                    y += img_size + 50
        
        summary.save(summary_path)
        print(f"Group summary saved: {summary_path}")
        
        return summary_path
    
    def combine_group_images(self, group_images, group_id):
        """Combine images from a group into a single product image"""
        if not group_images:
            return None
        
        # Load images and get dimensions
        images = []
        total_height = 0
        max_width = 0
        
        for img_data in group_images:
            img = Image.open(img_data['path'])
            images.append(img)
            total_height += img.height
            max_width = max(max_width, img.width)
        
        # Create combined image
        combined = Image.new('RGB', (max_width, total_height), 'white')
        
        # Paste images vertically
        y_offset = 0
        for img in images:
            # Center horizontally
            x_offset = (max_width - img.width) // 2
            combined.paste(img, (x_offset, y_offset))
            y_offset += img.height
        
        # Auto-crop to remove excessive white space
        combined = self.auto_crop(combined)
        
        # Save combined image
        output_path = self.output_dir / f"product_{group_id}.png"
        combined.save(output_path, 'PNG', quality=95)
        
        print(f"Created product image: {output_path}")
        return output_path
    
    def auto_crop(self, image):
        """Auto-crop image to remove white borders"""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Find non-white pixels
        # White pixels have all RGB values > 240
        non_white = np.where(np.any(img_array < 240, axis=2))
        
        if len(non_white[0]) == 0:
            return image  # All white, return as-is
        
        # Get bounding box
        top = np.min(non_white[0])
        bottom = np.max(non_white[0])
        left = np.min(non_white[1])
        right = np.max(non_white[1])
        
        # Add small padding
        padding = 20
        top = max(0, top - padding)
        bottom = min(img_array.shape[0], bottom + padding)
        left = max(0, left - padding)
        right = min(img_array.shape[1], right + padding)
        
        # Crop image
        cropped = image.crop((left, top, right, bottom))
        return cropped
    
    def process_images(self, image_dir):
        """Main processing function"""
        print("Loading images...")
        images = self.load_images(image_dir)
        
        if not images:
            print("No images found!")
            return
        
        print("\nExtracting features...")
        images_with_features = []
        for img_data in images:
            features = self.extract_features(img_data)
            img_data['features'] = features
            images_with_features.append(img_data)
        
        print("\nGrouping by similarity...")
        groups = self.group_by_similarity(images_with_features)
        
        print(f"\nFound {len(groups)} groups:")
        for group_id, group_images in sorted(groups.items()):
            image_names = [img['name'] for img in group_images]
            print(f"Group {group_id}: {image_names}")
        
        print("\nCreating group summary...")
        self.create_group_summary(groups)
        
        print("\nCreating product images...")
        product_paths = []
        for group_id, group_images in sorted(groups.items()):
            path = self.combine_group_images(group_images, group_id)
            if path:
                product_paths.append(path)
        
        # Save grouping info as JSON
        group_info = {}
        for group_id, group_images in groups.items():
            group_info[f"group_{group_id}"] = [img['name'] for img in group_images]
        
        info_path = self.output_dir / "grouping_info.json"
        with open(info_path, 'w') as f:
            json.dump(group_info, f, indent=2)
        
        print(f"\nProcessing complete!")
        print(f"Output directory: {self.output_dir}")
        print(f"Product images: {len(product_paths)}")
        print(f"Grouping info saved: {info_path}")
        
        return product_paths


def main():
    parser = argparse.ArgumentParser(description='Group fashion images by outfit similarity')
    parser.add_argument('input_dir', help='Directory containing input images')
    parser.add_argument('--output', '-o', default='output', help='Output directory')
    
    args = parser.parse_args()
    
    grouper = OutfitGrouper(args.output)
    grouper.process_images(args.input_dir)


if __name__ == "__main__":
    main()