#!/usr/bin/env python3
"""
Simple Sequential Image Pairing
Just pairs images sequentially: 1+2, 2+3, 3+4, etc.
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import json

class SimpleImagePairer:
    def __init__(self, output_dir="simple_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def load_images(self, image_dir):
        """Load all images from directory in order"""
        image_dir = Path(image_dir)
        image_files = (list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg")) + 
                      list(image_dir.glob("*.png")) + list(image_dir.glob("*.gif")) + 
                      list(image_dir.glob("*.webp")) + list(image_dir.glob("*.avif")))
        
        # Sort numerically by filename
        image_files.sort(key=lambda x: x.name)
        
        images = []
        for img_path in image_files:
            try:
                # Try OpenCV first
                img = cv2.imread(str(img_path))
                
                # If OpenCV fails, try Pillow
                if img is None:
                    pil_img = Image.open(img_path)
                    if pil_img.mode != 'RGB':
                        pil_img = pil_img.convert('RGB')
                    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                
                if img is not None:
                    images.append({
                        'path': img_path,
                        'image': img,
                        'name': img_path.name
                    })
                    print(f"Loaded: {img_path.name}")
                    
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        return images
    
    def create_sequential_pairs(self, images):
        """Create sequential pairs: 1+2, 2+3, 3+4, etc."""
        pairs = []
        
        for i in range(len(images) - 1):
            pair = {
                'product_id': i + 1,
                'images': [images[i], images[i + 1]],
                'names': [images[i]['name'], images[i + 1]['name']]
            }
            pairs.append(pair)
            print(f"Pair {i + 1}: {images[i]['name']} + {images[i + 1]['name']}")
        
        return pairs
    
    def combine_pair_images(self, pair):
        """Combine two images vertically into one product image"""
        img1 = pair['images'][0]['image']
        img2 = pair['images'][1]['image']
        
        # Get dimensions
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Use maximum width
        max_width = max(w1, w2)
        total_height = h1 + h2
        
        # Create combined image with white background
        combined = np.full((total_height, max_width, 3), 255, dtype=np.uint8)
        
        # Center first image horizontally
        x1_offset = (max_width - w1) // 2
        combined[0:h1, x1_offset:x1_offset + w1] = img1
        
        # Center second image horizontally  
        x2_offset = (max_width - w2) // 2
        combined[h1:h1 + h2, x2_offset:x2_offset + w2] = img2
        
        # Convert to PIL for auto-cropping
        combined_pil = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        
        # Auto-crop to remove excessive white space
        cropped = self.auto_crop(combined_pil)
        
        # Save product image
        output_path = self.output_dir / f"product_{pair['product_id']}.png"
        cropped.save(output_path, 'PNG', quality=95)
        
        print(f"Created: {output_path.name} ({cropped.size[0]}x{cropped.size[1]}px)")
        return output_path
    
    def auto_crop(self, image):
        """Auto-crop image to remove white borders"""
        img_array = np.array(image)
        
        # Find non-white pixels (allowing for slight variations)
        non_white = np.where(np.any(img_array < 250, axis=2))
        
        if len(non_white[0]) == 0:
            return image  # All white, return as-is
        
        # Get bounding box
        top = np.min(non_white[0])
        bottom = np.max(non_white[0])
        left = np.min(non_white[1])
        right = np.max(non_white[1])
        
        # Add small padding
        padding = 10
        top = max(0, top - padding)
        bottom = min(img_array.shape[0], bottom + padding)
        left = max(0, left - padding)
        right = min(img_array.shape[1], right + padding)
        
        # Crop image
        cropped = image.crop((left, top, right, bottom))
        return cropped
    
    def process_images(self, image_dir):
        """Main processing function"""
        print("🔄 Simple Sequential Image Pairing")
        print("=" * 40)
        
        # Load images
        print("Loading images...")
        images = self.load_images(image_dir)
        
        if len(images) < 2:
            print("❌ Need at least 2 images to create pairs!")
            return []
        
        print(f"✅ Loaded {len(images)} images")
        
        # Create sequential pairs
        print("\nCreating sequential pairs...")
        pairs = self.create_sequential_pairs(images)
        print(f"✅ Created {len(pairs)} pairs")
        
        # Process each pair
        print("\nGenerating product images...")
        product_paths = []
        
        for pair in pairs:
            try:
                path = self.combine_pair_images(pair)
                product_paths.append({
                    'product_id': pair['product_id'],
                    'path': str(path),
                    'filename': path.name,
                    'images': pair['names']
                })
            except Exception as e:
                print(f"❌ Error creating product {pair['product_id']}: {e}")
        
        # Save pairing info
        info_path = self.output_dir / "pairing_info.json"
        pairing_info = {}
        for product in product_paths:
            pairing_info[f"product_{product['product_id']}"] = product['images']
        
        with open(info_path, 'w') as f:
            json.dump(pairing_info, f, indent=2)
        
        print(f"\n✅ Processing complete!")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🎯 Created {len(product_paths)} product images")
        print(f"📋 Pairing info saved: {info_path}")
        
        return product_paths


def main():
    # Use current directory as input
    current_dir = Path(".")
    
    # Check if numbered images exist
    image_files = list(current_dir.glob("*.jpg")) + list(current_dir.glob("*.png"))
    image_files = [f for f in image_files if f.name[0].isdigit()]
    
    if not image_files:
        print("❌ No numbered image files found!")
        print("Expected files: 1.jpg, 2.jpg, etc.")
        return
    
    print(f"📸 Found {len(image_files)} images")
    
    # Create pairer and process
    pairer = SimpleImagePairer()
    pairer.process_images(current_dir)


if __name__ == "__main__":
    main()