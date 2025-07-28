#!/usr/bin/env python3
"""
Simple Fixed Position Splitter for Fashion Images
Just splits each image in half - no complex detection needed!
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import json


class SimpleFixedSplitter:
    def __init__(self, result_dir):
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(exist_ok=True)
        
    def load_images(self, image_dir):
        """Load all images from directory"""
        image_dir = Path(image_dir)
        images = []
        
        supported_formats = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.avif']
        
        for file_path in sorted(image_dir.glob('*')):
            if file_path.suffix.lower() in supported_formats:
                try:
                    # Try OpenCV first
                    img = cv2.imread(str(file_path))
                    if img is not None:
                        images.append({
                            'path': str(file_path),
                            'name': file_path.name,
                            'cv2_image': img
                        })
                        print(f"Loaded: {file_path.name}")
                    else:
                        # Fallback to PIL for formats like AVIF
                        pil_img = Image.open(file_path).convert('RGB')
                        cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                        images.append({
                            'path': str(file_path),
                            'name': file_path.name,
                            'cv2_image': cv2_img
                        })
                        print(f"Loaded (PIL): {file_path.name}")
                        
                except Exception as e:
                    print(f"Error loading {file_path.name}: {e}")
                    
        return images
    
    def process_images(self, images):
        """Process all images - simply split each in half"""
        products = []
        
        for i, img_data in enumerate(images):
            print(f"\nProcessing image {i+1}/{len(images)}: {img_data['name']}")
            
            image = img_data['cv2_image']
            height, width = image.shape[:2]
            
            # Simple approach: just split in the middle!
            middle_y = height // 2
            
            # Top half
            top_half = image[0:middle_y, :]
            
            # Bottom half  
            bottom_half = image[middle_y:height, :]
            
            # Create product by combining them
            product_img = np.vstack([top_half, bottom_half])
            
            # Save product
            product_id = i + 1
            filename = f"product_{product_id}.png"
            output_path = self.result_dir / filename
            cv2.imwrite(str(output_path), product_img)
            
            products.append({
                'product_id': product_id,
                'filename': filename,
                'path': str(output_path),
                'source': img_data['name'],
                'dimensions': {
                    'width': width,
                    'height': height
                }
            })
            
            print(f"Created product {product_id}")
            
            # Also save debug image showing the split
            debug_img = image.copy()
            cv2.line(debug_img, (0, middle_y), (width, middle_y), (0, 255, 0), 3)
            cv2.putText(debug_img, "SPLIT HERE", (10, middle_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            debug_filename = f"debug_{img_data['name']}"
            debug_path = self.result_dir / debug_filename
            cv2.imwrite(str(debug_path), debug_img)
        
        return products


def main():
    """Test the simple splitter"""
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python simple_fixed_splitter.py <input_dir> <output_dir>")
        return
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    splitter = SimpleFixedSplitter(output_dir)
    
    print("Loading images...")
    images = splitter.load_images(input_dir)
    if not images:
        print("No images found!")
        return
    
    print(f"Processing {len(images)} images...")
    products = splitter.process_images(images)
    
    print(f"\nCreated {len(products)} products")
    
    # Save processing info
    info = {
        'total_images': len(images),
        'total_products': len(products),
        'products': products
    }
    
    info_path = Path(output_dir) / "processing_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\nProcessing complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()