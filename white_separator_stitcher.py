#!/usr/bin/env python3
"""
White Background Separator Stitcher - Uses white background as product boundary detector
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

class WhiteSeparatorStitcher:
    def __init__(self, result_dir, session_id=None):
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(exist_ok=True)
        self.session_id = session_id
        self.white_threshold = 240  # Pixels brighter than this are considered "white"
        
    def detect_white_separators(self, image):
        """Detect significant horizontal white separator regions in image"""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        separators = []
        already_found = set()
        
        # Look for transitions from content to white background
        content_regions = self.find_content_regions(gray)
        
        print(f"    Found {len(content_regions)} content regions")
        
        for region in content_regions:
            # Look for white space after this content region
            search_start = region['end'] + 10
            search_end = min(h - 50, search_start + 200)
            
            if search_start >= search_end:
                continue
                
            # Find the largest white region in this search area
            best_separator = None
            max_white_height = 0
            
            for y in range(search_start, search_end):
                if y in already_found:
                    continue
                    
                row = gray[y, :]
                white_ratio = np.sum(row > self.white_threshold) / w
                
                if white_ratio > 0.9:  # Very white row
                    white_height = self.measure_white_region_height(gray, y)
                    
                    if white_height > max_white_height and white_height > 50:
                        max_white_height = white_height
                        best_separator = {
                            'y': y,
                            'height': white_height,
                            'white_ratio': white_ratio
                        }
            
            if best_separator:
                separators.append(best_separator)
                # Mark this region as found
                for y in range(best_separator['y'], best_separator['y'] + best_separator['height']):
                    already_found.add(y)
                print(f"    Significant separator at y={best_separator['y']}, height={best_separator['height']}")
        
        return separators
    
    def find_content_regions(self, gray_image):
        """Find regions with actual content (non-white)"""
        h, w = gray_image.shape
        content_regions = []
        
        in_content = False
        current_start = 0
        
        for y in range(h):
            row = gray_image[y, :]
            white_ratio = np.sum(row > self.white_threshold) / w
            has_content = white_ratio < 0.8  # Less than 80% white = has content
            
            if has_content and not in_content:
                # Start of content region
                current_start = y
                in_content = True
            elif not has_content and in_content:
                # End of content region
                if y - current_start > 100:  # Only significant regions
                    content_regions.append({
                        'start': current_start,
                        'end': y,
                        'height': y - current_start
                    })
                in_content = False
        
        # Handle case where content goes to end
        if in_content and h - current_start > 100:
            content_regions.append({
                'start': current_start,
                'end': h,
                'height': h - current_start
            })
        
        return content_regions
    
    def measure_white_region_height(self, gray_image, start_y):
        """Measure how tall a white region is starting from start_y"""
        h, w = gray_image.shape
        height = 0
        
        # Go down from start_y
        for y in range(start_y, min(h, start_y + 100)):
            row = gray_image[y, :]
            white_ratio = np.sum(row > self.white_threshold) / w
            
            if white_ratio > 0.8:
                height += 1
            else:
                break
        
        return height
    
    def stack_and_split(self, img1, img2, take_from_img1="auto"):
        """Stack two images and split at white separator
        take_from_img1: 'top', 'bottom', or 'auto' (detect automatically)
        """
        # Ensure same width
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        if w1 != w2:
            # Resize img2 to match img1 width
            scale = w1 / w2
            new_h2 = int(h2 * scale)
            img2 = cv2.resize(img2, (w1, new_h2))
        
        # For complex images like 7.jpg, we might need specific part
        if take_from_img1 == "bottom":
            # Take bottom part of img1 (after last separator)
            separators_img1 = self.detect_white_separators(img1)
            if separators_img1:
                # Take everything after the last separator
                last_sep = separators_img1[-1]
                cut_start = last_sep['y'] + last_sep['height']
                img1_part = img1[cut_start:, :]
                print(f"  Taking bottom part of img1 from y={cut_start}")
            else:
                img1_part = img1
                print("  No separators in img1, using full image")
        else:
            img1_part = img1
        
        # Stack vertically
        stacked = np.vstack([img1_part, img2])
        print(f"  Stacked images: {img1_part.shape} + {img2.shape} = {stacked.shape}")
        
        # Find white separators
        separators = self.detect_white_separators(stacked)
        
        if separators:
            # Use the first significant separator
            best_separator = separators[0]
            cut_point = best_separator['y']
            
            # Cut at separator
            product = stacked[:cut_point, :]
            print(f"  Cut at white separator y={cut_point}")
            
            # Remove any white space at bottom
            product = self.trim_bottom_white(product)
            
            return product
        else:
            print("  No white separator found, using full stack")
            # No separator found - remove white from bottom
            return self.trim_bottom_white(stacked)
    
    def trim_bottom_white(self, image):
        """Remove white space from bottom of image"""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find last non-white row
        for y in range(h - 1, -1, -1):
            row = gray[y, :]
            white_ratio = np.sum(row > self.white_threshold) / w
            
            if white_ratio < 0.9:  # Found non-white content
                return image[:y + 20, :]  # Keep a small margin
        
        return image  # All white or no change needed
    
    def enforce_4_5_ratio(self, image):
        """Enforce 4:5 aspect ratio with white padding"""
        height, width = image.shape[:2]
        target_ratio = 4.0 / 5.0
        current_ratio = width / height
        
        if abs(current_ratio - target_ratio) < 0.01:
            return image
        
        if current_ratio > target_ratio:
            # Too wide - add white padding to height
            new_height = int(width / target_ratio)
            padding = new_height - height
            pad_top = padding // 2
            pad_bottom = padding - pad_top
            
            result = cv2.copyMakeBorder(
                image, pad_top, pad_bottom, 0, 0,
                cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
        else:
            # Too tall - add white padding to width
            new_width = int(height * target_ratio)
            padding = new_width - width
            pad_left = padding // 2
            pad_right = padding - pad_left
            
            result = cv2.copyMakeBorder(
                image, 0, 0, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
        
        return result
    
    def process_pair(self, img1, img2):
        """Process a pair of images with white separator detection"""
        print(f"\nProcessing pair with white separator detection...")
        
        # Stack and split at white separator
        result = self.stack_and_split(img1, img2)
        
        # Enforce 4:5 ratio
        result = self.enforce_4_5_ratio(result)
        
        return result
    
    def process_triple(self, img1, img2, img3):
        """Process 3 images - stack all and find separators"""
        print(f"\nProcessing triple with white separator detection...")
        
        # First stack img1 + img2
        intermediate = self.stack_and_split(img1, img2)
        
        # Then add img3 and look for separator
        final_result = self.stack_and_split(intermediate, img3)
        
        # Enforce 4:5 ratio
        final_result = self.enforce_4_5_ratio(final_result)
        
        return final_result


# Test function
def test_white_separator():
    """Test white separator detection with 1-9.jpg"""
    from puzzle_reconstructor import PuzzleReconstructor
    
    print("Testing white separator stitching...")
    
    # Load images
    reconstructor = PuzzleReconstructor('white_separator_results')
    images = reconstructor.load_images('results')
    
    # Filter to 1-9.jpg
    test_images = [img for img in images if img['name'] in 
                   ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg', '7.jpg', '8.jpg', '9.jpg']]
    
    stitcher = WhiteSeparatorStitcher('white_separator_results')
    
    # Test specific pairs according to user pattern
    print("\n=== Testing 1 (Product 1 - Standalone) ===")
    result_1 = stitcher.enforce_4_5_ratio(test_images[0]['cv2_image'])
    cv2.imwrite('white_separator_results/product_1_white.jpg', result_1)
    print(f"Saved product_1_white.jpg: {result_1.shape}")
    
    print("\n=== Testing 2+3 (Product 2) ===")
    result_2_3 = stitcher.process_pair(
        test_images[1]['cv2_image'],  # 2.jpg
        test_images[2]['cv2_image']   # 3.jpg
    )
    cv2.imwrite('white_separator_results/product_2_white.jpg', result_2_3)
    print(f"Saved product_2_white.jpg: {result_2_3.shape}")
    
    print("\n=== Testing 4+5 (Product 3) ===")
    result_4_5 = stitcher.process_pair(
        test_images[3]['cv2_image'],  # 4.jpg
        test_images[4]['cv2_image']   # 5.jpg
    )
    cv2.imwrite('white_separator_results/product_3_white.jpg', result_4_5)
    print(f"Saved product_3_white.jpg: {result_4_5.shape}")
    
    print("\n=== Testing 5+6+7 (Product 4) ===")
    result_5_6_7 = stitcher.process_triple(
        test_images[4]['cv2_image'],  # 5.jpg
        test_images[5]['cv2_image'],  # 6.jpg
        test_images[6]['cv2_image']   # 7.jpg
    )
    cv2.imwrite('white_separator_results/product_4_white.jpg', result_5_6_7)
    print(f"Saved product_4_white.jpg: {result_5_6_7.shape}")
    
    print("\n=== Testing 7+8 (Product 5) ===")
    # For 7+8, we need the bottom part of 7.jpg (head/torso) + 8.jpg  
    result_7_8 = stitcher.stack_and_split(
        test_images[6]['cv2_image'],  # 7.jpg
        test_images[7]['cv2_image'],  # 8.jpg
        take_from_img1="bottom"
    )
    result_7_8 = stitcher.enforce_4_5_ratio(result_7_8)
    cv2.imwrite('white_separator_results/product_5_white.jpg', result_7_8)
    print(f"Saved product_5_white.jpg: {result_7_8.shape}")
    
    print("\nWhite separator stitching complete!")


if __name__ == "__main__":
    test_white_separator()