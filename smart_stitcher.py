#!/usr/bin/env python3
"""
Smart Taobao Image Stitcher - Actually combines fragments intelligently
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class SmartStitcher:
    def __init__(self, result_dir, session_id=None):
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(exist_ok=True)
        self.session_id = session_id
        
    def find_best_merge_point(self, img1, img2):
        """Find where to merge two images by looking for overlapping content"""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Convert to grayscale for analysis
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        best_match_score = -1
        best_merge_point = h1 // 2  # Default to middle
        
        # Search for matching regions in the overlap zone
        search_range = min(h1 // 3, h2 // 3, 200)  # Search in reasonable range
        
        for offset in range(max(50, h1 - search_range), min(h1 - 50, h1)):
            # Get bottom strip from img1
            strip1 = gray1[offset-30:offset, :]
            # Get top strip from img2  
            strip2 = gray2[0:30, :]
            
            if strip1.shape[1] != strip2.shape[1]:
                # Resize to match widths
                strip2 = cv2.resize(strip2, (strip1.shape[1], strip2.shape[0]))
            
            # Calculate similarity
            if strip1.shape == strip2.shape:
                diff = cv2.absdiff(strip1, strip2)
                score = 1.0 - (np.mean(diff) / 255.0)
                
                if score > best_match_score:
                    best_match_score = score
                    best_merge_point = offset
        
        print(f"  Best merge point: {best_merge_point} (score: {best_match_score:.3f})")
        return best_merge_point
    
    def smart_stitch(self, fragment_img, full_img, fragment_type='head'):
        """Intelligently stitch fragment with full image"""
        fh, fw = fragment_img.shape[:2]
        ih, iw = full_img.shape[:2]
        
        print(f"  Smart stitching: fragment {fw}x{fh}, full {iw}x{ih}")
        
        # Ensure same width
        if fw != iw:
            # Resize fragment to match full image width
            scale = iw / fw
            new_fh = int(fh * scale)
            fragment_img = cv2.resize(fragment_img, (iw, new_fh))
            fh, fw = fragment_img.shape[:2]
            print(f"  Resized fragment to {fw}x{fh}")
        
        if fragment_type == 'head':
            # Fragment is head/upper body - use it for top portion
            # Find where fragment ends (where to merge)
            merge_point = self.find_best_merge_point(fragment_img, full_img)
            
            # Take top part from fragment
            top_part = fragment_img[0:merge_point, :]
            
            # Find corresponding point in full image
            # Look for where the fragment content appears in full image
            search_start = max(0, merge_point - 100)
            search_end = min(ih, merge_point + 200)
            
            best_match_y = merge_point
            best_score = -1
            
            for y in range(search_start, search_end - 30):
                full_strip = cv2.cvtColor(full_img[y:y+30, :], cv2.COLOR_BGR2GRAY)
                frag_strip = cv2.cvtColor(fragment_img[merge_point-30:merge_point, :], cv2.COLOR_BGR2GRAY)
                
                if full_strip.shape == frag_strip.shape:
                    diff = cv2.absdiff(full_strip, frag_strip)
                    score = 1.0 - (np.mean(diff) / 255.0)
                    
                    if score > best_score:
                        best_score = score
                        best_match_y = y
            
            print(f"  Found match point in full image at y={best_match_y}")
            
            # Take bottom part from full image
            bottom_part = full_img[best_match_y:, :]
            
            # Combine them
            if top_part.shape[1] == bottom_part.shape[1]:
                result = np.vstack([top_part, bottom_part])
                print(f"  Combined result: {result.shape}")
                return result
            else:
                print(f"  Width mismatch, using full image")
                return full_img
                
        elif fragment_type == 'bottom':
            # Fragment is lower body - use it for bottom portion
            # This is less common but handle it similarly
            merge_point = fh // 2
            
            # Find where this fragment starts in the full image
            top_part = full_img[0:ih-fh+merge_point, :]
            bottom_part = fragment_img[merge_point:, :]
            
            if top_part.shape[1] == bottom_part.shape[1]:
                result = np.vstack([top_part, bottom_part])
                return result
            else:
                return full_img
        
        else:
            # Unknown fragment type, just return full image
            return full_img
    
    def detect_fragment_type(self, fragment_img, full_img):
        """Detect if fragment is head/upper body or lower body"""
        fh, fw = fragment_img.shape[:2]
        ih, iw = full_img.shape[:2]
        
        # Simple heuristic: if fragment is much shorter than full, it's likely upper body
        height_ratio = fh / ih
        
        if height_ratio < 0.4:
            # Very short fragment - likely just head/upper torso
            return 'head'
        elif height_ratio < 0.7:
            # Medium fragment - could be upper or lower
            # Check which part has more content
            upper_content = np.mean(fragment_img[:fh//2, :])
            lower_content = np.mean(fragment_img[fh//2:, :])
            
            if upper_content > lower_content:
                return 'head'
            else:
                return 'bottom'
        else:
            # Large fragment - might not need stitching
            return 'full'
    
    def process_image_pair(self, img1_path, img2_path, output_path):
        """Process a pair of images (fragment + full)"""
        # Load images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            print(f"Error loading images: {img1_path}, {img2_path}")
            return False
        
        print(f"\nProcessing pair: {Path(img1_path).name} + {Path(img2_path).name}")
        
        # Determine which is fragment and which is full
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        if h1 < h2:
            # img1 is fragment, img2 is full
            fragment_img = img1
            full_img = img2
            fragment_type = self.detect_fragment_type(fragment_img, full_img)
            print(f"  Fragment type detected: {fragment_type}")
            
            if fragment_type == 'full':
                # Not really a fragment, just use img2
                result = full_img
            else:
                # Actually stitch them
                result = self.smart_stitch(fragment_img, full_img, fragment_type)
        else:
            # img2 might be fragment or they're similar size
            if h2 < h1 * 0.7:
                # img2 is fragment
                fragment_img = img2
                full_img = img1
                fragment_type = self.detect_fragment_type(fragment_img, full_img)
                print(f"  Fragment type detected: {fragment_type}")
                result = self.smart_stitch(fragment_img, full_img, fragment_type)
            else:
                # Similar sizes, just use img2 (the second one)
                result = img2
        
        # Apply 4:5 ratio if needed
        result = self.enforce_ratio(result)
        
        # Save result
        success = cv2.imwrite(output_path, result)
        if success:
            print(f"  Saved: {output_path}")
        else:
            print(f"  Failed to save: {output_path}")
        
        return success
    
    def enforce_ratio(self, image, target_ratio=4/5):
        """Enforce aspect ratio with padding"""
        height, width = image.shape[:2]
        current_ratio = width / height
        
        if abs(current_ratio - target_ratio) < 0.01:
            return image
        
        if current_ratio > target_ratio:
            # Too wide - add padding to height
            new_height = int(width / target_ratio)
            padding = new_height - height
            pad_top = padding // 2
            pad_bottom = padding - pad_top
            
            padded = cv2.copyMakeBorder(
                image, pad_top, pad_bottom, 0, 0,
                cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
        else:
            # Too tall - add padding to width
            new_width = int(height * target_ratio)
            padding = new_width - width
            pad_left = padding // 2
            pad_right = padding - pad_left
            
            padded = cv2.copyMakeBorder(
                image, 0, 0, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
        
        return padded


# Test function
def test_smart_stitcher():
    """Test the smart stitcher with images 2+3 and 4+5"""
    stitcher = SmartStitcher('smart_stitch_results')
    
    # Test case 1: Image 2 (head fragment) + Image 3 (full body)
    stitcher.process_image_pair(
        'results/2.jpg',
        'results/3.jpg', 
        'smart_stitch_results/product_2_smart.jpg'
    )
    
    # Test case 2: Image 4 (upper fragment) + Image 5 (full body)
    stitcher.process_image_pair(
        'results/4.jpg',
        'results/5.jpg',
        'smart_stitch_results/product_3_smart.jpg'
    )
    
    print("\nSmart stitching complete! Check smart_stitch_results folder.")


if __name__ == "__main__":
    test_smart_stitcher()