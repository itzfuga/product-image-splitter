#!/usr/bin/env python3
"""
Content-Aware Smart Stitcher - Generic solution for Taobao image reconstruction
Uses content detection instead of pixel matching for natural merging
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

class ContentAwareStitcher:
    def __init__(self, result_dir, session_id=None):
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(exist_ok=True)
        self.session_id = session_id
        
    def detect_content_regions(self, image):
        """Detect key content regions in the image"""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find content bounds (non-background areas)
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Get bounding box of all content
        x_min, y_min = w, h
        x_max, y_max = 0, 0
        
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + cw)
            y_max = max(y_max, y + ch)
        
        # Detect natural break points (like waistline)
        break_points = self.find_natural_breaks(image, y_min, y_max)
        
        return {
            'bounds': (x_min, y_min, x_max, y_max),
            'height': y_max - y_min,
            'width': x_max - x_min,
            'break_points': break_points,
            'center_y': (y_min + y_max) // 2
        }
    
    def find_natural_breaks(self, image, y_start, y_end):
        """Find natural horizontal break points in the image (waist, neckline, etc)"""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        break_points = []
        
        # Look for horizontal edges (clothing boundaries)
        for y in range(max(y_start, h//3), min(y_end, 2*h//3)):
            # Get horizontal slice
            slice_img = gray[y-2:y+2, :]
            
            # Calculate variance across the slice
            variance = np.var(slice_img)
            
            # High variance might indicate a boundary
            if variance > 500:  # Threshold
                # Check if this is a consistent edge
                edge_strength = cv2.Sobel(gray[y-5:y+5, :], cv2.CV_64F, 0, 1)
                if np.mean(np.abs(edge_strength)) > 20:
                    break_points.append(y)
        
        # Filter out points too close together
        filtered_points = []
        for point in break_points:
            if not filtered_points or point - filtered_points[-1] > 50:
                filtered_points.append(point)
        
        return filtered_points
    
    def analyze_fragment_quality(self, fragment, full_img, fragment_region):
        """Analyze which image has better quality for specific regions"""
        fh, fw = fragment.shape[:2]
        ih, iw = full_img.shape[:2]
        
        # Resize for comparison if needed
        if fw != iw:
            scale = iw / fw
            fragment_resized = cv2.resize(fragment, (iw, int(fh * scale)))
        else:
            fragment_resized = fragment
        
        # Detect if fragment is actually a fragment (partial height)
        is_fragment = fh < ih * 0.5  # Fragment is less than half the height
        
        # For fragments, we usually want to use them for detail
        if is_fragment:
            print(f"  Detected fragment: {fw}x{fh} vs full: {iw}x{ih}")
            # Check if fragment has clear head/upper body
            gray_frag = cv2.cvtColor(fragment_resized, cv2.COLOR_BGR2GRAY)
            
            # Look for face/head features in top portion
            top_portion = gray_frag[:fh//2, :]
            edges = cv2.Canny(top_portion, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            has_detail = edge_density > 0.05  # Has significant detail
            
            return {
                'fragment_better_top': is_fragment and has_detail,
                'is_fragment': is_fragment,
                'fragment_region': self.detect_content_regions(fragment_resized),
                'full_region': self.detect_content_regions(full_img),
                'quality_ratio': 1.5 if is_fragment else 0.5
            }
        
        # Not a fragment - use normal quality comparison
        top_height = min(fragment_resized.shape[0], full_img.shape[0]) // 3
        
        frag_top = fragment_resized[:top_height, :]
        full_top = full_img[:top_height, :]
        
        frag_sharpness = cv2.Laplacian(cv2.cvtColor(frag_top, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        full_sharpness = cv2.Laplacian(cv2.cvtColor(full_top, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        
        return {
            'fragment_better_top': frag_sharpness > full_sharpness * 1.2,
            'is_fragment': False,
            'fragment_region': self.detect_content_regions(fragment_resized),
            'full_region': self.detect_content_regions(full_img),
            'quality_ratio': frag_sharpness / (full_sharpness + 1e-6)
        }
    
    def smart_merge(self, fragment, full_img):
        """Smart content-aware merging"""
        analysis = self.analyze_fragment_quality(fragment, full_img, None)
        
        # Resize fragment if needed
        fh, fw = fragment.shape[:2]
        ih, iw = full_img.shape[:2]
        
        if fw != iw:
            scale = iw / fw
            fragment = cv2.resize(fragment, (iw, int(fh * scale)))
            fh = fragment.shape[0]
        
        # If it's a fragment, analyze what it contains
        if analysis['is_fragment']:
            print(f"  Fragment detected ({fw}x{fh})")
            
            # Check if fragment has a head/face
            has_head = self.detect_head_in_image(fragment)
            
            if not has_head:
                # Fragment is torso only - use sandwich approach
                print("  Fragment is torso-only, using sandwich merge")
                
                # Find where the torso starts in the full image
                torso_start = self.find_torso_start(full_img)
                # Find where the torso ends in the fragment  
                torso_end_in_fragment = int(fh * 0.9)
                # Find where this maps to in the full image
                torso_end_in_full = self.find_matching_point(fragment, full_img, torso_end_in_fragment)
                
                if torso_start > 50:  # Make sure we have reasonable head region
                    # Take head from full image
                    head_part = full_img[:torso_start, :]
                    # Take torso from fragment (scaled)
                    torso_part = fragment[:torso_end_in_fragment, :]
                    # Take legs from full image  
                    legs_start = max(torso_end_in_full, torso_start + torso_end_in_fragment)
                    legs_part = full_img[legs_start:, :]
                    
                    if legs_part.shape[0] > 50:  # Make sure we have reasonable legs
                        # Combine all three parts
                        result = self.blend_three_parts(head_part, torso_part, legs_part)
                        print(f"  Sandwich merge: head({torso_start}) + torso({torso_end_in_fragment}) + legs({legs_start})")
                        return result
                    else:
                        print("  Legs part too small, combining head + torso only")
                        result = self.blend_images(head_part, torso_part)
                        return result
                else:
                    print("  Torso start too early, using fragment for top part")
                    # Use fragment for the whole top portion
                    merge_point = int(fh * 0.9)
                    top_part = fragment[:merge_point, :]
                    match_point = self.find_matching_point(fragment, full_img, merge_point)
                    bottom_part = full_img[match_point:, :]
                    result = self.blend_images(top_part, bottom_part)
                    return result
            else:
                # Fragment has head - use it for top
                print("  Fragment has head, using for top detail")
                merge_point = int(fh * 0.85)
                top_part = fragment[:merge_point, :]
                
                match_point = self.find_matching_point(fragment, full_img, merge_point)
                if match_point < ih - 100:
                    bottom_part = full_img[match_point:, :]
                    result = self.blend_images(top_part, bottom_part)
                    print(f"  Merged at point {merge_point} (fragment) -> {match_point} (full)")
                    return result
        
        # Default: use full image
        print("  Using full image")
        return full_img
    
    def find_optimal_merge_point(self, fragment, full_img):
        """Find the best point to merge based on content"""
        fh = fragment.shape[0]
        
        # Look for natural breaks in the fragment
        frag_content = self.detect_content_regions(fragment)
        
        if frag_content and frag_content['break_points']:
            # Use the first major break point (e.g., waistline)
            for point in frag_content['break_points']:
                if point > fh * 0.4 and point < fh * 0.8:  # Middle region
                    return point
        
        # Fallback: use proportion
        return int(fh * 0.6)  # Default to 60% height
    
    def find_matching_point(self, fragment, full_img, frag_point):
        """Find corresponding point in full image"""
        ih = full_img.shape[0]
        
        # Get content around merge point in fragment
        search_height = 50
        start = max(0, frag_point - search_height)
        end = min(fragment.shape[0], frag_point + search_height)
        
        frag_region = cv2.cvtColor(fragment[start:end, :], cv2.COLOR_BGR2GRAY)
        
        # Search in full image
        best_match = frag_point
        best_score = -1
        
        search_start = max(0, frag_point - 200)
        search_end = min(ih - search_height, frag_point + 200)
        
        for y in range(search_start, search_end, 5):
            full_region = cv2.cvtColor(full_img[y:y+(end-start), :], cv2.COLOR_BGR2GRAY)
            
            if full_region.shape == frag_region.shape:
                # Use correlation
                score = cv2.matchTemplate(full_region, frag_region, cv2.TM_CCORR_NORMED)[0, 0]
                
                if score > best_score:
                    best_score = score
                    best_match = y + (frag_point - start)
        
        return best_match
    
    def blend_images(self, top_part, bottom_part):
        """Blend two image parts with smooth transition"""
        if top_part.shape[1] != bottom_part.shape[1]:
            # Width mismatch - use bottom image
            return bottom_part
            
        # Create a blended transition zone
        blend_height = 20  # Height of blend zone
        
        if top_part.shape[0] > blend_height and bottom_part.shape[0] > blend_height:
            # Extract overlap regions
            top_blend = top_part[-blend_height:, :].astype(np.float32)
            bottom_blend = bottom_part[:blend_height, :].astype(np.float32)
            
            # Create gradient weights
            weights = np.linspace(1, 0, blend_height).reshape(-1, 1, 1)
            
            # Blend the overlap
            blended = top_blend * weights + bottom_blend * (1 - weights)
            
            # Combine all parts
            result = np.vstack([
                top_part[:-blend_height, :],
                blended.astype(np.uint8),
                bottom_part[blend_height:, :]
            ])
            
            return result
        else:
            # Too small to blend - just stack
            return np.vstack([top_part, bottom_part])
    
    
    def detect_head_in_image(self, image):
        """Detect if image contains a head/face region"""
        h, w = image.shape[:2]
        
        # Look in the top 1/3 of the image
        top_region = image[:h//3, :]
        gray = cv2.cvtColor(top_region, cv2.COLOR_BGR2GRAY)
        
        # Use Haar cascades for face detection (basic approach)
        # For now, use edge density as proxy for head/face detail
        edges = cv2.Canny(gray, 30, 100)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Also check for skin-like colors in top region
        hsv = cv2.cvtColor(top_region, cv2.COLOR_BGR2HSV)
        skin_mask = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([20, 255, 255]))
        skin_ratio = np.sum(skin_mask > 0) / (skin_mask.shape[0] * skin_mask.shape[1])
        
        has_head = edge_density > 0.08 or skin_ratio > 0.05
        print(f"    Head detection: edge_density={edge_density:.3f}, skin_ratio={skin_ratio:.3f}, has_head={has_head}")
        return has_head
    
    def find_torso_start(self, full_img):
        """Find where the torso/upper body starts in full image"""
        h, w = full_img.shape[:2]
        
        # Look for the neck/shoulder line (usually around 20-30% down)
        search_start = int(h * 0.15)
        search_end = int(h * 0.4)
        
        gray = cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY)
        
        # Find horizontal edges that might indicate shoulder line
        best_line = search_start + int((search_end - search_start) * 0.6)  # Default
        max_edge_strength = 0
        
        for y in range(search_start, search_end, 5):
            # Get horizontal slice
            slice_region = gray[max(0, y-3):min(h, y+3), :]
            
            # Calculate horizontal gradient
            grad = cv2.Sobel(slice_region, cv2.CV_64F, 0, 1)
            edge_strength = np.mean(np.abs(grad))
            
            if edge_strength > max_edge_strength:
                max_edge_strength = edge_strength
                best_line = y
        
        print(f"    Torso start detected at y={best_line}")
        return best_line
    
    def blend_three_parts(self, head_part, torso_part, legs_part):
        """Blend three parts together"""
        # Simple approach: use small blending zones
        blend_height = 15
        
        # Blend head with torso
        if head_part.shape[0] > blend_height and torso_part.shape[0] > blend_height:
            head_blend = head_part[-blend_height:, :].astype(np.float32)
            torso_blend_top = torso_part[:blend_height, :].astype(np.float32)
            
            weights = np.linspace(1, 0, blend_height).reshape(-1, 1, 1)
            blended_1 = head_blend * weights + torso_blend_top * (1 - weights)
            
            top_combined = np.vstack([
                head_part[:-blend_height, :],
                blended_1.astype(np.uint8),
                torso_part[blend_height:, :]
            ])
        else:
            top_combined = np.vstack([head_part, torso_part])
        
        # Blend top_combined with legs
        if top_combined.shape[0] > blend_height and legs_part.shape[0] > blend_height:
            top_blend = top_combined[-blend_height:, :].astype(np.float32)
            legs_blend = legs_part[:blend_height, :].astype(np.float32)
            
            weights = np.linspace(1, 0, blend_height).reshape(-1, 1, 1)
            blended_2 = top_blend * weights + legs_blend * (1 - weights)
            
            result = np.vstack([
                top_combined[:-blend_height, :],
                blended_2.astype(np.uint8),
                legs_part[blend_height:, :]
            ])
        else:
            result = np.vstack([top_combined, legs_part])
        
        return result
    
    def enforce_4_5_ratio(self, image):
        """Enforce 4:5 aspect ratio with padding"""
        height, width = image.shape[:2]
        target_ratio = 4.0 / 5.0
        current_ratio = width / height
        
        if abs(current_ratio - target_ratio) < 0.01:
            return image  # Already close to 4:5
        
        if current_ratio > target_ratio:
            # Too wide - add padding to height
            new_height = int(width / target_ratio)
            padding = new_height - height
            pad_top = padding // 2
            pad_bottom = padding - pad_top
            
            result = cv2.copyMakeBorder(
                image, pad_top, pad_bottom, 0, 0,
                cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
        else:
            # Too tall - add padding to width
            new_width = int(height * target_ratio)
            padding = new_width - width
            pad_left = padding // 2
            pad_right = padding - pad_left
            
            result = cv2.copyMakeBorder(
                image, 0, 0, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
        
        return result
    
    def process_pair(self, fragment, full_img):
        """Process a fragment-full image pair with 4:5 ratio"""
        result = self.smart_merge(fragment, full_img)
        return self.enforce_4_5_ratio(result)
    
    def process_triple(self, img1, img2, img3):
        """Process a 3-image group with 4:5 ratio"""
        # Analyze which image is most complete
        contents = []
        for img in [img1, img2, img3]:
            content = self.detect_content_regions(img)
            if content:
                contents.append(content['height'])
            else:
                contents.append(0)
        
        # Use the image with most content
        best_idx = np.argmax(contents)
        images = [img1, img2, img3]
        
        print(f"  Using image {best_idx + 1} as base (most complete)")
        result = images[best_idx]
        return self.enforce_4_5_ratio(result)


# Test function
def test_content_aware():
    """Test content-aware stitching with 1-9.jpg"""
    from puzzle_reconstructor import PuzzleReconstructor
    
    print("Testing content-aware stitching...")
    
    # Load images
    reconstructor = PuzzleReconstructor('content_aware_results')
    images = reconstructor.load_images('results')
    
    # Filter to 1-9.jpg
    test_images = [img for img in images if img['name'] in 
                   ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg', '7.jpg', '8.jpg', '9.jpg']]
    
    stitcher = ContentAwareStitcher('content_aware_results')
    
    # Test specific pairs
    print("\nTesting 2+3:")
    result_2_3 = stitcher.process_pair(
        test_images[1]['cv2_image'],  # 2.jpg
        test_images[2]['cv2_image']   # 3.jpg
    )
    cv2.imwrite('content_aware_results/test_2_3.jpg', result_2_3)
    
    print("\nTesting 4+5:")
    result_4_5 = stitcher.process_pair(
        test_images[3]['cv2_image'],  # 4.jpg
        test_images[4]['cv2_image']   # 5.jpg
    )
    cv2.imwrite('content_aware_results/test_4_5.jpg', result_4_5)
    
    print("\nTesting 5+6+7:")
    result_5_6_7 = stitcher.process_triple(
        test_images[4]['cv2_image'],  # 5.jpg
        test_images[5]['cv2_image'],  # 6.jpg
        test_images[6]['cv2_image']   # 7.jpg
    )
    cv2.imwrite('content_aware_results/test_5_6_7.jpg', result_5_6_7)
    
    print("\nContent-aware stitching complete!")


if __name__ == "__main__":
    test_content_aware()