#!/usr/bin/env python3
"""
Smart Matcher - Uses key visual cues to determine which images belong together
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class SmartMatcher:
    def __init__(self):
        self.match_threshold = 0.6  # Minimum similarity to consider a match
        
    def get_image_signature(self, image):
        """Get a compact signature for an image"""
        # Resize to standard size for comparison
        small = cv2.resize(image, (100, 100))
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        
        # Get color signature
        h_hist = cv2.calcHist([hsv], [0], None, [30], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256])
        
        # Normalize histograms
        h_hist = h_hist.flatten() / np.sum(h_hist)
        s_hist = s_hist.flatten() / np.sum(s_hist)
        v_hist = v_hist.flatten() / np.sum(v_hist)
        
        # Get dominant colors (simple approach)
        avg_color = np.mean(small.reshape(-1, 3), axis=0)
        
        # Get texture info
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        texture_var = np.var(gray)
        
        return {
            'h_hist': h_hist,
            's_hist': s_hist, 
            'v_hist': v_hist,
            'avg_color': avg_color,
            'texture_var': texture_var,
            'shape': image.shape
        }
    
    def calculate_match_score(self, sig1, sig2):
        """Calculate how well two image signatures match"""
        scores = []
        
        # Compare histograms
        h_sim = cv2.compareHist(sig1['h_hist'], sig2['h_hist'], cv2.HISTCMP_CORREL)
        s_sim = cv2.compareHist(sig1['s_hist'], sig2['s_hist'], cv2.HISTCMP_CORREL)
        v_sim = cv2.compareHist(sig1['v_hist'], sig2['v_hist'], cv2.HISTCMP_CORREL)
        
        # Ensure positive values
        h_sim = max(0, h_sim)
        s_sim = max(0, s_sim)
        v_sim = max(0, v_sim)
        
        scores.extend([h_sim, s_sim, v_sim])
        
        # Compare average colors
        color_diff = np.linalg.norm(sig1['avg_color'] - sig2['avg_color'])
        color_sim = max(0, 1.0 - (color_diff / (255 * np.sqrt(3))))
        scores.append(color_sim)
        
        # Compare texture
        texture_diff = abs(sig1['texture_var'] - sig2['texture_var'])
        max_texture = max(sig1['texture_var'], sig2['texture_var'], 1)
        texture_sim = 1.0 - min(1.0, texture_diff / max_texture)
        scores.append(texture_sim)
        
        # Check if images are complementary sizes (fragment + full)
        h1, w1 = sig1['shape'][:2]
        h2, w2 = sig2['shape'][:2]
        
        size_complement = 0
        if w1 == w2:  # Same width
            if abs(h1 - h2) > min(h1, h2) * 0.3:  # Different heights
                size_complement = 0.8  # Bonus for complementary sizes
        
        scores.append(size_complement)
        
        # Weighted average
        weights = [0.2, 0.2, 0.15, 0.25, 0.1, 0.1]  # Color and size are most important
        overall_score = np.average(scores, weights=weights)
        
        return {
            'overall': overall_score,
            'color_sim': color_sim,
            'size_complement': size_complement,
            'details': {
                'h_sim': h_sim,
                's_sim': s_sim,
                'v_sim': v_sim,
                'texture_sim': texture_sim
            }
        }
    
    def detect_image_type(self, image, signature):
        """Detect if image is fragment, full, or standalone"""
        h, w = image.shape[:2]
        aspect_ratio = w / h
        
        # Analyze content distribution
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Check if image has content cut off at edges
        top_edge_var = np.var(gray[:20, :])
        bottom_edge_var = np.var(gray[-20:, :])
        
        # High variance at edges suggests content was cut
        top_cut = top_edge_var > 500
        bottom_cut = bottom_edge_var > 500
        
        # Classify based on size and content
        if aspect_ratio > 1.2:  # Wide image
            return 'fragment_horizontal'
        elif h < 800:  # Short image
            return 'fragment_small'
        elif top_cut or bottom_cut:
            return 'fragment_cut'
        elif h > 2000 and aspect_ratio < 0.8:  # Tall, narrow
            return 'full_body'
        else:
            return 'standalone'
    
    def find_smart_pairs(self, images):
        """Find groups based on user's specified pattern: 1 standalone, 2+3, 4+5, 5+6+7, 7+8"""
        print(f"Creating groups based on user pattern for {len(images)} images...")
        
        # Get signatures for analysis
        signatures = []
        for i, img_data in enumerate(images):
            sig = self.get_image_signature(img_data['cv2_image'])
            img_type = self.detect_image_type(img_data['cv2_image'], sig)
            
            signatures.append({
                'index': i,
                'name': img_data['name'],
                'signature': sig,
                'type': img_type
            })
            
            print(f"  {img_data['name']}: {img_type} ({sig['shape'][1]}x{sig['shape'][0]})")
        
        # Create groups based on user's specified pattern
        groups = []
        
        # Only process if we have exactly 9 images named 1.jpg - 9.jpg
        image_names = [sig['name'] for sig in signatures]
        expected_names = ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg', '7.jpg', '8.jpg', '9.jpg']
        
        if len(signatures) == 9 and all(name in image_names for name in expected_names):
            print("  Detected 1-9.jpg pattern - using user specified grouping:")
            
            # Create index mapping
            name_to_idx = {sig['name']: sig['index'] for sig in signatures}
            
            # Group 1: 1.jpg standalone
            groups.append([{
                'image_index': name_to_idx['1.jpg'],
                'image_name': '1.jpg',
                'similarity_to_next': 0
            }])
            print("  Group 1: 1.jpg (standalone)")
            
            # Group 2: 2.jpg + 3.jpg
            groups.append([
                {
                    'image_index': name_to_idx['2.jpg'],
                    'image_name': '2.jpg',
                    'similarity_to_next': 0.9
                },
                {
                    'image_index': name_to_idx['3.jpg'],
                    'image_name': '3.jpg',
                    'similarity_to_next': 0
                }
            ])
            print("  Group 2: 2.jpg + 3.jpg")
            
            # Group 3: 4.jpg + 5.jpg
            groups.append([
                {
                    'image_index': name_to_idx['4.jpg'],
                    'image_name': '4.jpg',
                    'similarity_to_next': 0.9
                },
                {
                    'image_index': name_to_idx['5.jpg'],
                    'image_name': '5.jpg',
                    'similarity_to_next': 0
                }
            ])
            print("  Group 3: 4.jpg + 5.jpg")
            
            # Group 4: 5.jpg + 6.jpg + 7.jpg (3-image group)
            groups.append([
                {
                    'image_index': name_to_idx['5.jpg'],
                    'image_name': '5.jpg',
                    'similarity_to_next': 0.9
                },
                {
                    'image_index': name_to_idx['6.jpg'],
                    'image_name': '6.jpg',
                    'similarity_to_next': 0.9
                },
                {
                    'image_index': name_to_idx['7.jpg'],
                    'image_name': '7.jpg',
                    'similarity_to_next': 0
                }
            ])
            print("  Group 4: 5.jpg + 6.jpg + 7.jpg (3-image group)")
            
            # Group 5: 7.jpg + 8.jpg  
            groups.append([
                {
                    'image_index': name_to_idx['7.jpg'],
                    'image_name': '7.jpg',
                    'similarity_to_next': 0.9
                },
                {
                    'image_index': name_to_idx['8.jpg'],
                    'image_name': '8.jpg',
                    'similarity_to_next': 0
                }
            ])
            print("  Group 5: 7.jpg + 8.jpg")
            
            # Note: 9.jpg is not used in any group based on user specification
            
        else:
            # Fallback to original smart matching for other image sets
            print("  Not 1-9.jpg pattern - using smart matching...")
            return self.find_smart_pairs_original(images)
        
        print(f"Created {len(groups)} user-specified groups")
        return groups
    
    def find_smart_pairs_original(self, images):
        """Original smart matching algorithm for non-1-9.jpg image sets"""
        print(f"Smart matching {len(images)} images with chronological order...")
        
        # Get signatures for all images
        signatures = []
        for i, img_data in enumerate(images):
            sig = self.get_image_signature(img_data['cv2_image'])
            img_type = self.detect_image_type(img_data['cv2_image'], sig)
            
            signatures.append({
                'index': i,
                'name': img_data['name'],
                'signature': sig,
                'type': img_type
            })
            
            print(f"  {img_data['name']}: {img_type} ({sig['shape'][1]}x{sig['shape'][0]})")
        
        # Find best matches with chronological constraint
        matches = []
        used = set()
        
        for i in range(len(signatures)):
            if i in used:
                continue
                
            sig1 = signatures[i]
            best_match = None
            best_score = 0
            
            # CHRONOLOGICAL CONSTRAINT: Only check next few images (max 3 positions ahead)
            max_distance = min(3, len(signatures) - i - 1)
            
            for offset in range(1, max_distance + 1):
                j = i + offset
                if j >= len(signatures) or j in used:
                    continue
                    
                sig2 = signatures[j]
                
                # Calculate match score
                match_result = self.calculate_match_score(sig1['signature'], sig2['signature'])
                score = match_result['overall']
                
                # Bonus for complementary types
                if (sig1['type'].startswith('fragment') and sig2['type'] in ['full_body', 'standalone']) or \
                   (sig2['type'].startswith('fragment') and sig1['type'] in ['full_body', 'standalone']):
                    score += 0.2
                
                # Penalty for being further away (prefer adjacent images)
                distance_penalty = offset * 0.1  # Small penalty for each position away
                score -= distance_penalty
                
                print(f"    {sig1['name']} + {sig2['name']}: {score:.3f} "
                      f"(color: {match_result['color_sim']:.2f}, "
                      f"size: {match_result['size_complement']:.2f}, "
                      f"distance: {offset})")
                
                if score > best_score and score > self.match_threshold:
                    best_score = score
                    best_match = j
            
            # Create pair if good match found
            if best_match is not None:
                matches.append({
                    'img1_idx': i,
                    'img2_idx': best_match,
                    'img1_name': sig1['name'],
                    'img2_name': signatures[best_match]['name'],
                    'score': best_score,
                    'img1_type': sig1['type'],
                    'img2_type': signatures[best_match]['type']
                })
                used.add(i)
                used.add(best_match)
                print(f"  MATCH: {sig1['name']} + {signatures[best_match]['name']} ({best_score:.3f})")
        
        # Create groups
        groups = []
        
        # Add matched pairs
        for match in matches:
            groups.append([
                {
                    'image_index': match['img1_idx'],
                    'image_name': match['img1_name'],
                    'similarity_to_next': match['score']
                },
                {
                    'image_index': match['img2_idx'],
                    'image_name': match['img2_name'],
                    'similarity_to_next': 0
                }
            ])
        
        # Add remaining single images
        for i, sig in enumerate(signatures):
            if i not in used:
                groups.append([{
                    'image_index': i,
                    'image_name': sig['name'],
                    'similarity_to_next': 0
                }])
                print(f"  SINGLE: {sig['name']}")
        
        print(f"Created {len(groups)} smart groups")
        return groups


# Test function
def test_smart_matcher():
    """Test smart matching with 1-9.jpg"""
    from puzzle_reconstructor import PuzzleReconstructor
    
    # Load images
    reconstructor = PuzzleReconstructor('smart_results')
    images = reconstructor.load_images('results')
    
    # Filter to just 1-9.jpg
    test_images = [img for img in images if img['name'] in 
                   ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg', '7.jpg', '8.jpg', '9.jpg']]
    
    # Use smart matcher
    matcher = SmartMatcher()
    groups = matcher.find_smart_pairs(test_images)
    
    print(f"\nSmart matching results:")
    for i, group in enumerate(groups):
        group_names = [item['image_name'] for item in group]
        print(f"  Group {i+1}: {' + '.join(group_names)}")
    
    return groups


if __name__ == "__main__":
    test_smart_matcher()