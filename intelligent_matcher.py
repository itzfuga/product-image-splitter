#!/usr/bin/env python3
"""
Intelligent Image Matcher - Uses CV to determine which images belong together
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import itertools


class IntelligentMatcher:
    def __init__(self):
        self.similarity_threshold = 0.3  # Minimum similarity to consider a match
        
    def extract_features(self, image):
        """Extract features from an image for matching"""
        # Convert to different color spaces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Extract features
        features = {
            'color_hist': cv2.calcHist([hsv], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256]),
            'texture': self.get_texture_features(gray),
            'edges': self.get_edge_features(gray),
            'dominant_colors': self.get_dominant_colors(image),
            'clothing_regions': self.detect_clothing_regions(image),
            'skin_regions': self.detect_skin_regions(image)
        }
        
        return features
    
    def get_texture_features(self, gray):
        """Extract texture features using LBP-like approach"""
        # Calculate local binary patterns
        h, w = gray.shape
        texture_map = np.zeros_like(gray)
        
        # Simple texture analysis
        kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
        texture_map = cv2.filter2D(gray, -1, kernel)
        
        # Get texture statistics
        return {
            'mean': np.mean(texture_map),
            'std': np.std(texture_map),
            'histogram': np.histogram(texture_map, bins=50)[0]
        }
    
    def get_edge_features(self, gray):
        """Extract edge features"""
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return {
            'edge_density': np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]),
            'num_contours': len(contours),
            'avg_contour_area': np.mean([cv2.contourArea(c) for c in contours]) if contours else 0
        }
    
    def get_dominant_colors(self, image):
        """Extract dominant colors using K-means"""
        # Reshape image to be a list of pixels
        pixels = image.reshape(-1, 3).astype(np.float32)
        
        # Use K-means to find dominant colors
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        k = 5  # Number of dominant colors
        
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Calculate color percentages
        unique, counts = np.unique(labels, return_counts=True)
        percentages = counts / len(labels)
        
        return {
            'colors': centers.astype(int),
            'percentages': percentages
        }
    
    def detect_clothing_regions(self, image):
        """Detect clothing regions (non-skin areas with texture)"""
        # Convert to HSV for better color detection
        hsv = image.copy()
        
        # Define skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create mask for skin
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Clothing is everything that's not skin
        clothing_mask = cv2.bitwise_not(skin_mask)
        
        # Calculate clothing area percentage
        clothing_area = np.sum(clothing_mask > 0) / (clothing_mask.shape[0] * clothing_mask.shape[1])
        
        return {
            'area_percentage': clothing_area,
            'mask': clothing_mask
        }
    
    def detect_skin_regions(self, image):
        """Detect skin regions (face, hands, etc.)"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Skin color range
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Find contours in skin regions
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get largest skin regions (likely face/hands)
        if contours:
            skin_areas = [cv2.contourArea(c) for c in contours]
            largest_skin_area = max(skin_areas)
            total_skin_area = sum(skin_areas)
        else:
            largest_skin_area = 0
            total_skin_area = 0
        
        return {
            'total_area': total_skin_area,
            'largest_area': largest_skin_area,
            'num_regions': len(contours)
        }
    
    def calculate_visual_similarity(self, features1, features2):
        """Calculate how visually similar two images are"""
        similarity_scores = []
        
        # 1. Color histogram similarity
        hist_sim = cv2.compareHist(features1['color_hist'], features2['color_hist'], cv2.HISTCMP_CORREL)
        similarity_scores.append(max(0, hist_sim))
        
        # 2. Dominant colors similarity
        color_sim = self.compare_dominant_colors(features1['dominant_colors'], features2['dominant_colors'])
        similarity_scores.append(color_sim)
        
        # 3. Texture similarity
        texture_sim = self.compare_textures(features1['texture'], features2['texture'])
        similarity_scores.append(texture_sim)
        
        # 4. Clothing similarity
        clothing_sim = abs(features1['clothing_regions']['area_percentage'] - 
                          features2['clothing_regions']['area_percentage'])
        clothing_sim = 1.0 - min(1.0, clothing_sim)  # Convert difference to similarity
        similarity_scores.append(clothing_sim)
        
        # Weighted average
        weights = [0.3, 0.3, 0.2, 0.2]  # Color and dominant colors are most important
        overall_similarity = np.average(similarity_scores, weights=weights)
        
        return {
            'overall': overall_similarity,
            'color_hist': hist_sim,
            'dominant_colors': color_sim,
            'texture': texture_sim,
            'clothing': clothing_sim
        }
    
    def compare_dominant_colors(self, colors1, colors2):
        """Compare dominant colors between two images"""
        c1 = colors1['colors']
        c2 = colors2['colors']
        p1 = colors1['percentages']
        p2 = colors2['percentages']
        
        # Find best matches between colors
        similarities = []
        
        for i, (color1, perc1) in enumerate(zip(c1, p1)):
            best_match = 0
            for j, (color2, perc2) in enumerate(zip(c2, p2)):
                # Calculate color distance
                color_dist = np.linalg.norm(color1 - color2)
                # Convert to similarity (0-1, where 1 is identical)
                color_sim = max(0, 1.0 - (color_dist / (255 * np.sqrt(3))))
                
                # Weight by percentage importance
                weighted_sim = color_sim * min(perc1, perc2)
                best_match = max(best_match, weighted_sim)
            
            similarities.append(best_match)
        
        return np.mean(similarities)
    
    def compare_textures(self, tex1, tex2):
        """Compare texture features"""
        # Compare texture statistics
        mean_diff = abs(tex1['mean'] - tex2['mean']) / 255.0
        std_diff = abs(tex1['std'] - tex2['std']) / 255.0
        
        # Compare histograms
        hist_sim = cv2.compareHist(tex1['histogram'].astype(np.float32), 
                                  tex2['histogram'].astype(np.float32), 
                                  cv2.HISTCMP_CORREL)
        hist_sim = max(0, hist_sim)
        
        # Combine metrics
        texture_similarity = (1.0 - mean_diff) * 0.3 + (1.0 - std_diff) * 0.3 + hist_sim * 0.4
        return max(0, texture_similarity)
    
    def detect_complementary_regions(self, img1, img2):
        """Detect if images have complementary regions (one has what the other lacks)"""
        features1 = self.extract_features(img1)
        features2 = self.extract_features(img2)
        
        # Check if one image has more skin (face) and other has more clothing
        skin1 = features1['skin_regions']['total_area']
        skin2 = features2['skin_regions']['total_area']
        
        clothing1 = features1['clothing_regions']['area_percentage']
        clothing2 = features2['clothing_regions']['area_percentage']
        
        # If one has significantly more skin regions, they might complement each other
        skin_complement = abs(skin1 - skin2) > (max(skin1, skin2) * 0.5)
        
        # If they have similar clothing but different proportions, they might complement
        clothing_similar = abs(clothing1 - clothing2) < 0.3
        
        return {
            'skin_complement': skin_complement,
            'clothing_similar': clothing_similar,
            'complementary_score': 0.7 if skin_complement and clothing_similar else 0.2
        }
    
    def find_best_matches(self, images):
        """Find the best matching pairs from a list of images"""
        print(f"Analyzing {len(images)} images for intelligent matching...")
        
        # Extract features for all images
        all_features = []
        for i, img_data in enumerate(images):
            print(f"  Extracting features from image {i+1}...")
            features = self.extract_features(img_data['cv2_image'])
            all_features.append({
                'features': features,
                'index': i,
                'name': img_data['name']
            })
        
        # Find all possible pairs and calculate similarities
        matches = []
        
        for i in range(len(all_features)):
            for j in range(i+1, len(all_features)):
                img1_data = all_features[i]
                img2_data = all_features[j]
                
                # Calculate visual similarity
                similarity = self.calculate_visual_similarity(
                    img1_data['features'], 
                    img2_data['features']
                )
                
                # Check for complementary regions
                complement = self.detect_complementary_regions(
                    images[i]['cv2_image'],
                    images[j]['cv2_image']
                )
                
                # Combined score
                combined_score = (similarity['overall'] * 0.7 + 
                                complement['complementary_score'] * 0.3)
                
                if combined_score > self.similarity_threshold:
                    matches.append({
                        'img1_idx': i,
                        'img2_idx': j,
                        'img1_name': img1_data['name'],
                        'img2_name': img2_data['name'],
                        'similarity': similarity,
                        'complement': complement,
                        'combined_score': combined_score
                    })
                    
                    print(f"  Match found: {img1_data['name']} + {img2_data['name']} "
                          f"(score: {combined_score:.3f})")
        
        # Sort matches by combined score
        matches.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return matches
    
    def create_optimal_groups(self, images):
        """Create optimal groups from intelligent matching"""
        matches = self.find_best_matches(images)
        
        groups = []
        used_images = set()
        
        # Process matches in order of quality
        for match in matches:
            img1_idx = match['img1_idx']
            img2_idx = match['img2_idx']
            
            # Skip if either image is already used
            if img1_idx in used_images or img2_idx in used_images:
                continue
            
            # Create group
            groups.append([
                {
                    'image_index': img1_idx,
                    'image_name': match['img1_name'],
                    'similarity_to_next': match['combined_score']
                },
                {
                    'image_index': img2_idx, 
                    'image_name': match['img2_name'],
                    'similarity_to_next': 0
                }
            ])
            
            used_images.add(img1_idx)
            used_images.add(img2_idx)
            
            print(f"Created group: {match['img1_name']} + {match['img2_name']}")
        
        # Add remaining single images
        for i, img_data in enumerate(images):
            if i not in used_images:
                groups.append([{
                    'image_index': i,
                    'image_name': img_data['name'],
                    'similarity_to_next': 0
                }])
                print(f"Single image: {img_data['name']}")
        
        print(f"Created {len(groups)} intelligent groups")
        return groups


# Test function
def test_intelligent_matcher():
    """Test intelligent matching with 1-9.jpg"""
    import os
    from puzzle_reconstructor import PuzzleReconstructor
    
    # Load images
    reconstructor = PuzzleReconstructor('intelligent_results')
    images = reconstructor.load_images('results')
    
    # Filter to just 1-9.jpg
    test_images = [img for img in images if img['name'] in 
                   ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg', '7.jpg', '8.jpg', '9.jpg']]
    
    # Use intelligent matcher
    matcher = IntelligentMatcher()
    groups = matcher.create_optimal_groups(test_images)
    
    print(f"\nIntelligent matching results:")
    for i, group in enumerate(groups):
        group_names = [item['image_name'] for item in group]
        print(f"  Group {i+1}: {' + '.join(group_names)}")
    
    return groups


if __name__ == "__main__":
    test_intelligent_matcher()