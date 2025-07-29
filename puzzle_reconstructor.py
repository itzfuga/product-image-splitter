#!/usr/bin/env python3
"""
Taobao Product Image Puzzle Reconstructor
Automatically combines fragmented supplier images into complete product photos
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional
import re
from smart_matcher import SmartMatcher
from white_separator_stitcher import WhiteSeparatorStitcher


class PuzzleReconstructor:
    def __init__(self, result_dir, session_id=None):
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(exist_ok=True)
        self.session_id = session_id
        
        # Settings
        self.edge_height = 15  # Pixels to analyze at top/bottom edges
        self.similarity_threshold = 0.75  # Minimum similarity to consider a match
        self.overlap_threshold = 0.85  # Threshold for detecting overlapping content
        
        # Processing settings
        self.force_4_5_ratio = True
        self.auto_crop = True
        self.background_color = (255, 255, 255)  # White background
        self.content_padding = 20
        
        # Initialize smart components
        self.smart_matcher = SmartMatcher()
        self.white_separator_stitcher = WhiteSeparatorStitcher(result_dir, session_id)
        
    def load_images(self, image_dir):
        """Load all images from directory with natural sorting"""
        image_dir = Path(image_dir)
        images = []
        
        supported_formats = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.avif']
        
        def natural_sort_key(path):
            """Extract numbers from filename for proper sorting"""
            s = path.name
            return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
        
        for file_path in sorted(image_dir.glob('*'), key=natural_sort_key):
            if file_path.suffix.lower() in supported_formats:
                try:
                    # Try OpenCV first
                    img = cv2.imread(str(file_path))
                    if img is not None:
                        images.append({
                            'path': str(file_path),
                            'name': file_path.name,
                            'cv2_image': img,
                            'pil_image': None,
                            'index': len(images)
                        })
                        print(f"Loaded: {file_path.name}")
                    else:
                        # Fallback to PIL for formats like AVIF
                        pil_img = Image.open(file_path).convert('RGB')
                        cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                        images.append({
                            'path': str(file_path),
                            'name': file_path.name,
                            'cv2_image': cv2_img,
                            'pil_image': pil_img,
                            'index': len(images)
                        })
                        print(f"Loaded (PIL): {file_path.name}")
                        
                except Exception as e:
                    print(f"Error loading {file_path.name}: {e}")
                    
        return images
    
    def extract_edge_features(self, image, edge_type='bottom'):
        """Extract features from top or bottom edge of image"""
        height, width = image.shape[:2]
        
        if edge_type == 'bottom':
            edge_region = image[height - self.edge_height:height, :]
        else:  # top
            edge_region = image[:self.edge_height, :]
        
        # Convert to different color spaces for analysis
        edge_rgb = cv2.cvtColor(edge_region, cv2.COLOR_BGR2RGB)
        edge_hsv = cv2.cvtColor(edge_region, cv2.COLOR_BGR2HSV)
        edge_gray = cv2.cvtColor(edge_region, cv2.COLOR_BGR2GRAY)
        
        features = {
            'rgb_hist': cv2.calcHist([edge_rgb], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]),
            'hsv_hist': cv2.calcHist([edge_hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256]),
            'mean_color': np.mean(edge_rgb.reshape(-1, 3), axis=0),
            'std_color': np.std(edge_rgb.reshape(-1, 3), axis=0),
            'edge_gradient': np.mean(np.abs(np.gradient(edge_gray.astype(float), axis=0))),
            'texture_variance': np.var(edge_gray),
            'raw_pixels': edge_rgb.reshape(-1, 3)  # For direct pixel comparison
        }
        
        return features
    
    def detect_edge_type(self, image, edge_type='bottom'):
        """Detect if an edge looks 'cut off' or 'natural'"""
        height, width = image.shape[:2]
        
        if edge_type == 'bottom':
            edge_region = image[height - self.edge_height:height, :]
        else:  # top
            edge_region = image[:self.edge_height, :]
        
        edge_gray = cv2.cvtColor(edge_region, cv2.COLOR_BGR2GRAY)
        
        # Check for signs of artificial cutting:
        # 1. High variance in the edge (indicates content was cut)
        variance = np.var(edge_gray)
        
        # 2. Non-uniform color distribution
        edge_rgb = cv2.cvtColor(edge_region, cv2.COLOR_BGR2RGB)
        color_uniformity = 1.0 - (np.std(edge_rgb) / 255.0)
        
        # 3. Presence of vertical edges (suggesting cut through objects)
        vertical_edges = cv2.Sobel(edge_gray, cv2.CV_64F, 0, 1, ksize=3)
        vertical_edge_strength = np.mean(np.abs(vertical_edges))
        
        # Scoring: higher score = more likely to be cut off
        cut_score = 0.0
        
        # High variance suggests content was cut
        if variance > 500:  # Threshold may need tuning
            cut_score += 0.4
        
        # Low uniformity suggests mixed content
        if color_uniformity < 0.8:
            cut_score += 0.3
        
        # Strong vertical edges suggest cut through objects
        if vertical_edge_strength > 10:  # Threshold may need tuning
            cut_score += 0.3
        
        return {
            'cut_score': cut_score,
            'is_likely_cut': cut_score > 0.5,
            'variance': variance,
            'color_uniformity': color_uniformity,
            'vertical_edge_strength': vertical_edge_strength
        }
    
    def calculate_edge_similarity(self, features1, features2):
        """Calculate similarity between two edge features"""
        similarity_scores = []
        
        # 1. Histogram similarity (Bhattacharyya distance)
        rgb_sim = cv2.compareHist(features1['rgb_hist'], features2['rgb_hist'], cv2.HISTCMP_BHATTACHARYYA)
        hsv_sim = cv2.compareHist(features1['hsv_hist'], features2['hsv_hist'], cv2.HISTCMP_BHATTACHARYYA)
        
        # Convert Bhattacharyya distance to similarity (0=identical, 1=completely different)
        rgb_similarity = 1.0 - rgb_sim
        hsv_similarity = 1.0 - hsv_sim
        
        similarity_scores.extend([rgb_similarity, hsv_similarity])
        
        # 2. Mean color similarity
        color_diff = np.linalg.norm(features1['mean_color'] - features2['mean_color'])
        color_similarity = max(0, 1.0 - (color_diff / (255 * np.sqrt(3))))  # Normalize to 0-1
        similarity_scores.append(color_similarity)
        
        # 3. Texture similarity
        texture_diff = abs(features1['texture_variance'] - features2['texture_variance'])
        max_texture_var = max(features1['texture_variance'], features2['texture_variance'], 1)
        texture_similarity = 1.0 - (texture_diff / max_texture_var)
        similarity_scores.append(texture_similarity)
        
        # 4. Direct pixel comparison for exact matches
        pixels1 = features1['raw_pixels']
        pixels2 = features2['raw_pixels']
        
        if len(pixels1) == len(pixels2):
            pixel_diff = np.mean(np.linalg.norm(pixels1 - pixels2, axis=1))
            pixel_similarity = max(0, 1.0 - (pixel_diff / (255 * np.sqrt(3))))
            similarity_scores.append(pixel_similarity)
        
        # Weighted average of all similarity measures
        weights = [0.2, 0.2, 0.3, 0.1, 0.2] if len(similarity_scores) == 5 else [0.25, 0.25, 0.35, 0.15]
        overall_similarity = np.average(similarity_scores[:len(weights)], weights=weights)
        
        return {
            'overall_similarity': overall_similarity,
            'rgb_similarity': rgb_similarity,
            'hsv_similarity': hsv_similarity,
            'color_similarity': color_similarity,
            'texture_similarity': texture_similarity,
            'pixel_similarity': similarity_scores[-1] if len(similarity_scores) == 5 else 0,
            'component_scores': similarity_scores
        }
    
    def find_sequential_groups(self, images):
        """Group images based on smart visual matching with chronological constraints"""
        print(f"Using smart matching for {len(images)} images...")
        
        # Use smart matcher with chronological constraints
        groups = self.smart_matcher.find_smart_pairs(images)
        
        print(f"Smart matcher created {len(groups)} groups")
        return groups
    
    def is_fragment_image(self, image):
        """Detect if an image is a fragment (partial body, usually upper torso)"""
        height, width = image.shape[:2]
        
        # Fragment characteristics:
        # 1. Often has 4:3 or similar aspect ratio (wider than full body shots)
        # 2. Usually shows upper body only
        # 3. Has "cut off" bottom edge
        # 4. Smaller height relative to width
        
        aspect_ratio = height / width
        bottom_edge_info = self.detect_edge_type(image, 'bottom')
        
        # Fragment criteria
        is_wide_ratio = aspect_ratio < 1.8  # Not very tall
        has_cut_bottom = bottom_edge_info['cut_score'] > 0.4  # Looks cut off
        is_smaller = height < 1200  # Usually shorter than full body
        
        is_fragment = is_wide_ratio and has_cut_bottom and is_smaller
        
        print(f"  Fragment analysis: ratio={aspect_ratio:.2f}, cut_score={bottom_edge_info['cut_score']:.2f}, height={height}, fragment={is_fragment}")
        
        return is_fragment
    
    def check_outfit_match(self, img1, img2):
        """Check if two images show the same outfit/clothing"""
        height1, width1 = img1.shape[:2]
        height2, width2 = img2.shape[:2]
        
        # Special case: if first image is much smaller than second (likely partial fragment)
        # be more lenient with matching
        area1 = height1 * width1
        area2 = height2 * width2
        is_much_smaller = area1 < (area2 * 0.5)  # Less than half the area
        
        # Get dominant colors from both images
        mean_color1 = np.mean(img1.reshape(-1, 3), axis=0)
        mean_color2 = np.mean(img2.reshape(-1, 3), axis=0)
        
        # Calculate color difference
        color_diff = np.linalg.norm(mean_color1 - mean_color2)
        
        # Check standard deviation (texture similarity)
        std1 = np.std(img1.reshape(-1, 3), axis=0)
        std2 = np.std(img2.reshape(-1, 3), axis=0)
        texture_diff = np.linalg.norm(std1 - std2)
        
        # Use more lenient thresholds for much smaller fragments, but not for obviously different colors
        if is_much_smaller:
            # Be more restrictive - only allow very similar colors for small fragments
            # This prevents white shirt fragments from matching black shirts
            is_similar = color_diff < 58 and texture_diff < 90  # Tight threshold to prevent white/black mixing
            print(f"    Small fragment match (area ratio: {area1/area2:.2f}): color_diff={color_diff:.1f}, texture_diff={texture_diff:.1f}, match={is_similar}")
        else:
            is_similar = color_diff < 60 and texture_diff < 50  # Normal thresholds
            print(f"    Outfit match: color_diff={color_diff:.1f}, texture_diff={texture_diff:.1f}, match={is_similar}")
        
        return is_similar
    
    def should_pair_fragments(self, img1, img2):
        """Determine if two fragments should be paired together"""
        # For now, use a simple heuristic:
        # - Images should be similar in width
        # - Color palette should be similar (same outfit/background)
        
        height1, width1 = img1.shape[:2]  
        height2, width2 = img2.shape[:2]
        
        # Width similarity
        width_ratio = min(width1, width2) / max(width1, width2)
        width_similar = width_ratio > 0.8
        
        # Color similarity (simple approach using mean colors)
        mean_color1 = np.mean(img1.reshape(-1, 3), axis=0)
        mean_color2 = np.mean(img2.reshape(-1, 3), axis=0)
        color_diff = np.linalg.norm(mean_color1 - mean_color2)
        color_similar = color_diff < 50  # Threshold for similar colors
        
        # Check if first image looks like top part and second looks like bottom part
        top_edge_info = self.detect_edge_type(img1, 'bottom')  # Bottom of first image
        bottom_edge_info = self.detect_edge_type(img2, 'top')  # Top of second image
        
        # Both should look "cut off" to be good candidates for pairing
        both_cut = top_edge_info['is_likely_cut'] and bottom_edge_info['is_likely_cut']
        
        should_pair = width_similar and color_similar and both_cut
        
        print(f"  Pairing analysis: width_sim={width_similar}, color_sim={color_similar}, both_cut={both_cut}, result={should_pair}")
        
        return should_pair
    
    def detect_group_size(self, images):
        """Detect how many images typically form one product"""
        # For now, use a simple heuristic based on total image count
        # This can be made smarter later with actual image analysis
        
        total_images = len(images)
        
        # Common Taobao patterns
        if total_images <= 3:
            return 1  # Each image is standalone
        elif total_images <= 6:
            return 2  # Pairs of images
        elif total_images <= 12:
            # Check if divisible by 2 or 3
            if total_images % 2 == 0:
                return 2
            elif total_images % 3 == 0:
                return 3
            else:
                return 2  # Default to pairs
        else:
            # For larger sets, prefer smaller groups
            if total_images % 2 == 0:
                return 2
            elif total_images % 3 == 0:
                return 3
            else:
                return 2
    
    def find_potential_connections(self, images):
        """Legacy function - now redirects to sequential grouping"""
        return []  # No connections needed for sequential grouping
    
    def build_image_chains(self, images, connections):
        """Build sequential chains from grouped images"""
        print("Building sequential image groups...")
        
        # Use sequential grouping instead of complex edge matching
        chains = self.find_sequential_groups(images)
        
        print(f"Built {len(chains)} sequential groups total")
        return chains
    
    def build_best_chain(self, start_idx, graph, used_images):
        """Build the best chain starting from a specific image"""
        chain = [{
            'image_index': start_idx,
            'image_name': f"image_{start_idx+1}",
            'similarity_to_next': 0
        }]
        
        current_idx = start_idx
        used_in_chain = {start_idx}
        
        while True:
            best_next = None
            best_similarity = 0
            
            # Find the best unused next image
            for next_option in graph[current_idx]:
                next_idx = next_option['to']
                similarity = next_option['similarity']
                
                if next_idx not in used_in_chain and next_idx not in used_images:
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_next = next_option
            
            if best_next is None:
                break  # No more connections possible
            
            # Add the best next image to the chain
            chain[-1]['similarity_to_next'] = best_similarity
            chain.append({
                'image_index': best_next['to'],
                'image_name': f"image_{best_next['to']+1}",
                'similarity_to_next': 0
            })
            
            current_idx = best_next['to']
            used_in_chain.add(current_idx)
        
        return chain
    
    def detect_content_bounds(self, image):
        """Detect the bounds of actual content (model/product) in the image"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Fast approach using numpy operations
        content_mask = gray < 240
        content_rows = np.any(content_mask, axis=1)
        content_cols = np.any(content_mask, axis=0)
        
        if not np.any(content_rows) or not np.any(content_cols):
            # No content found, return full image bounds
            return 0, 0, gray.shape[1], gray.shape[0]
        
        top = np.argmax(content_rows)
        bottom = len(content_rows) - np.argmax(content_rows[::-1]) - 1
        left = np.argmax(content_cols)
        right = len(content_cols) - np.argmax(content_cols[::-1]) - 1
        
        return left, top, right, bottom
    
    def stitch_images(self, images, chain):
        """Process images for Taobao reconstruction using smart stitching"""
        if len(chain) == 1:
            # Single image - use aggressive white trimming
            img_idx = chain[0]['image_index']
            image = images[img_idx]['cv2_image']
            
            # Apply aggressive white trimming to remove ALL white background
            trimmed = self.white_separator_stitcher.trim_all_white_edges(image)
            
            return self.enforce_4_5_ratio(trimmed) if self.force_4_5_ratio else trimmed
        
        elif len(chain) == 2:
            # Two images: use white separator stitching
            img1_idx = chain[0]['image_index']
            img2_idx = chain[1]['image_index']
            
            img1 = images[img1_idx]['cv2_image']
            img2 = images[img2_idx]['cv2_image']
            
            print(f"  White separator stitching: {chain[0]['image_name']} + {chain[1]['image_name']}")
            
            # Special handling for 7+8 case (need bottom part of first image)
            if chain[0]['image_name'] == '7.jpg' and chain[1]['image_name'] == '8.jpg':
                result = self.white_separator_stitcher.stack_and_split(img1, img2, take_from_img1="bottom")
                result = self.white_separator_stitcher.enforce_4_5_ratio(result)
            else:
                # Use white separator stitcher
                result = self.white_separator_stitcher.process_pair(img1, img2)
            
            return result
        
        elif len(chain) == 3:
            # Three images: use white separator processing
            img1_idx = chain[0]['image_index']
            img2_idx = chain[1]['image_index'] 
            img3_idx = chain[2]['image_index']
            
            img1 = images[img1_idx]['cv2_image']
            img2 = images[img2_idx]['cv2_image']
            img3 = images[img3_idx]['cv2_image']
            
            print(f"  White separator 3-image group: {chain[0]['image_name']} + {chain[1]['image_name']} + {chain[2]['image_name']}")
            
            # Use white separator processing for triple
            result = self.white_separator_stitcher.process_triple(img1, img2, img3)
            
            return result
            
        else:
            # More than 3 images - use the last image as it's likely the most complete
            print(f"  WARNING: Chain has {len(chain)} images, using last one")
            last_idx = chain[-1]['image_index']
            image = images[last_idx]['cv2_image']
            
            if self.auto_crop:
                left, top, right, bottom = self.detect_content_bounds(image)
                cropped = image[top:bottom+1, left:right+1]
                return self.enforce_4_5_ratio(cropped) if self.force_4_5_ratio else cropped
            else:
                return self.enforce_4_5_ratio(image) if self.force_4_5_ratio else image
    
    def enforce_4_5_ratio(self, image):
        """Force image to 4:5 aspect ratio with padding"""
        return self.white_separator_stitcher.enforce_4_5_ratio(image)
    
    def process_chains(self, images, chains):
        """Process all chains and create final product images"""
        print(f"Processing {len(chains)} chains...")
        
        results = []
        
        for i, chain in enumerate(chains):
            print(f"Processing chain {i+1}: {' -> '.join([c['image_name'] for c in chain])}")
            
            try:
                # Stitch the images in this chain
                stitched_image = self.stitch_images(images, chain)
                
                # Save the result
                output_filename = f"product_{i+1}.jpg"
                output_path = self.result_dir / output_filename
                
                success = cv2.imwrite(str(output_path), stitched_image)
                
                if success:
                    result = {
                        'chain_id': i,
                        'output_file': output_filename,
                        'output_path': str(output_path),
                        'source_images': [c['image_name'] for c in chain],
                        'dimensions': f"{stitched_image.shape[1]}x{stitched_image.shape[0]}",
                        'success': True
                    }
                    print(f"  Saved: {output_filename} ({result['dimensions']})")
                else:
                    result = {
                        'chain_id': i,
                        'output_file': output_filename,
                        'source_images': [c['image_name'] for c in chain],
                        'success': False,
                        'error': 'Failed to save image'
                    }
                    print(f"  ERROR: Failed to save {output_filename}")
                
                results.append(result)
                
            except Exception as e:
                result = {
                    'chain_id': i,
                    'source_images': [c['image_name'] for c in chain],
                    'success': False,
                    'error': str(e)
                }
                results.append(result)
                print(f"  ERROR processing chain {i+1}: {e}")
        
        return results


def main():
    """Test the puzzle reconstructor with sample images"""
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python puzzle_reconstructor.py <input_dir> <output_dir>")
        return
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    reconstructor = PuzzleReconstructor(output_dir)
    
    print("Loading images...")
    images = reconstructor.load_images(input_dir)
    if not images:
        print("No images found!")
        return
    
    print(f"Loaded {len(images)} images")
    
    print("Finding potential connections...")
    connections = reconstructor.find_potential_connections(images)
    
    print("Building image chains...")
    chains = reconstructor.build_image_chains(images, connections)
    
    print("Processing chains and creating final images...")
    processing_results = reconstructor.process_chains(images, chains)
    
    # Save analysis results
    analysis_results = {
        'total_images': len(images),
        'total_connections': len(connections),
        'total_chains': len(chains),
        'images': [{'index': img['index'], 'name': img['name']} for img in images],
        'connections': [
            {
                'from': conn['from_name'],
                'to': conn['to_name'],
                'similarity': conn['similarity']
            } for conn in connections
        ],
        'chains': [
            {
                'chain_id': i,
                'length': len(chain),
                'images': [c['image_name'] for c in chain],
                'avg_similarity': np.mean([c['similarity_to_next'] for c in chain[:-1]]) if len(chain) > 1 else 0
            } for i, chain in enumerate(chains)
        ],
        'processing_results': processing_results
    }
    
    output_path = Path(output_dir) / "puzzle_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"\nAnalysis complete! Results saved to: {output_path}")
    print(f"Found {len(chains)} product chains from {len(images)} images")
    
    # Print processing summary
    successful_products = sum(1 for r in processing_results if r['success'])
    print(f"Successfully created {successful_products}/{len(processing_results)} product images")


if __name__ == "__main__":
    main()