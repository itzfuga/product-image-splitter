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
    
    def find_potential_connections(self, images):
        """Find all potential connections between images"""
        connections = []
        
        print(f"Analyzing {len(images)} images for potential connections...")
        
        for i, img1 in enumerate(images):
            print(f"Analyzing image {i+1}: {img1['name']}")
            
            # Extract bottom edge features of current image
            bottom_features = self.extract_edge_features(img1['cv2_image'], 'bottom')
            bottom_edge_info = self.detect_edge_type(img1['cv2_image'], 'bottom')
            
            # Only consider connecting if the bottom edge looks "cut off"
            if not bottom_edge_info['is_likely_cut']:
                print(f"  Bottom edge looks natural (cut_score: {bottom_edge_info['cut_score']:.2f}), skipping connections")
                continue
            
            for j, img2 in enumerate(images):
                if i >= j:  # Avoid self-comparison and duplicates
                    continue
                
                # Extract top edge features of candidate next image
                top_features = self.extract_edge_features(img2['cv2_image'], 'top')
                top_edge_info = self.detect_edge_type(img2['cv2_image'], 'top')
                
                # Only consider if the top edge also looks "cut off"
                if not top_edge_info['is_likely_cut']:
                    continue
                
                # Calculate similarity between bottom of img1 and top of img2
                similarity = self.calculate_edge_similarity(bottom_features, top_features)
                
                if similarity['overall_similarity'] > self.similarity_threshold:
                    connection = {
                        'from_image': i,
                        'to_image': j,
                        'from_name': img1['name'],
                        'to_name': img2['name'],
                        'similarity': similarity['overall_similarity'],
                        'detailed_similarity': similarity,
                        'from_edge_info': bottom_edge_info,
                        'to_edge_info': top_edge_info
                    }
                    connections.append(connection)
                    print(f"  Found connection: {img1['name']} -> {img2['name']} (similarity: {similarity['overall_similarity']:.3f})")
        
        # Sort connections by similarity score (best first)
        connections.sort(key=lambda x: x['similarity'], reverse=True)
        
        print(f"Found {len(connections)} potential connections")
        return connections
    
    def build_image_chains(self, images, connections):
        """Build chains of connected images using dynamic programming approach"""
        print("Building image chains from connections...")
        
        # Create adjacency list
        graph = {i: [] for i in range(len(images))}
        for conn in connections:
            graph[conn['from_image']].append({
                'to': conn['to_image'],
                'similarity': conn['similarity'],
                'connection': conn
            })
        
        chains = []
        used_images = set()
        
        # Try starting a chain from each unused image
        for start_idx in range(len(images)):
            if start_idx in used_images:
                continue
            
            # Build the best chain starting from this image
            chain = self.build_best_chain(start_idx, graph, used_images.copy())
            
            if len(chain) > 1:  # Only consider chains with multiple images
                chains.append(chain)
                # Mark images in this chain as used
                for img_idx in [c['image_index'] for c in chain]:
                    used_images.add(img_idx)
                    
                chain_names = " -> ".join([c['image_name'] for c in chain])
                print(f"Built chain: {chain_names}")
            elif len(chain) == 1:
                # Single image - might be standalone
                chains.append(chain)
                used_images.add(start_idx)
                print(f"Standalone image: {chain[0]['image_name']}")
        
        print(f"Built {len(chains)} chains total")
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
        """Stitch images in a chain together vertically"""
        if len(chain) <= 1:
            # Single image - just crop and return
            img_idx = chain[0]['image_index']
            image = images[img_idx]['cv2_image']
            
            if self.auto_crop:
                left, top, right, bottom = self.detect_content_bounds(image)
                cropped = image[top:bottom+1, left:right+1]
                return self.enforce_4_5_ratio(cropped) if self.force_4_5_ratio else cropped
            else:
                return self.enforce_4_5_ratio(image) if self.force_4_5_ratio else image
        
        # Multiple images - stitch vertically
        stitched_images = []
        
        for i, chain_item in enumerate(chain):
            img_idx = chain_item['image_index']
            image = images[img_idx]['cv2_image'].copy()
            
            # For middle images in chain, remove edges where stitching occurs
            if i > 0:  # Not first image - remove top edge
                edge_pixels = min(self.edge_height, image.shape[0] // 4)
                image = image[edge_pixels:, :]
            
            if i < len(chain) - 1:  # Not last image - remove bottom edge
                edge_pixels = min(self.edge_height, image.shape[0] // 4)
                image = image[:-edge_pixels, :]
            
            stitched_images.append(image)
        
        # Find the maximum width for consistency
        max_width = max(img.shape[1] for img in stitched_images)
        
        # Resize all images to same width and stitch vertically
        resized_images = []
        for img in stitched_images:
            if img.shape[1] != max_width:
                # Resize to match width while maintaining aspect ratio
                height = int(img.shape[0] * max_width / img.shape[1])
                resized = cv2.resize(img, (max_width, height), interpolation=cv2.INTER_AREA)
                resized_images.append(resized)
            else:
                resized_images.append(img)
        
        # Concatenate vertically
        stitched = np.vstack(resized_images)
        
        # Apply auto-crop and ratio enforcement
        if self.auto_crop:
            left, top, right, bottom = self.detect_content_bounds(stitched)
            stitched = stitched[top:bottom+1, left:right+1]
        
        if self.force_4_5_ratio:
            stitched = self.enforce_4_5_ratio(stitched)
        
        return stitched
    
    def enforce_4_5_ratio(self, image):
        """Force image to 4:5 aspect ratio with padding"""
        height, width = image.shape[:2]
        target_ratio = 4.0 / 5.0
        current_ratio = width / height
        
        if abs(current_ratio - target_ratio) < 0.01:
            return image  # Already close to 4:5
        
        if current_ratio > target_ratio:
            # Image is too wide - add padding to height
            new_height = int(width / target_ratio)
            padding_needed = new_height - height
            pad_top = padding_needed // 2
            pad_bottom = padding_needed - pad_top
            
            padded = cv2.copyMakeBorder(
                image, pad_top, pad_bottom, 0, 0,
                cv2.BORDER_CONSTANT, value=self.background_color
            )
        else:
            # Image is too tall - add padding to width
            new_width = int(height * target_ratio)
            padding_needed = new_width - width
            pad_left = padding_needed // 2
            pad_right = padding_needed - pad_left
            
            padded = cv2.copyMakeBorder(
                image, 0, 0, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=self.background_color
            )
        
        return padded
    
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