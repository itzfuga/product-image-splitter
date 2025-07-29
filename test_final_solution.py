#!/usr/bin/env python3
"""
Test the final integrated Taobao image reconstruction solution
"""

import os
from pathlib import Path
from puzzle_reconstructor import PuzzleReconstructor

def test_final_solution():
    """Test the complete solution with the 1-9.jpg images"""
    print("Testing final integrated Taobao image reconstruction solution...")
    print("=" * 60)
    
    # Create result directory
    result_dir = 'final_test_results'
    reconstructor = PuzzleReconstructor(result_dir)
    
    # Load only the 1-9.jpg images for clean testing
    print("Loading test images (1-9.jpg)...")
    all_images = reconstructor.load_images('results')
    
    # Filter to just 1-9.jpg
    test_images = [img for img in all_images if img['name'] in 
                   ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg', '7.jpg', '8.jpg', '9.jpg']]
    
    print(f"Loaded {len(test_images)} test images")
    print()
    
    # Process with integrated smart matching and stitching
    print("Using smart matching with chronological constraints...")
    chains = reconstructor.find_sequential_groups(test_images)
    
    print()
    print("Expected groupings:")
    print("  1.jpg (standalone)")
    print("  2.jpg + 3.jpg = Product 2") 
    print("  4.jpg + 5.jpg = Product 3")
    print("  6.jpg + 7.jpg = Product 4")
    print("  8.jpg + 9.jpg = Product 5")
    print()
    
    print("Actual groupings:")
    for i, chain in enumerate(chains):
        chain_names = [item['image_name'] for item in chain]
        print(f"  Group {i+1}: {' + '.join(chain_names)}")
    print()
    
    # Process chains and create final images
    print("Processing chains with smart stitching...")
    results = reconstructor.process_chains(test_images, chains)
    
    # Print results summary
    print()
    print("RESULTS SUMMARY:")
    print("=" * 40)
    successful = sum(1 for r in results if r['success'])
    print(f"Successfully created {successful}/{len(results)} product images")
    
    for result in results:
        if result['success']:
            source_str = ' + '.join(result['source_images'])
            print(f"✓ {result['output_file']}: {source_str} ({result['dimensions']})")
        else:
            source_str = ' + '.join(result['source_images'])
            print(f"✗ Failed: {source_str} - {result.get('error', 'Unknown error')}")
    
    print()
    print(f"Results saved to: {result_dir}/")
    print("Check the generated product images to verify quality!")

if __name__ == "__main__":
    test_final_solution()