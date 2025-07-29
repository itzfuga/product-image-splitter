#!/usr/bin/env python3
"""Test the core algorithm that powers the web interface"""

import os
import shutil

def test_web_workflow():
    """Test the core algorithm with 1-9.jpg images"""
    
    # Create a temporary directory for uploads
    upload_dir = "temp_uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    # Copy 1-9.jpg files to upload directory
    for i in range(1, 10):
        src = f"results/{i}.jpg"
        dst = f"{upload_dir}/{i}.jpg"
        if os.path.exists(src):
            shutil.copy2(src, dst)
    
    print(f"Created upload directory with {len(os.listdir(upload_dir))} images")
    
    # Test the core processing logic directly
    print("\n=== Testing Core Processing Logic ===")
    
    from puzzle_reconstructor import PuzzleReconstructor
    
    # Use the upload directory as input
    output_dir = "test_web_output"
    reconstructor = PuzzleReconstructor(output_dir)
    
    print("Loading images...")
    images = reconstructor.load_images(upload_dir)
    print(f"Loaded {len(images)} images: {[img['name'] for img in images]}")
    
    print("\nBuilding chains...")
    chains = reconstructor.build_image_chains(images, [])
    
    print(f"\nFound {len(chains)} chains:")
    for i, chain in enumerate(chains):
        chain_images = [item['image_name'] for item in chain]
        print(f"  Chain {i+1}: {' + '.join(chain_images)}")
    
    print("\nProcessing chains...")
    results = reconstructor.process_chains(images, chains)
    
    print(f"\nProcessing Results:")
    for i, result in enumerate(results):
        if result['success']:
            print(f"  ✅ Product {i+1}: {' + '.join(result['source_images'])} → {result['output_file']}")
        else:
            print(f"  ❌ Product {i+1}: ERROR - {result.get('error', 'Unknown')}")
    
    # Check if results match expectations
    expected_results = [
        ["1.jpg"],           # Product 1: standalone
        ["2.jpg", "3.jpg"],  # Product 2: fragment + full
        ["4.jpg", "5.jpg"],  # Product 3: fragment + full  
        ["6.jpg", "7.jpg"],  # Product 4: fragment + full
        ["8.jpg"],           # Product 5: standalone fragment
        ["9.jpg"]            # Product 6: standalone black outfit
    ]
    
    print(f"\n=== VALIDATION ===")
    success_count = 0
    for i, (result, expected) in enumerate(zip(results, expected_results)):
        if result['success'] and result['source_images'] == expected:
            print(f"  ✅ Product {i+1}: CORRECT")
            success_count += 1
        else:
            print(f"  ❌ Product {i+1}: Expected {expected}, got {result.get('source_images', 'ERROR')}")
    
    print(f"\nOVERALL: {success_count}/{len(expected_results)} products correct ({success_count/len(expected_results)*100:.0f}%)")
    
    # Cleanup
    shutil.rmtree(upload_dir)
    
    return success_count == len(expected_results)

if __name__ == "__main__":
    success = test_web_workflow()
    print(f"\n{'🎉 ALL TESTS PASSED!' if success else '❌ TESTS FAILED!'}")