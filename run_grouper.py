#!/usr/bin/env python3
"""
Simple runner script for outfit grouper
"""

from outfit_grouper import OutfitGrouper
import sys
from pathlib import Path

def main():
    # Use current directory as input (where the numbered images are)
    current_dir = Path(".")
    output_dir = Path("./grouped_output")
    
    print("🎯 Fashion Image Outfit Grouper")
    print("=" * 40)
    
    # Check if numbered images exist
    image_files = list(current_dir.glob("*.jpg")) + list(current_dir.glob("*.png"))
    image_files = [f for f in image_files if f.name[0].isdigit()]  # Only numbered files
    
    if not image_files:
        print("❌ No numbered image files found in current directory!")
        print("Expected files: 1.jpg, 2.jpg, etc.")
        return
    
    print(f"📸 Found {len(image_files)} images")
    for img in sorted(image_files):
        print(f"   - {img.name}")
    
    # Create grouper and process
    grouper = OutfitGrouper(output_dir)
    
    try:
        product_paths = grouper.process_images(current_dir)
        
        print("\n✅ SUCCESS!")
        print(f"📁 Results saved to: {output_dir}")
        print(f"🎯 Created {len(product_paths)} product images")
        
        # List output files
        if output_dir.exists():
            output_files = list(output_dir.glob("*"))
            print("\n📋 Output files:")
            for file in sorted(output_files):
                print(f"   - {file.name}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()