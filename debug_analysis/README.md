# Debug Analysis Folder

This folder contains all uploaded images and processing results for analysis by Claude.

## Folder Structure:

### `/originals/`
- Contains all original uploaded images
- Format: `{session_id}_{original_filename}`
- Example: `a1b2c3d4_image_001_dress.jpg`

### `/results/`
- Contains all final processed product images
- Format: `{session_id}_product_{id}.png`  
- Example: `a1b2c3d4_product_1.png`

### `/debug_images/`
- Contains debug comparison images showing before/after processing
- Format: `{session_id}_debug_product_{id}.png`
- Shows original vs processed with content bounds overlay

### `/sessions/`
- Contains JSON metadata for each processing session
- Format: `{session_id}_session.json`
- Includes processing info, segment data, and product details

## Usage:
When you upload images to the web app, all files are automatically saved here for Claude to analyze and identify processing issues.

**Command for Claude:** "Analyze the latest session" or "Look at session {session_id}"