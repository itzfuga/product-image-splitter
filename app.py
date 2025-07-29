#!/usr/bin/env python3
"""
Flask Web Application for Outfit Image Grouper
"""

from flask import Flask, request, jsonify, render_template, send_file, session
from werkzeug.utils import secure_filename
import os
import uuid
import shutil
from pathlib import Path
import json
import threading
import time
from separator_splitter import SeparatorSplitter

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Configuration
UPLOAD_FOLDER = Path('uploads')
RESULTS_FOLDER = Path('results')
DEBUG_ANALYSIS_FOLDER = Path('debug_analysis')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'avif'}

# Create directories
UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULTS_FOLDER.mkdir(exist_ok=True)
DEBUG_ANALYSIS_FOLDER.mkdir(exist_ok=True)
(DEBUG_ANALYSIS_FOLDER / 'originals').mkdir(exist_ok=True)
(DEBUG_ANALYSIS_FOLDER / 'results').mkdir(exist_ok=True)
(DEBUG_ANALYSIS_FOLDER / 'debug_images').mkdir(exist_ok=True)
(DEBUG_ANALYSIS_FOLDER / 'sessions').mkdir(exist_ok=True)

# Store processing status
processing_status = {}


def allowed_file(filename):
    if not filename or '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def cleanup_old_sessions():
    """Clean up old upload and result folders"""
    try:
        # Clean up folders older than 1 hour
        cutoff_time = time.time() - 3600
        
        for folder in [UPLOAD_FOLDER, RESULTS_FOLDER]:
            if folder.exists():
                for subfolder in folder.iterdir():
                    if subfolder.is_dir() and subfolder.stat().st_mtime < cutoff_time:
                        shutil.rmtree(subfolder, ignore_errors=True)
    except Exception as e:
        print(f"Cleanup error: {e}")


def process_images_async(session_id, upload_dir, result_dir):
    """Process images in background thread"""
    try:
        processing_status[session_id] = {
            'status': 'processing',
            'progress': 0,
            'message': 'Initializing...',
            'results': None
        }
        
        # Create separator splitter
        splitter = SeparatorSplitter(result_dir, session_id)
        
        # Update status
        processing_status[session_id]['progress'] = 10
        processing_status[session_id]['message'] = 'Loading images...'
        
        # Load images
        images = splitter.load_images(upload_dir)
        if not images:
            processing_status[session_id]['status'] = 'error'
            processing_status[session_id]['message'] = 'No valid images found'
            return
        
        processing_status[session_id]['progress'] = 20
        processing_status[session_id]['message'] = f'Detecting separators in {len(images)} images...'
        
        # Process images and split at separators
        segments = splitter.process_images(images)
        
        if not segments:
            processing_status[session_id]['status'] = 'error'
            processing_status[session_id]['message'] = 'No segments created - no separators detected'
            return
        
        processing_status[session_id]['progress'] = 60
        processing_status[session_id]['message'] = f'Creating products from {len(segments)} segments...'
        
        # Create debug visualizations
        splitter.create_debug_visualization(images, segments)
        
        # Create products by combining segments
        products = splitter.create_products_from_segments(segments)
        
        if not products:
            processing_status[session_id]['status'] = 'error'
            processing_status[session_id]['message'] = 'No products created from segments'
            return
        
        processing_status[session_id]['progress'] = 90
        processing_status[session_id]['message'] = f'Generated {len(products)} product images...'
        
        # Create product paths list for compatibility
        product_paths = []
        for product in products:
            # Handle both combined segments and single segments
            if 'bottom_segment' in product and 'top_segment' in product:
                # Combined segments product
                product_paths.append({
                    'product_id': product['product_id'],
                    'path': product['path'],
                    'filename': product['filename'],
                    'image_count': 2,  # Bottom segment + top segment
                    'images': [
                        f"{product['bottom_segment']['source']} (segment {product['bottom_segment']['segment_index']})",
                        f"{product['top_segment']['source']} (segment {product['top_segment']['segment_index']})"
                    ]
                })
            elif 'single_segment' in product:
                # Single segment product
                product_paths.append({
                    'product_id': product['product_id'],
                    'path': product['path'],
                    'filename': product['filename'],
                    'image_count': 1,  # Single segment
                    'images': [
                        f"{product['single_segment']['source']} (segment {product['single_segment']['segment_index']})"
                    ]
                })
            else:
                # Fallback for unknown structure
                product_paths.append({
                    'product_id': product['product_id'],
                    'path': product['path'],
                    'filename': product['filename'],
                    'image_count': 1,
                    'images': ['Unknown structure']
                })
        
        # Save processing info
        processing_info = {
            'session_id': session_id,
            'total_images': len(images),
            'total_segments': len(segments),
            'total_products': len(products),
            'products': products
        }
        
        info_path = result_dir / "processing_info.json"
        with open(info_path, 'w') as f:
            json.dump(processing_info, f, indent=2)
        
        # Also save to debug analysis folder for later inspection
        debug_session_path = DEBUG_ANALYSIS_FOLDER / 'sessions' / f"{session_id}_session.json"
        with open(debug_session_path, 'w') as f:
            json.dump(processing_info, f, indent=2)
        
        # Update final status
        processing_status[session_id] = {
            'status': 'completed',
            'progress': 100,
            'message': f'Processing complete! Created {len(product_paths)} products from separator detection.',
            'results': {
                'groups': len(products),  # Number of products created 
                'products': product_paths,
                'summary_path': None,  # No summary for separator splitting
                'info_path': str(info_path),
                'total_images': len(images),
                'total_segments': len(segments),
                'debug_analysis_url': f'/debug/analysis/{session_id}'  # Add debug URL
            }
        }
        
    except Exception as e:
        processing_status[session_id] = {
            'status': 'error',
            'progress': 0,
            'message': f'Error: {str(e)}',
            'results': None
        }


@app.route('/')
def index():
    cleanup_old_sessions()
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        if 'files[]' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files[]')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No files selected'}), 400
        
        # Create session ID and directories
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        
        upload_dir = UPLOAD_FOLDER / session_id
        result_dir = RESULTS_FOLDER / session_id
        upload_dir.mkdir(exist_ok=True)
        result_dir.mkdir(exist_ok=True)
        
        # Save uploaded files
        saved_files = []
        for i, file in enumerate(files):
            if file and file.filename and allowed_file(file.filename):
                print(f"Processing file: {file.filename}, content-type: {file.content_type}")
                
                filename = secure_filename(file.filename)
                if not filename:  # If secure_filename returns empty, create a name
                    ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'jpg'
                    filename = f"image_{i+1}.{ext}"
                
                # Ensure unique filename
                counter = 1
                original_name = filename
                while (upload_dir / filename).exists():
                    if '.' in original_name:
                        name, ext = original_name.rsplit('.', 1)
                        filename = f"{name}_{counter}.{ext}"
                    else:
                        filename = f"{original_name}_{counter}"
                    counter += 1
                
                filepath = upload_dir / filename
                try:
                    file.save(str(filepath))
                    saved_files.append(filename)
                    print(f"Successfully saved: {filename}")
                    
                    # Also save to debug analysis folder for later inspection
                    debug_original_path = DEBUG_ANALYSIS_FOLDER / 'originals' / f"{session_id}_{filename}"
                    shutil.copy2(str(filepath), str(debug_original_path))
                    
                except Exception as e:
                    print(f"Error saving file {filename}: {e}")
            else:
                print(f"Skipped file: {file.filename if file else 'None'} - not allowed or empty")
        
        if not saved_files:
            return jsonify({'error': 'No valid image files uploaded'}), 400
        
        # Start processing in background
        thread = threading.Thread(
            target=process_images_async,
            args=(session_id, upload_dir, result_dir)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'files_uploaded': len(saved_files),
            'message': f'Uploaded {len(saved_files)} files. Processing started...'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/status/<session_id>')
def get_status(session_id):
    """Get processing status for a session"""
    if session_id in processing_status:
        return jsonify(processing_status[session_id])
    else:
        return jsonify({
            'status': 'not_found',
            'message': 'Session not found'
        }), 404


@app.route('/results/<session_id>/<filename>')
def get_result_file(session_id, filename):
    """Serve result files"""
    try:
        result_dir = RESULTS_FOLDER / session_id
        file_path = result_dir / filename
        
        if file_path.exists() and file_path.is_file():
            return send_file(str(file_path))
        else:
            return jsonify({'error': 'File not found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/debug/originals/<session_id>/<filename>')
def get_debug_original(session_id, filename):
    """Serve original uploaded images for analysis"""
    try:
        upload_dir = UPLOAD_FOLDER / session_id
        file_path = upload_dir / filename
        
        if file_path.exists() and file_path.is_file():
            return send_file(str(file_path))
        else:
            return jsonify({'error': 'Original file not found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/debug/analysis/<session_id>')
def get_debug_analysis(session_id):
    """Get debug analysis for a session with image URLs"""
    try:
        result_dir = RESULTS_FOLDER / session_id
        upload_dir = UPLOAD_FOLDER / session_id
        
        if not result_dir.exists():
            return jsonify({'error': 'Session not found'}), 404
        
        # Get processing info
        info_path = result_dir / "processing_info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                processing_info = json.load(f)
        else:
            processing_info = {}
        
        # Get list of original images
        original_images = []
        if upload_dir.exists():
            for img_file in upload_dir.glob('*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp', '.avif']:
                    original_images.append({
                        'filename': img_file.name,
                        'url': f'/debug/originals/{session_id}/{img_file.name}'
                    })
        
        # Get list of result images
        result_images = []
        debug_images = []
        if result_dir.exists():
            for img_file in result_dir.glob('product_*.png'):
                result_images.append({
                    'filename': img_file.name,
                    'url': f'/results/{session_id}/{img_file.name}'
                })
            
            for debug_file in result_dir.glob('debug_*.png'):
                debug_images.append({
                    'filename': debug_file.name,
                    'url': f'/results/{session_id}/{debug_file.name}'
                })
        
        return jsonify({
            'session_id': session_id,
            'processing_info': processing_info,
            'original_images': original_images,
            'result_images': result_images,
            'debug_images': debug_images
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download/<session_id>')
def download_all(session_id):
    """Download all results as a ZIP file"""
    try:
        result_dir = RESULTS_FOLDER / session_id
        if not result_dir.exists():
            return jsonify({'error': 'Results not found'}), 404
        
        # Create ZIP file
        zip_path = result_dir / 'results.zip'
        shutil.make_archive(str(zip_path.with_suffix('')), 'zip', str(result_dir))
        
        return send_file(str(zip_path), as_attachment=True, download_name='outfit_groups.zip')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("🎯 Fashion Image Outfit Grouper Web App")
    print("=" * 50)
    print("Starting server...")
    print("Open your browser to: http://localhost:8080")
    print("=" * 50)
    
    # Get port from environment variable (for deployment) or use 8080
    import os
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)