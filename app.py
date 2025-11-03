#!/usr/bin/env python3
"""
Flask Web Application for Outfit Image Grouper with White Separator Stitching
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
from taobao_splitter import TaobaoSplitter
from enhanced_taobao_splitter import EnhancedTaobaoSplitter
from ultra_precise_taobao_splitter import UltraPreciseTaobaoSplitter
from fixed_taobao_splitter import FixedTaobaoSplitter
from puzzle_reconstructor import PuzzleReconstructor

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


def process_puzzle_async(session_id, upload_dir, result_dir):
    """Process puzzle reconstruction in background thread"""
    try:
        processing_status[session_id] = {
            'status': 'processing',
            'progress': 0,
            'message': 'Initializing puzzle reconstruction...',
            'results': None
        }
        
        # Create puzzle reconstructor
        reconstructor = PuzzleReconstructor(result_dir, session_id)
        
        # Update status
        processing_status[session_id]['progress'] = 10
        processing_status[session_id]['message'] = 'Loading images...'
        
        # Load images
        images = reconstructor.load_images(upload_dir)
        if not images:
            processing_status[session_id]['status'] = 'error'
            processing_status[session_id]['message'] = 'No valid images found'
            return
        
        processing_status[session_id]['progress'] = 30
        processing_status[session_id]['message'] = f'Finding connections between {len(images)} images...'
        
        # Find potential connections (now uses sequential grouping)
        connections = reconstructor.find_potential_connections(images)
        
        processing_status[session_id]['progress'] = 60
        processing_status[session_id]['message'] = f'Building sequential groups...'
        
        # Build image chains (now creates sequential groups)
        chains = reconstructor.build_image_chains(images, connections)
        
        processing_status[session_id]['progress'] = 80
        processing_status[session_id]['message'] = f'Reconstructing {len(chains)} product images...'
        
        # Process chains and create final images
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
                    'avg_similarity': sum([c['similarity_to_next'] for c in chain[:-1]]) / max(1, len(chain)-1) if len(chain) > 1 else 0
                } for i, chain in enumerate(chains)
            ],
            'processing_results': processing_results
        }
        
        # Save analysis to JSON
        analysis_path = result_dir / "puzzle_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        # Count successful products
        successful_products = sum(1 for r in processing_results if r['success'])
        
        # Create product paths list for web interface
        product_paths = []
        for result in processing_results:
            if result['success']:
                # Parse dimensions string (e.g., "1857x2322") into width/height
                dims_str = result.get('dimensions', '0x0')
                try:
                    width, height = dims_str.split('x')
                    dimensions = {'width': int(width), 'height': int(height)}
                except:
                    dimensions = {'width': 0, 'height': 0}
                
                product_paths.append({
                    'product_id': result['chain_id'],
                    'path': result['output_path'],
                    'filename': result['output_file'],
                    'image_count': len(result['source_images']),
                    'images': result['source_images'],
                    'dimensions': dimensions
                })
        
        # Update final status
        processing_status[session_id] = {
            'status': 'completed',
            'progress': 100,
            'message': f'Puzzle reconstruction complete! Created {successful_products} products from {len(images)} images.',
            'results': {
                'groups': successful_products,
                'products': product_paths,
                'analysis_path': str(analysis_path),
                'info_path': str(analysis_path),
                'total_images': len(images),
                'total_chains': len(chains),
                'total_connections': len(connections),
                'debug_analysis_url': f'/debug/puzzle/{session_id}'
            }
        }
        
    except Exception as e:
        processing_status[session_id] = {
            'status': 'error',
            'progress': 0,
            'message': f'Puzzle reconstruction error: {str(e)}',
            'results': None
        }


def process_taobao_images_async(session_id, upload_dir, result_dir):
    """Process Taobao images with improved separator detection"""
    try:
        processing_status[session_id] = {
            'status': 'processing',
            'progress': 0,
            'message': 'Initializing Taobao image processing...',
            'results': None
        }
        
        # Create Robust White Detector (always finds largest central rectangle)
        from robust_white_detector import RobustWhiteDetector
        splitter = RobustWhiteDetector(result_dir, session_id)
        
        # Update status
        processing_status[session_id]['progress'] = 10
        processing_status[session_id]['message'] = 'Loading images...'
        
        # Load images
        images = splitter.load_images(upload_dir)
        if not images:
            processing_status[session_id]['status'] = 'error'
            processing_status[session_id]['message'] = 'No valid images found'
            return
        
        processing_status[session_id]['progress'] = 30
        processing_status[session_id]['message'] = f'Detecting separators in {len(images)} images...'
        
        # Process images and split at separators (chronological)
        image_parts = splitter.process_images_chronologically(images)
        
        processing_status[session_id]['progress'] = 70
        processing_status[session_id]['message'] = f'Combining {len(image_parts)} image parts into products...'
        
        # Combine parts into products (chronological)
        products = splitter.combine_chronologically(image_parts)
        
        if not products:
            processing_status[session_id]['status'] = 'error'
            processing_status[session_id]['message'] = 'No products created'
            return
        
        processing_status[session_id]['progress'] = 90
        processing_status[session_id]['message'] = f'Generated {len(products)} product images...'
        
        # Create product paths list for web interface
        product_paths = []
        for product in products:
            product_paths.append({
                'product_id': product['product_id'],
                'path': product['path'],
                'filename': product['filename'],
                'image_count': 2 if 'top_source' in product else 1,
                'images': [
                    product.get('bottom_source', product.get('source', 'Unknown')),
                    product.get('top_source', '')
                ] if 'top_source' in product else [product.get('source', 'Unknown')],
                'dimensions': product.get('dimensions', '0x0')
            })
        
        # Save processing info
        processing_info = {
            'session_id': session_id,
            'total_images': len(images),
            'total_products': len(products),
            'products': products
        }
        
        info_path = result_dir / "taobao_processing_info.json"
        with open(info_path, 'w') as f:
            json.dump(processing_info, f, indent=2)
        
        # Update final status
        processing_status[session_id] = {
            'status': 'completed',
            'progress': 100,
            'message': f'Taobao processing complete! Created {len(products)} products from {len(images)} images.',
            'results': {
                'groups': len(products),
                'products': product_paths,
                'info_path': str(info_path),
                'total_images': len(images),
                'processing_type': 'taobao'
            }
        }
        
    except Exception as e:
        processing_status[session_id] = {
            'status': 'error',
            'progress': 0,
            'message': f'Taobao processing error: {str(e)}',
            'results': None
        }


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


@app.route('/version')
def version():
    """Check deployed version"""
    return jsonify({
        'version': 'enhanced_white_v9_perfect',
        'timestamp': '2025-11-03_18:45',
        'features': [
            'enhanced_white_rectangle_detection',
            'chronological_image_pairing',
            'perfect_separator_removal',
            'contour_based_boundary_detection',
            'clean_product_extraction',
            'full_model_preservation',
            'pure_white_background',
            'complete_text_elimination'
        ]
    })

@app.route('/')
def index():
    cleanup_old_sessions()
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_files():
    print("✂️ SEPARATOR SPLITTING ENDPOINT CALLED")
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


@app.route('/upload_taobao', methods=['POST'])
def upload_taobao_files():
    """Upload files for Taobao separator processing"""
    print("🏷️ TAOBAO SEPARATOR PROCESSING ENDPOINT CALLED")
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
                print(f"Processing Taobao file: {file.filename}, content-type: {file.content_type}")
                
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
                    print(f"Successfully saved Taobao file: {filename}")
                    
                    # Also save to debug analysis folder for later inspection
                    debug_original_path = DEBUG_ANALYSIS_FOLDER / 'originals' / f"{session_id}_{filename}"
                    shutil.copy2(str(filepath), str(debug_original_path))
                    
                except Exception as e:
                    print(f"Error saving Taobao file {filename}: {e}")
            else:
                print(f"Skipped Taobao file: {file.filename if file else 'None'} - not allowed or empty")
        
        if not saved_files:
            return jsonify({'error': 'No valid image files uploaded'}), 400
        
        # Start Taobao processing in background
        thread = threading.Thread(
            target=process_taobao_images_async,
            args=(session_id, upload_dir, result_dir)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'files_uploaded': len(saved_files),
            'message': f'Uploaded {len(saved_files)} files. Taobao processing started...'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/upload_puzzle', methods=['POST'])
def upload_puzzle_files():
    """Upload files for puzzle reconstruction"""
    print("🧩 PUZZLE RECONSTRUCTION ENDPOINT CALLED")
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
                print(f"Processing puzzle file: {file.filename}, content-type: {file.content_type}")
                
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
                    print(f"Successfully saved puzzle file: {filename}")
                    
                    # Also save to debug analysis folder for later inspection
                    debug_original_path = DEBUG_ANALYSIS_FOLDER / 'originals' / f"{session_id}_{filename}"
                    shutil.copy2(str(filepath), str(debug_original_path))
                    
                except Exception as e:
                    print(f"Error saving puzzle file {filename}: {e}")
            else:
                print(f"Skipped puzzle file: {file.filename if file else 'None'} - not allowed or empty")
        
        if not saved_files:
            return jsonify({'error': 'No valid image files uploaded'}), 400
        
        # Start puzzle processing in background
        thread = threading.Thread(
            target=process_puzzle_async,
            args=(session_id, upload_dir, result_dir)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'files_uploaded': len(saved_files),
            'message': f'Uploaded {len(saved_files)} files. Puzzle reconstruction started...'
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


@app.route('/debug/puzzle/<session_id>')
def get_puzzle_debug_analysis(session_id):
    """Get puzzle reconstruction debug analysis for a session"""
    try:
        result_dir = RESULTS_FOLDER / session_id
        upload_dir = UPLOAD_FOLDER / session_id
        
        if not result_dir.exists():
            return jsonify({'error': 'Session not found'}), 404
        
        # Get puzzle analysis info
        analysis_path = result_dir / "puzzle_analysis.json"
        if analysis_path.exists():
            with open(analysis_path, 'r') as f:
                analysis_info = json.load(f)
        else:
            analysis_info = {}
        
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
        if result_dir.exists():
            for img_file in result_dir.glob('product_*.jpg'):
                result_images.append({
                    'filename': img_file.name,
                    'url': f'/results/{session_id}/{img_file.name}'
                })
        
        return jsonify({
            'session_id': session_id,
            'analysis_type': 'puzzle_reconstruction',
            'analysis_info': analysis_info,
            'original_images': original_images,
            'result_images': result_images
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