#!/usr/bin/env python3
"""
Startup script for Fashion Image Outfit Grouper Web App
"""

import webbrowser
import time
import threading
from app import app

def open_browser():
    """Open browser after short delay"""
    time.sleep(1.5)
    webbrowser.open('http://localhost:8080')

if __name__ == '__main__':
    print("🎯 Fashion Image Outfit Grouper Web App")
    print("=" * 50)
    print("Starting web server...")
    print("Your browser will open automatically!")
    print("\nFeatures:")
    print("• 🖼️  Drag & drop fashion images")
    print("• 🧠 AI-powered outfit detection")
    print("• 📸 Automatic grouping by similarity")
    print("• ✂️  Auto-cropping and combining")
    print("• 📦 Download individual or all results")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 50)
    
    # Start browser opener in background
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start Flask app
    app.run(debug=False, host='0.0.0.0', port=8080)