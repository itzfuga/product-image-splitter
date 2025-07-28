#!/usr/bin/env python3
"""
Simple startup script with better WSL compatibility
"""

from app import app
import socket

def get_ip():
    """Get local IP address"""
    try:
        # Connect to a remote address to get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

if __name__ == '__main__':
    local_ip = get_ip()
    
    print("🎯 Fashion Image Outfit Grouper Web App")
    print("=" * 50)
    print("Starting web server...")
    print(f"Open your browser to one of these URLs:")
    print(f"• http://localhost:8080")
    print(f"• http://127.0.0.1:8080") 
    print(f"• http://{local_ip}:8080")
    print("\nIf localhost doesn't work, try the IP address!")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Start Flask app on all interfaces
    app.run(debug=False, host='0.0.0.0', port=8080, threaded=True)