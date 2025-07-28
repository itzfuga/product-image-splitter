# ✂️ Product Image Separator Splitter

A web application for processing fashion product images from e-commerce sites (particularly Taobao). It automatically detects white separator areas between product photos and splits/combines them correctly.

## 🌟 Features

- 🔍 **Automatic Separator Detection**: Finds white background separators between product photos
- ✂️ **Smart Image Splitting**: Splits images at detected separators
- 🔄 **Proper Segment Combination**: Combines bottom of image N with top of image N+1
- 📱 **Drag & Drop Interface**: Easy file upload via drag and drop
- 💾 **Fast Batch Download**: Download all results as individual PNG files (no ZIP)
- 🌐 **URL Support**: Load images directly from Taobao URLs
- 📊 **Natural Sorting**: Processes files in correct order (1, 2, 3... not 1, 10, 11)

## 🚀 Quick Start

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/product-image-splitter.git
cd product-image-splitter
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR (optional, for text separator detection):
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
```

4. Run the application:
```bash
python app.py
```

5. Open http://localhost:8080 in your browser

### 🚀 Deployment Options

#### Deploy to Heroku
```bash
heroku create your-app-name
heroku buildpacks:add https://github.com/heroku/heroku-buildpack-apt
heroku buildpacks:add heroku/python
git push heroku main
```

#### Deploy to Railway
Click the button below to deploy:

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template)

## 📖 Usage

1. **Upload Images**: 
   - Drag and drop your images (1.jpg, 2.jpg, etc.) in order
   - Or drag image URLs from Taobao product pages
   
2. **Automatic Processing**: 
   - The tool detects white separators automatically
   - Creates products by combining segments properly
   
3. **Download Results**: 
   - Click "Download All Results" to get individual PNG files
   - Files download directly - no ZIP extraction needed
   
4. **Upload to Shopify**: 
   - Drag the downloaded files directly into Shopify
   - Ready for immediate use!

## 🔧 How It Works

1. **Separator Detection**:
   - Analyzes each image for white/background areas
   - Detects horizontal separator regions
   - Supports images with 0, 1, or 2 separators

2. **Smart Combination**:
   - Bottom of image 1 + Top of image 2 = Product 1
   - Bottom of image 2 + Top of image 3 = Product 2
   - And so on...

3. **File Processing**:
   - Natural sorting ensures correct order (1→2→3, not 1→10→11)
   - Maintains high quality PNG output
   - Preserves original image dimensions

## 📁 Project Structure

```
product-image-splitter/
├── app.py                    # Flask web server
├── separator_splitter.py     # Core separator detection engine
├── templates/
│   └── index.html           # Web interface
├── uploads/                 # Temporary upload storage
├── results/                 # Processing results
├── requirements.txt         # Python dependencies
├── Procfile                # Heroku deployment
├── runtime.txt             # Python version
└── README.md               # This file
```

## 🛠️ Configuration

The application works out of the box, but you can adjust:

- **Port**: Set `PORT` environment variable (default: 8080)
- **Upload Size**: Max 50MB per file
- **Separator Detection**: Adjust thresholds in `separator_splitter.py`

## 📝 License

MIT License - feel free to use for commercial projects!

## 🤝 Contributing

Pull requests welcome! Areas for improvement:
- Better separator detection algorithms
- Support for more image formats
- Batch processing optimizations
- Cloud storage integration

---

**Made for e-commerce sellers** | Perfect for Taobao, Shopify, and other platforms