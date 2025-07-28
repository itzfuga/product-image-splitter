# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a standalone web application for processing product images from e-commerce sites (particularly Taobao). It automatically detects separator images containing "STAFF/START EXCEED END" text to split multiple product photos into individual product listings, with automatic cropping capabilities.

## Key Functionality

- **Image URL Loading**: Can load images directly from URLs (including Taobao) via paste/drag-drop
- **OCR Integration**: Uses Tesseract.js for detecting separator text in images
- **Sorting Options**: Upload order (for Taobao), filename, or manual drag-and-drop
- **Auto-Cropping**: Removes gray borders and text areas from product images
- **Batch Processing**: Combines multiple image segments into single product images

## Architecture

The application is a single HTML file with embedded CSS and JavaScript:
- **OCR Engine**: Tesseract.js loaded from local file (`tesseract.min.js.Download`)
- **Image Processing**: Canvas-based manipulation for cropping and combining
- **No Build Process**: Pure HTML/CSS/JS, no bundling or compilation needed

## Development Commands

Since this is a standalone HTML file, there are no build/test commands. To develop:
- Open `Product Image Splitter & Auto-Crop v7.html` directly in a browser
- Ensure `tesseract.min.js.Download` is in the correct subfolder path

## Important Technical Details

- **CORS Handling**: Uses proxy fallback (`api.allorigins.win`) for cross-origin images
- **Manual Separator Marking**: Users can manually mark separator positions if OCR fails
- **Content Detection**: Auto-crop uses color analysis to distinguish product from background
- **Memory Management**: OCR worker is terminated on page unload