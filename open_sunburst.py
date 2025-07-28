#!/usr/bin/env python3
"""
Simple script to open the sunburst HTML file in browser
"""

import webbrowser
import os
import sys

def open_sunburst():
    # List of HTML files to try
    html_files = [
        "clean_working_sunburst.html",
        "clean_working_extended_sunburst.html", 
        "test_simple_sunburst.html",
        "test_manual_sunburst.html"
    ]
    
    print("🌐 Opening sunburst HTML files in browser...")
    
    for filename in html_files:
        if os.path.exists(filename):
            file_path = os.path.abspath(filename)
            file_url = f"file://{file_path}"
            
            print(f"Opening: {filename}")
            print(f"URL: {file_url}")
            print(f"Size: {os.path.getsize(filename):,} bytes")
            
            try:
                webbrowser.open(file_url)
                print(f"✅ Opened {filename} in browser")
                break
            except Exception as e:
                print(f"❌ Failed to open {filename}: {e}")
        else:
            print(f"❌ File not found: {filename}")
    
    print("\n" + "="*50)
    print("TROUBLESHOOTING TIPS:")
    print("="*50)
    print("If the sunburst appears empty:")
    print("1. Check if JavaScript is enabled in your browser")
    print("2. Try a different browser (Chrome, Firefox, etc.)")
    print("3. Check browser console for errors (F12)")
    print("4. Try opening the file manually:")
    
    for filename in html_files:
        if os.path.exists(filename):
            file_path = os.path.abspath(filename)
            print(f"   file://{file_path}")
            break

if __name__ == "__main__":
    open_sunburst() 