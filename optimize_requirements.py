
#!/usr/bin/env python3
"""Script to optimize requirements.txt for deployment by removing unnecessary packages"""
import os
import sys

def optimize_requirements():
    """Remove unnecessary packages from requirements.txt for deployment"""
    print("Optimizing requirements.txt for deployment...")
    
    # Read original requirements
    with open("requirements.txt", "r") as f:
        lines = f.readlines()
    
    # Keep track of essential packages
    essential_packages = [
        "streamlit",
        "pillow",
        "numpy",
        "pandas",
        "pydicom",
        "pylibjpeg",
        "pylibjpeg-libjpeg",
        "scikit-image",
        "opencv-python-headless",
        "deep-translator",
        "requests",
        "fpdf2",
        "umls-api-client"
    ]
    
    # Filter requirements to keep only essential packages
    optimized_lines = []
    for line in lines:
        line = line.strip()
        # Skip comments and empty lines
        if not line or line.startswith("#"):
            optimized_lines.append(line)
            continue
        
        # Check if this is an essential package
        is_essential = False
        for pkg in essential_packages:
            if line.startswith(pkg) or pkg in line.split(">=")[0]:
                is_essential = True
                break
        
        if is_essential:
            optimized_lines.append(line)
    
    # Create a backup of the original requirements
    if os.path.exists("requirements.txt"):
        with open("requirements.txt.bak", "w") as f:
            f.write("\n".join(lines))
    
    # Write the optimized requirements
    with open("requirements.txt", "w") as f:
        f.write("\n".join(optimized_lines))
    
    print("Requirements optimization complete!")
    print("Original requirements saved as requirements.txt.bak")

if __name__ == "__main__":
    optimize_requirements()
