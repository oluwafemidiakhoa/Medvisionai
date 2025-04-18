
#!/usr/bin/env python3
"""Script to prepare the codebase for deployment by cleaning unnecessary files"""
import os
import shutil
import glob
import sys

def cleanup_for_deployment():
    """Remove unnecessary files to reduce deployment size"""
    print("Preparing for deployment...")
    
    # Files and directories to remove
    patterns_to_remove = [
        "__pycache__/**/*.pyc",
        "**/__pycache__",
        "attached_assets/*",
        ".pytest_cache",
        "test_*.py",
        "evaluation_suite.py",
        "**/fda_*.py",
        "fda_docs/**",
        "test_gold_standard/**",
        "**/*.md",
        "profiling_ui.py",
        "smart_recommendations_ui.py",
        "move_credentials_to_secrets.py",
        "update_code_for_secrets.py",
        "assets/*",
        "**/.git*",
        "**/*.ipynb",
        "diagnose_service_account.py",
        "fix_service_account.py",
        "fix_google_sheets.py",
        "performance_profiler.py",
        "gen.json",
        ".env",
        ".pythonlibs/**/*.html",
        ".pythonlibs/**/*.md",
        ".pythonlibs/**/test*",
        ".pythonlibs/**/tests",
        ".pythonlibs/**/example*",
        ".pythonlibs/**/docs",
        ".pythonlibs/**/__pycache__",
        # Additional patterns to further reduce size
        ".streamlit/config.toml",
        "**/*.pyc",
        "**/*.pyo",
        "**/*.pyd",
        ".git*",
        ".vscode",
        ".idea",
        ".coverage",
        "htmlcov",
        "**/*.log",
        "**/*.bak",
        "**/*.swp",
        "**/*.swo",
        ".DS_Store",
        "Thumbs.db",
        "**/*.gz",
        "**/*.zip",
        "**/node_modules"
    ]
    
    # Clean up Python cache files
    for pattern in patterns_to_remove:
        for path in glob.glob(pattern, recursive=True):
            if os.path.isdir(path):
                try:
                    shutil.rmtree(path)
                    print(f"Removed directory: {path}")
                except Exception as e:
                    print(f"Error removing {path}: {e}")
            elif os.path.isfile(path):
                try:
                    os.remove(path)
                    print(f"Removed file: {path}")
                except Exception as e:
                    print(f"Error removing {path}: {e}")
    
    # Clean up .pythonlibs more aggressively
    pythonlibs_path = ".pythonlibs"
    if os.path.exists(pythonlibs_path):
        try:
            # Keep only essential library files
            for root, dirs, files in os.walk(pythonlibs_path, topdown=False):
                # Remove unnecessary files
                for file in files:
                    if file.endswith(('.html', '.md', '.txt', '.rst', '.c', '.h', '.cpp', '.hpp', '.pyx', '.pxd')):
                        file_path = os.path.join(root, file)
                        try:
                            os.remove(file_path)
                            print(f"Removed file: {file_path}")
                        except Exception as e:
                            print(f"Error removing {file_path}: {e}")
            
            print("Additional .pythonlibs cleanup completed")
        except Exception as e:
            print(f"Error during aggressive .pythonlibs cleanup: {e}")
    
    # Report approximate size
    try:
        total_size = get_directory_size('.')
        print(f"Current directory size after cleanup: {format_size(total_size)}")
    except Exception as e:
        print(f"Error calculating directory size: {e}")
    
    print("Deployment preparation complete!")

def get_directory_size(path='.'):
    """Calculate the total size of a directory and its contents"""
    total = 0
    for entry in os.scandir(path):
        if entry.is_file():
            total += entry.stat().st_size
        elif entry.is_dir():
            total += get_directory_size(entry.path)
    return total

def format_size(size):
    """Format size in bytes to a human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"

if __name__ == "__main__":
    cleanup_for_deployment()
