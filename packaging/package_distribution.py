"""
package_distribution.py - Create MyAIAssistant portable distribution package
Creates a zip file ready for sharing with end users
"""

import os
import shutil
import zipfile
from pathlib import Path
from datetime import datetime

def create_portable_package():
    """Create a portable zip package for distribution."""
    
    # Paths
    dist_dir = Path(__file__).parent
    root_dir = dist_dir.parent
    package_dir = dist_dir / "MyAIAssistant_Portable"
    
    print("Creating MyAIAssistant Portable Distribution...")
    
    # Clean previous package
    if package_dir.exists():
        shutil.rmtree(package_dir)
    
    # Create package directory
    package_dir.mkdir()
    
    # Files to include
    files_to_copy = [
        "MyAIAssistant.exe",
        "MyAIAssistant_Launcher.bat",
        "README_PORTABLE.md",
        "SETUP_PORTABLE.bat"
    ]
    
    # Copy executable files
    print("Copying main files...")
    for file in files_to_copy:
        src = dist_dir / file
        if src.exists():
            shutil.copy2(src, package_dir / file)
            print(f"  OK {file}")
        else:
            print(f"  MISSING {file} not found")
    
    # Copy directories
    dirs_to_copy = ["backend", "frontend"]
    for dir_name in dirs_to_copy:
        src_dir = dist_dir / dir_name
        dst_dir = package_dir / dir_name
        if src_dir.exists():
            shutil.copytree(src_dir, dst_dir, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            print(f"  OK {dir_name}/")
    
    # Create version info
    version_info = f"""MyAIAssistant Portable v1.2.0
Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Platform: Windows x64
Type: Portable (No installation required)

This package contains everything needed to run MyAIAssistant
on any Windows 10/11 system without dependencies.
"""
    
    with open(package_dir / "VERSION.txt", "w") as f:
        f.write(version_info)
    
    # Create zip archive
    zip_name = f"MyAIAssistant_Portable_v1.2.0_{datetime.now().strftime('%Y%m%d_%H%M')}.zip"
    zip_path = dist_dir / zip_name
    
    print(f"\nCreating zip archive: {zip_name}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in package_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(package_dir)
                zipf.write(file_path, arcname)
    
    # Calculate sizes
    zip_size = zip_path.stat().st_size / (1024 * 1024)  # MB
    package_size = sum(f.stat().st_size for f in package_dir.rglob('*') if f.is_file()) / (1024 * 1024)  # MB
    
    print(f"\nPackage created successfully!")
    print(f"  Package size: {package_size:.1f} MB")
    print(f"  Zip size: {zip_size:.1f} MB")
    print(f"  Location: {zip_path}")
    
    # Clean up temporary directory
    shutil.rmtree(package_dir)
    
    print(f"\nDistribution Ready!")
    print(f"Share this file with users: {zip_name}")
    print(f"Users just need to unzip and run MyAIAssistant_Launcher.bat")

if __name__ == "__main__":
    create_portable_package()
