import os
import sys
import subprocess

def run_script(script_path, cwd):
    print(f"\n{'='*60}")
    print(f" SCRAPING: {os.path.basename(script_path)}")
    print(f"{'='*60}")
    
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=cwd,
        text=True
    )
    
    if result.returncode != 0:
        print(f"\n ERROR: Data collection failed at {os.path.basename(script_path)}")
        sys.exit(1)
    
    print(f" SUCCESS: {os.path.basename(script_path)} completed.\n")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the exact execution order for scrapers
    scrapers = [
        (os.path.join(base_dir, "Build_Dataset", "Telegram", "01_telegram.py"), os.path.join(base_dir, "Build_Dataset", "Telegram")),
        (os.path.join(base_dir, "Build_Dataset", "Reddit", "01_reddit.py"), os.path.join(base_dir, "Build_Dataset", "Reddit")),
        (os.path.join(base_dir, "Build_Dataset", "Papers", "01_kyiv_paper.py"), os.path.join(base_dir, "Build_Dataset", "Papers")),
        (os.path.join(base_dir, "Build_Dataset", "Papers", "02_kyiv_paper.py"), os.path.join(base_dir, "Build_Dataset", "Papers")),
        (os.path.join(base_dir, "Build_Dataset", "Papers", "03_bbc_paper.py"), os.path.join(base_dir, "Build_Dataset", "Papers")),
        (os.path.join(base_dir, "Build_Dataset", "Papers", "04_bbc_paper.py"), os.path.join(base_dir, "Build_Dataset", "Papers")),
        (os.path.join(base_dir, "Build_Dataset", "Papers", "05_guardian.py"), os.path.join(base_dir, "Build_Dataset", "Papers"))
    ]
    
    print(" STARTING FULL DATA COLLECTION PIPELINE (Last 1-Year) ")
    for script, cwd in scrapers:
        if not os.path.exists(script):
            print(f" ERROR: Cannot find script {script}")
            sys.exit(1)
        run_script(script, cwd)
        
    print(f"\n{'='*60}")
    print(" DATA COLLECTION COMPLETED SUCCESSFULLY! ")
    print("All raw CSV files are saved in data/raw/")
    print("You can now run: python run_pipeline.py")
    print(f"{'='*60}")
