import os
import sys
import urllib.request
import hashlib

MODELS = {
    "best_model_RIDGE_manet.pth": {
        "url": "https://media.githubusercontent.com/media/Sohamgujar71/HVD_ROP_EPICS_Deploy/main/outputs/best_model_RIDGE.pth",
        "description": "Demarcation Ridge Model (MAnet / UNet++ EfficientNet-B4)",
        "expected_size": 103349772, # will preserve local MAnet weight if available
    },
    "best_model_OD.pth": {
        "url": "https://media.githubusercontent.com/media/Sohamgujar71/HVD_ROP_EPICS_Deploy/main/outputs/best_model_OD.pth",
        "description": "Optic Disc (OD) Model (UNet++ EfficientNet-B4)",
        "expected_size": 84063803,
    },
    "optimized_best_model_BV.pth": {
        "url": "https://media.githubusercontent.com/media/Sohamgujar71/HVD_ROP_EPICS_Deploy/main/outputs/optimized_best_model_BV.pth",
        "description": "Blood Vessels (BV) Model (UNet++ EfficientNet-B4)",
        "expected_size": 84073011,
    }
}

def download_file(url, target_path, description):
    print(f"\nDownloading {description}...")
    print(f"Target: {target_path}")
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    req = urllib.request.Request(url, headers=headers)
    
    with urllib.request.urlopen(req) as resp, open(target_path, 'wb') as out_file:
        total_length = resp.headers.get('content-length')
        if total_length is None:
            out_file.write(resp.read())
        else:
            total_length = int(total_length)
            downloaded = 0
            block_size = 1024 * 1024 # 1 MB
            while True:
                buffer = resp.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)
                percent = (downloaded / total_length) * 100
                mb_down = downloaded / (1024 * 1024)
                mb_total = total_length / (1024 * 1024)
                sys.stdout.write(f"\rProgress: {mb_down:.1f} MB / {mb_total:.1f} MB ({percent:.1f}%)")
                sys.stdout.flush()
    print("\nDownload complete.")

def ensure_models(model_dir="model"):
    os.makedirs(model_dir, exist_ok=True)
    
    # Check if local outputs/best_model_RIDGE_manet.pth exists and copy if needed
    local_ridge_src = os.path.join("outputs", "best_model_RIDGE_manet.pth")
    target_ridge = os.path.join(model_dir, "best_model_RIDGE_manet.pth")
    if not os.path.exists(target_ridge) and os.path.exists(local_ridge_src):
        import shutil
        print(f"Copying local ridge model from {local_ridge_src} to {target_ridge}...")
        shutil.copyfile(local_ridge_src, target_ridge)

    for filename, info in MODELS.items():
        # Handle ridge specifically
        if filename == "best_model_RIDGE_manet.pth":
            target_path = os.path.join(model_dir, filename)
            if os.path.exists(target_path) and os.path.getsize(target_path) > 1000000:
                print(f"[OK] {filename} is ready ({os.path.getsize(target_path) / (1024*1024):.1f} MB).")
                continue
        
        target_path = os.path.join(model_dir, filename)
        # Verify if file exists and is not just a git-lfs pointer (< 1KB)
        if os.path.exists(target_path) and os.path.getsize(target_path) > 1000000:
            print(f"[OK] {filename} is ready ({os.path.getsize(target_path) / (1024*1024):.1f} MB).")
            continue
            
        try:
            download_file(info["url"], target_path, info["description"])
        except Exception as e:
            print(f"Failed to download {filename}: {e}", file=sys.stderr)
            if os.path.exists(target_path):
                os.remove(target_path)

if __name__ == "__main__":
    ensure_models()
