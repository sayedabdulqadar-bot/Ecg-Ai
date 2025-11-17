import time
import os
import requests

# Folder to watch
WATCH_FOLDER = r"C:\Users\sayed\Documents\heart_dataset\ecg_reports"

# Hugging Face Space API endpoint
HF_API = "https://Sayed223-ecgautoreport.hf.space/upload"

def send_to_hf(file_path):
    """Upload a file to Hugging Face Space"""
    if not os.path.exists(file_path):
        print("File does not exist:", file_path)
        return

    print("Uploading:", file_path)
    try:
        with open(file_path, "rb") as f:
            files = {"file": f}
            r = requests.post(HF_API, files=files, timeout=40)
        if r.status_code == 200:
            print("Upload successful:", r.text)
        else:
            print(f"Upload failed (status {r.status_code}):", r.text)
    except requests.exceptions.RequestException as e:
        print("UPLOAD ERROR:", e)
    except Exception as e:
        print("Unexpected error:", e)

def watch_folder(folder_path):
    """Continuously watch a folder and upload new files"""
    known_files = set(os.listdir(folder_path))
    print("Watching folder:", folder_path)

    while True:
        try:
            current_files = set(os.listdir(folder_path))
            new_files = current_files - known_files
            for f in new_files:
                full_path = os.path.join(folder_path, f)
                send_to_hf(full_path)
            known_files = current_files
            time.sleep(2)
        except KeyboardInterrupt:
            print("Stopped watching.")
            break
        except Exception as e:
            print("Error while watching folder:", e)
            time.sleep(5)

if __name__ == "__main__":
    if not os.path.exists(WATCH_FOLDER):
        os.makedirs(WATCH_FOLDER)
    watch_folder(WATCH_FOLDER)
