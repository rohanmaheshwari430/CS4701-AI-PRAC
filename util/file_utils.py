import os
import wget

def download_file(url, destination):
    if not os.path.exists(destination):
        wget.download(url, out=destination)

def clone_repository(url, destination):
    if not os.path.exists(destination):
        os.system(f"git clone {url} {destination}")
