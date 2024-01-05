from PIL import Image
import os

def clear_dir(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            clear_dir(file_path)
            os.rmdir(file_path)
