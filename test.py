
import os

cur_dir = os.getcwd()

def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

find_key = find("service_key.json", cur_dir)
os.system(f'export GOOGLE_APPLICATION_CREDENTIALS="{find_key}"')
