import os

def make_dir(path):
    if not os.path.exists(path):
        if not os.path.exists(path):
            os.makedirs(path)
        print(path, "is buildding")
    if os.path.exists(path):
        print(path, "have been build")

def is_exist_dir(path):
    if not os.path.exists(path):
        return False
    if os.path.exists(path):
        return True

def join_a_and_b_dir(a,b):
    return os.path.join(a,b)