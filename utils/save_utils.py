import os
import json


def append_to_json_list(file, new_object):
    if not os.path.exists(file):
        with open(file, 'w') as f:
            f.write('[]')
    
    with open(file, 'r+') as f:
        f.seek(0, os.SEEK_END)
        pos = f.tell() - 1
        
        while pos > 0:
            f.seek(pos, os.SEEK_SET)
            char = f.read(1)
            if char not in [' ', '\n', '\r']:
                break
            pos -= 1
        
        if char == ']':
            if pos > 1:
                f.seek(pos, os.SEEK_SET)
                f.write(',\n')
            else:
                f.seek(pos, os.SEEK_SET)
        else:
            raise ValueError("Invalid JSON format in file.")
        