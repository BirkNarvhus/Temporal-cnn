from datetime import datetime
import os


class FileHandeling:
    def __init__(self, file_name, rw="w", sub_folder="", root_default="./result/"):
        date = datetime.now()
        dt_string = date.strftime("%d_%m_%Y_%H_%M_%S")
        root = root_default + sub_folder + '/'

        os.makedirs(root, exist_ok=True)
        self.file = open(root + file_name + '-' + dt_string + ".csv", rw)

    def __del__(self):
        self.file.close()

    def write(self, *args):
        text = [str(x) for x in args]
        line = ','.join(text)
        self.file.write(line + '\n')

    def flush(self):
        self.file.flush()
