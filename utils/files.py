import os


def unique_filename(filename):
    counter = 1
    file_name_parts = os.path.splitext(filename)  # returns ('/path/file', '.ext')
    while 1:
        try:
            fd = os.open(filename, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            return os.fdopen(fd), filename
        except OSError:
            pass
        filename = file_name_parts[0] + '_' + str(counter) + file_name_parts[1]
        counter += 1
