import os


def touch_dir(wanted_dir):
    not_exist_collection = []
    while os.path.exists(wanted_dir) is not True and wanted_dir != '':
        wanted_dir, step = os.path.split(wanted_dir)
        not_exist_collection.append(step)
        pass

    while len(not_exist_collection) != 0:
        step = not_exist_collection.pop()
        wanted_dir = os.path.join(wanted_dir, step)
        os.mkdir(wanted_dir)
        pass
    pass