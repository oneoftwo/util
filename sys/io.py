import os
import shutil


def overwrite_dir(dir_str, ask=True):
    """ 
    overwrite directory
    """
    if dir_str[-1] != '/':
        dir_str += '/'
    is_dir = os.path.isdir(dir_str)
    if is_dir:
        if ask:
            is_rm = input(f'! overwrite {dir_str}? (y/n): ')
        else:
            is_rm = 'y'
        if is_rm in ['y']:
            shutil.rmtree(dir_str)
        else:
            print('exit')
            exit()
    os.mkdir(dir_str)
    return dir_str


if __name__ == '__main__':
    pass 

