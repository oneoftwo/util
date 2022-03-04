

def overwrite_dir(dir_str):
    """ 
    overwrite directory
    """
    if not dir_str == '/':
        dir_str += '/'
    try:
        is_rm = input(f'overwrite {args.save_dir} (y/n): ')
        if is_rm == 'y':
            shutil.rmtree(dir_str)
    except:
        pass 
    os.mkdir(dir_str)
    return dir_str


