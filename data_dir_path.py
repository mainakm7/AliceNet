import os

def data_dir_path(subdir="processed"):
    """
    Constructs the absolute path to a specified subdirectory within 'data'.
    
    This function starts by getting the current working directory, then 
    navigates to the parent directory and appends the relative path 'data/<subdir>'
    to it. The resulting absolute path is returned.

    Args:
        subdir (str): The name of the subdirectory within 'data'. Defaults to 'processed'.

    Returns:
        str: The absolute path to the specified subdirectory within 'data'.
    """
    data_path = os.path.join('data', subdir)
    curr_path = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(curr_path, os.pardir))
    data_path_whole = os.path.join(parent_dir, data_path)
    
    return data_path_whole
