from datetime import datetime


def get_file_name(path: str) -> str:
    """return the file name without extension given a path
    
    Args:
        path (str): absolute path to the file
    Returns:
        (str): file name
    """

    return path.split("/")[-1].split(".")[0]


def get_date_time() -> str:
    """get the current system date time

    Args:
        None
    Returns:
        (str): date time in the format of Y-m-d-H-M-S
    """

    now = datetime.now()
    return now.strftime("%Y-%m-%d-%H-%M-%S")