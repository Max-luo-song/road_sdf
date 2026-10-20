import os
from enum import Enum


class InputType(Enum):
    NOT_EXIST = 0
    IMAGE = 1
    VIDEO = 2
    FOLDER = 6
    UNKNOWN_FILE = -1


def check_input_type(input_path: str) -> InputType:
    if os.path.isdir(input_path):
        return InputType.FOLDER
    elif os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:
            return InputType.IMAGE
        elif ext in [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]:
            return InputType.VIDEO
        else:
            return InputType.UNKNOWN_FILE
    else:
        return InputType.NOT_EXIST
