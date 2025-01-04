import os

from ..constants import PROGRESS_MEMORY_DIRECTORY

class FileUtils:
    @staticmethod
    def get_file_path(file_name: str, sub_dir: str = PROGRESS_MEMORY_DIRECTORY) -> str:
        # Get the directory of the current file (this file)
        current_file_dir = os.path.dirname(os.path.abspath(__file__))

        # Move one level up to the parent directory
        parent_dir = os.path.dirname(current_file_dir)

        # Build the path to the target subdirectory (progress_memory or any subdirectory)
        target_dir = os.path.join(parent_dir, sub_dir)

        # Build the full path to the file
        file_path = os.path.join(target_dir, file_name)

        return file_path