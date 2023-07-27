import os


def get_package_root() -> str:
    return os.path.dirname(os.path.realpath(__file__))


def get_repo_root() -> str:
    return os.path.dirname(get_package_root())
