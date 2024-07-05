from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sleap-sweep")
except PackageNotFoundError:
    # package is not installed
    pass
