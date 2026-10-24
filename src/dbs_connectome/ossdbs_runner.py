"""Executed only by the separately configured OSS-DBS Python environment."""
import json
import sys


def main():
    import ngsolve
    import ossdbs
    from ossdbs.main import main_run
    config_path, threads = sys.argv[1:]
    ngsolve.SetNumThreads(int(threads))
    ossdbs.set_logger()
    with open(config_path, encoding="utf-8") as stream:
        settings = json.load(stream)
    main_run(settings)


if __name__ == "__main__":
    main()
