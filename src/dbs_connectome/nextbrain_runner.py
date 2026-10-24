"""Run the installed NextBrain script without macOS DataLoader respawning.

Compatible with FreeSurfer's Python 3.8. The installed script and atlas are never
edited. Only data-loading worker count is overridden in this child process;
the supplied --threads still controls torch numerical computation.
"""
import pathlib
import runpy
import sys


def serial_loader(loader):
    def wrapped(*args, **kwargs):
        # The supported upstream script specifies these by keyword.
        if len(args) > 5:
            raise ValueError("Unexpected positional DataLoader worker arguments")
        kwargs["num_workers"] = 0
        kwargs["prefetch_factor"] = None
        kwargs["persistent_workers"] = False
        return loader(*args, **kwargs)
    return wrapped


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Supply the installed NextBrain segment.py followed by its arguments")
    script = pathlib.Path(sys.argv[1]).resolve(strict=True)
    if script.name != "segment.py" or script.parent.name != "scripts" or script.parent.parent.name != "ERC_bayesian_segmentation":
        raise ValueError("Expected the installed FreeSurfer NextBrain segmentation script")
    import torch.utils.data
    torch.utils.data.DataLoader = serial_loader(torch.utils.data.DataLoader)
    sys.argv = [str(script)] + sys.argv[2:]
    print("NextBrain adapter: serial atlas loading; numerical --threads unchanged; installed sources unmodified", flush=True)
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
