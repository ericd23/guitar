import argparse, importlib

from probe import resource_monitor
import main                               # your existing entry point module

def one_run(label, headless, args):
    with resource_monitor(label) as tick:
        main.run_sim(headless=headless, tick=tick, **vars(args))

def main_cli():
    p = argparse.ArgumentParser()
    #  add *exactly* the CLI flags you already support  … ↓
    p.add_argument("--config", required=True)
    p.add_argument("--record", default="headless.mp4")
    p.add_argument("--frames", type=int, default=3000)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--capture-stride", type=int, default=1)
    p.add_argument("--greyscale", action="store_true")
    parsed = p.parse_args()

    one_run("GUI",       False, parsed)
    one_run("HEADLESS",  True,  parsed)

if __name__ == "__main__":
    main_cli()