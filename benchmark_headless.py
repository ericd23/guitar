#!/usr/bin/env python3
"""
Compare resource usage of head‑less vs GUI runs ‑‑ without touching main.py/env.py.

Example
-------
python benchmark_headless.py                                   \
       --frames 1000 --fps 30 --stride 1                       \
       "python main.py cfg/two_demo.py --note assets/notes/corcovado.json \
        --ckpt pretrained/corcovado --test --headless \
        --record output/video.mp4 --frames 1000"               \
       "python main.py cfg/two_demo.py --note assets/notes/corcovado.json \
        --ckpt pretrained/corcovado --test --frames 1000"

For every quoted command the script
  • samples RAM & GPU memory once per second
  • enforces a wall‑clock time‑out derived from frames/fps/stride
  • prints mean/peak usage and effective FPS
  • emits a JSON block you can copy into papers or READMEs
"""
import subprocess, shlex, time, statistics, sys, json, shutil, argparse

import psutil

# ------------------------------------------------------------------------ GPU utils
def _init_gpu_helpers():
    """returns a function gpu_mb(pid)->int (sum over all GPUs), or dummy."""
    try:
        import pynvml                                     # fast path
        pynvml.nvmlInit()
        def _gpu_mem_mb(pid: int) -> int:
            mb = 0
            for i in range(pynvml.nvmlDeviceGetCount()):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h):
                    if p.pid == pid:
                        mb += p.usedGpuMemory // (1024 * 1024)
            return mb
        return _gpu_mem_mb
    except Exception:
        smi = shutil.which("nvidia-smi")
        if smi is None:                       # no GPU metrics available
            return lambda _pid: 0

        def _gpu_mem_mb(pid: int) -> int:     # fallback via nvidia‑smi
            out = subprocess.check_output(
                [smi, "--query-compute-apps=pid,used_memory",
                      "--format=csv,noheader,nounits"])
            for line in out.decode().splitlines():
                p, mem = line.split(',')
                if int(p) == pid:
                    return int(mem)
            return 0
        return _gpu_mem_mb

gpu_mem_mb = _init_gpu_helpers()
# ------------------------------------------------------------------------


def run_and_monitor(cmd: str, n_frames: int, fps: float, stride: int,
                    grace_sec: float = 5.0):
    """
    Launch *cmd* as a subprocess, sample RAM/GPU mem every second,
    kill it after frames*stride/fps + grace_sec, and return stats dict.
    """
    budget = n_frames * stride / fps + grace_sec
    start  = time.time()

    proc = subprocess.Popen(shlex.split(cmd))
    ps   = psutil.Process(proc.pid)

    rss_log, gpu_log = [], []

    while True:
        # ---------- resource snapshot
        try:
            rss_log.append(ps.memory_info().rss / (1024 * 1024))   # MB
        except psutil.NoSuchProcess:
            break
        gpu_log.append(gpu_mem_mb(proc.pid))

        # ---------- check exit / timeout
        if proc.poll() is not None:            # child exited on its own
            break
        if time.time() - start > budget:       # we reached the wall‑time cap
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            break
        time.sleep(1)

    wall = time.time() - start
    fps_eff = n_frames / wall if n_frames else float("nan")

    def _stats(arr):
        return (max(arr or [0]), statistics.mean(arr or [0]))

    peak_ram, mean_ram = _stats(rss_log)
    peak_gpu, mean_gpu = _stats(gpu_log)

    return dict(
        wall_time_sec=wall,
        fps=fps_eff,
        ram_peak_mb=peak_ram, ram_mean_mb=mean_ram,
        gpu_peak_mb=peak_gpu, gpu_mean_mb=mean_gpu,
        samples=len(rss_log)
    )


# ------------------------------------------------------------------------ CLI
def parse_cli():
    p = argparse.ArgumentParser(
        description="Benchmark head‑less vs GUI IsaacGym runs.")
    p.add_argument("--frames", type=int, required=True,
                   help="Total frames simulated in each run.")
    p.add_argument("--fps", type=float, default=30,
                   help="Nominal simulation FPS (viewer or headless).")
    p.add_argument("--stride", type=int, default=1,
                   help="Capture stride if you skip frames when recording.")
    p.add_argument("commands", nargs="+", help="Quoted python main.py invocations.")
    return p.parse_args()
# ------------------------------------------------------------------------


def main():
    args = parse_cli()

    if not args.commands:
        print("No commands supplied!", file=sys.stderr)
        sys.exit(1)

    results = {}
    for idx, cmd in enumerate(args.commands, 1):
        print(f"\n♢ Run {idx}: {cmd}")
        res = run_and_monitor(cmd, args.frames, args.fps, args.stride)
        results[f"run_{idx}"] = dict(command=cmd, **res)

        print("   {:>5.1f} FPS  |  RAM mean/peak {:>5.0f}/{:>5.0f} MB"
              " | GPU mean/peak {:>5.0f}/{:>5.0f} MB  ({} samples)".format(
              res["fps"], res["ram_mean_mb"], res["ram_peak_mb"],
              res["gpu_mean_mb"], res["gpu_peak_mb"], res["samples"]))

    print("\n=== JSON summary ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()