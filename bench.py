#!/usr/bin/env python3
"""
Example:
  python benchmark_headless.py \
      "python main.py cfg/two_demo.py --note ./assets/notes/corcovado.json \
       --ckpt ./pretrained/corcovado --test --headless \
       --record output/video.mp4 --frames 1000" 1000 \
      "python main.py cfg/two_demo.py --note ./assets/notes/corcovado.json \
       --ckpt ./pretrained/corcovado --test --frames 1000" 1000
"""
import subprocess, shlex, time, statistics, sys, json, shutil
import psutil

# --------------------------------------------------------------------- GPU helpers
try:
    import pynvml
    pynvml.nvmlInit()
    GET_GPU = "pynvml"

    def gpu_mem_mb(pid):
        mb = 0
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h):
                if p.pid == pid: mb += p.usedGpuMemory // (1024*1024)
        return mb

except Exception:
    GET_GPU = "nvidia-smi"

    def gpu_mem_mb(pid):
        smi = shutil.which("nvidia-smi")
        if smi is None: return 0
        out = subprocess.check_output(
            [smi, "--query-compute-apps=pid,used_memory",
                   "--format=csv,noheader,nounits"])
        for line in out.decode().splitlines():
            p,s = line.split(',')
            if int(p)==pid: return int(s)
        return 0
# ---------------------------------------------------------------------


def run_and_monitor(cmd, n_frames):
    start = time.time()
    proc  = subprocess.Popen(shlex.split(cmd))
    ps    = psutil.Process(proc.pid)

    rss_samples, gpu_samples = [], []
    while proc.poll() is None:
        try:
            rss_samples.append(ps.memory_info().rss / (1024*1024))      # MB
        except psutil.NoSuchProcess:
            break
        gpu_samples.append(gpu_mem_mb(proc.pid))
        time.sleep(1)

    wall = time.time() - start
    fps  = n_frames / wall if n_frames else float("nan")
    stats = lambda arr: (max(arr or [0]), statistics.mean(arr or [0]))
    peak_ram, mean_ram = stats(rss_samples)
    peak_gpu, mean_gpu = stats(gpu_samples)

    return dict(time=wall, fps=fps,
                ram_peak=peak_ram, ram_mean=mean_ram,
                gpu_peak=peak_gpu, gpu_mean=mean_gpu)


def main():
    if len(sys.argv) < 3 or len(sys.argv)%2!=1:
        print("usage: benchmark_headless.py  \"<cmd1>\"  <frames1> "
              "\"<cmd2>\" <frames2> ...", file=sys.stderr)
        sys.exit(1)

    results = {}
    for i in range(1, len(sys.argv), 2):
        cmd, frames = sys.argv[i], int(sys.argv[i+1])
        print(f"\nRunning #{(i+1)//2}: {cmd}")
        res = run_and_monitor(cmd, frames)
        results[f"run_{(i+1)//2}"] = dict(command=cmd, **res)
        print(f"  ▸ {res['fps']:.1f} FPS  | "
              f"RAM mean/peak {res['ram_mean']:.0f}/{res['ram_peak']:.0f} MB | "
              f"GPU mean/peak {res['gpu_mean']:.0f}/{res['gpu_peak']:.0f} MB")

    print("\n=== JSON summary ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()