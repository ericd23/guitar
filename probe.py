# probe.py  ───────────────────────────────────────────────
import time, psutil, threading, contextlib, collections

try:
    import pynvml                          # NVIDIA GPU utilisation
    pynvml.nvmlInit()
    _gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    _gpu = None                            # non‑NVIDIA or import fail

Stat = collections.namedtuple(
    "Stat", "fps cpu_range mean_mem max_mem "
            "gpu_range mean_vram max_vram")

@contextlib.contextmanager
def resource_monitor(label="run", sample_hz=2):
    """
    Context‑manager: background‑samples process & GPU usage.
    Yields a `tick()` callable; call it exactly once per *captured* frame.
    Prints a Stat after the `with`‑block ends.
    """
    proc = psutil.Process()
    t0   = time.perf_counter()
    frames = 0

    cpu, mem = [], []
    gpu, vram = [], []

    def sampler():
        while not stop:
            cpu.append(proc.cpu_percent())
            mem.append(proc.memory_info().rss / 2**20)      # MiB
            if _gpu:
                u = pynvml.nvmlDeviceGetUtilizationRates(_gpu)
                m = pynvml.nvmlDeviceGetMemoryInfo(_gpu)
                gpu.append(u.gpu)                           # %
                vram.append(m.used / 2**20)                 # MiB
            time.sleep(1 / sample_hz)

    stop = False
    th   = threading.Thread(target=sampler, daemon=True)
    th.start()

    def _tick(n=1):
        nonlocal frames
        frames += n

    try:
        yield _tick
    finally:
        stop = True
        th.join()

        dt = time.perf_counter() - t0
        stat = Stat(
            fps       = frames / dt if dt else 0,
            cpu_range = (min(cpu), max(cpu)) if cpu else (0, 0),
            mean_mem  = sum(mem)/len(mem) if mem else 0,
            max_mem   = max(mem) if mem else 0,
            gpu_range = (min(gpu), max(gpu)) if gpu else (0, 0),
            mean_vram = sum(vram)/len(vram) if vram else 0,
            max_vram  = max(vram) if vram else 0,
        )
        print(f"\n[{label}] {stat}")