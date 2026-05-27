# faas_cache_stream_no_simpy.py
import os
import io
import csv
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd

import minute_stats_handler  # your per-minute container status writer

# ===================== CONFIG =====================
OUTPUT_DIR = "result"
INPUT_ZIP  = "/mydata/paper_warmFlex/complete_trace.zip"
CHUNKSIZE  = 10**3

# Cache capacity and costs
MAX_CONTAINERS   = 1000          # total cached capacity
COLD_START_COST  = 1/600     # 100ms cold start
SIZE_DEFAULT     = 10.0         # memory footprint
COST_DEFAULT     = COLD_START_COST

# CSV I/O
USECOLS = ["ID", "function name", "arrival time", "exe time (percentile 50)"]
DTYPES  = {
    "ID": np.int64,
    "function name": "category",
    "arrival time": np.float64,
    "exe time (percentile 50)": np.float64,
}

# Outputs
INVOCATION_OUT = os.path.join(OUTPUT_DIR, "faasCache_output.csv")
DELETION_OUT   = os.path.join(OUTPUT_DIR, "result_container_deletions_faasCache.csv")   # Function name,deletion number
WASTE_OUT      = os.path.join(OUTPUT_DIR, "result_container_waste_faascache.csv")      # func name,waste minutes (one row per eviction-idle)
LIFESPAN_OUT   = os.path.join(OUTPUT_DIR, "result_container_lifeSpan_faascache.csv")             # func name,sum lifeSpan (per chunk)

METHOD_NAME = "FaasCache"  # used by minute_stats_handler


# ===================== UTIL =====================
def _ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def _append_rows(path, header, rows, float_prec=15):
    _ensure_dir(path)
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(header)
    if rows:
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            for r in rows:
                out = []
                for x in r:
                    if isinstance(x, float):
                        out.append(f"{x:.{float_prec}f}")
                    else:
                        out.append(x)
                w.writerow(out)

def iter_zip_csv_chunks(zip_path, chunksize, **read_csv_kwargs):
    try:
        yield from pd.read_csv(zip_path, compression="zip", chunksize=chunksize, **read_csv_kwargs)
        return
    except Exception:
        pass
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith((".csv", ".txt"))]
        if not names:
            raise FileNotFoundError("No CSV/TXT file in ZIP.")
        inner = names[0]
        with zf.open(inner, "r") as raw:
            encoding = read_csv_kwargs.pop("encoding", "utf-8")
            with io.TextIOWrapper(raw, encoding=encoding, newline="") as fh:
                yield from pd.read_csv(fh, chunksize=chunksize, **read_csv_kwargs)


# ===================== GDSF PRIORITY =====================
def gdsf_priority(clock, frequency, cost, size):
    """Priority = Clock + (Freq * Cost) / Size."""
    if size <= 0:
        size = 1.0
    return clock + (frequency * cost) / size


# ===================== CONTAINER MODEL (NO SimPy) =====================
class Container:

    def __init__(self, func_name: str, is_cached: bool):
        self.func_name = func_name
        self.is_cached = is_cached

        # lifecycle times
        self.create_until: float | None = None  # cold-start completion time
        self.busy_until: float = 0.0            # current run finish
        self.last_used: float = 0.0             # last finish time
        self.start_lifespan: float | None = None  # set at first cold-start completion
        self.end_lifespan: float | None = None    # set at eviction or ephemeral destroy
        self.idle_since: float | None = None      # when became 'available' after last run

    def state_at(self, t: float) -> str:
        if self.create_until is not None and t < self.create_until:
            return 'creating'
        if t < self.busy_until:
            return 'busy'
        if self.end_lifespan is not None and t >= self.end_lifespan:
            return 'destroyed'
        # available if created (or warmed) and not destroyed
        if (self.create_until is not None and t >= self.create_until) or (self.busy_until > 0 and t >= self.busy_until):
            return 'available'
        return 'None'


class ServerlessFaasCache:
    def __init__(self, max_containers=MAX_CONTAINERS, cold_start=COLD_START_COST, gc_horizon=60.0):
        self.max_containers = int(max_containers)
        self.cold_start = float(cold_start)
        self.gc_horizon = float(gc_horizon)

        # Greedy-Dual clock
        self.clock_value: float = 0.0

        # Function stats for priority
        self.frequency = defaultdict(int)
        self.cost      = defaultdict(lambda: COST_DEFAULT)
        self.size      = defaultdict(lambda: SIZE_DEFAULT)

        # Active instances (one per function, cached or ephemeral)
        self.instances: dict[str, Container] = {}

        # Cache pool (subset of instances): fn -> cached Container
        self.cache: dict[str, Container] = {}

        # last seen time for GC
        self.last_seen_time: dict[str, float] = {}

        # streaming callbacks (set by driver each chunk)
        self.on_deletion = None            # fn -> None
        self.on_waste = None               # (fn, idle_minutes) -> None
        self.on_lifespan = None            # (fn, lifespan_minutes) -> None

    # ---------- helpers ----------
    @staticmethod
    def r5(x):
        return round(float(x), 5)

    def set_callbacks(self, on_deletion=None, on_waste=None, on_lifespan=None):
        self.on_deletion = on_deletion
        self.on_waste = on_waste
        self.on_lifespan = on_lifespan

    def _maybe_gc(self, now: float):
        to_evict = []
        for fn, c in list(self.cache.items()):
            # consider GC only if it was evicted long ago (end_lifespan set) and unseen
            last_seen = self.last_seen_time.get(fn)
            if c.end_lifespan is not None and (now - c.end_lifespan) > self.gc_horizon:
                if last_seen is None or (now - last_seen) > self.gc_horizon:
                    to_evict.append(fn)
        for fn in to_evict:
            self.cache.pop(fn, None)
            self.instances.pop(fn, None)
            self.frequency.pop(fn, None)
            self.cost.pop(fn, None)
            self.size.pop(fn, None)
            self.last_seen_time.pop(fn, None)

    def _priority(self, fn: str) -> float:
        return gdsf_priority(self.clock_value, self.frequency[fn], self.cost[fn], self.size[fn])

    def _evict_lowest_priority(self, now: float):
        """
        Evict the cached container with minimal priority.
        Clock := evicted priority; record deletion & (idle) waste; close lifespan.
        """
        if not self.cache:
            return
        # compute priorities
        victim_fn = min(self.cache.keys(), key=lambda f: self._priority(f))
        victim = self.cache[victim_fn]
        victim_prio = self._priority(victim_fn)

        # update clock
        self.clock_value = victim_prio

        # record deletion/waste/lifespan
        # lifespan from first creation-complete -> eviction time
        if victim.start_lifespan is not None:
            life = max(0.0, now - victim.start_lifespan)
            if self.on_lifespan:
                self.on_lifespan(victim_fn, life)
        if self.on_deletion:
            self.on_deletion(victim_fn)
        if victim.idle_since is not None and now >= victim.idle_since:
            waste_minutes = now - victim.idle_since
            if self.on_waste:
                self.on_waste(victim_fn, float(waste_minutes))

        # mark as ended and remove from cache
        victim.end_lifespan = now
        self.cache.pop(victim_fn, None)
        # frequency reset if none remain
        self.frequency[victim_fn] = 0

    # ---------- main step ----------
    def process_request(self, req_id: int, fn: str, arrival: float, exe_time: float):
        self.last_seen_time[fn] = arrival
        self.frequency[fn] += 1  # update function frequency first (paper flow)

        # Find or create the active instance for this function
        inst = self.instances.get(fn)

        # If there is no active instance, decide whether to cache this function
        if inst is None:
            admitted = False
            if fn in self.cache:
                admitted = True
                inst = self.cache[fn]
            else:
                cand_prio = self._priority(fn)
                admitted = cand_prio >= self.clock_value
                if admitted:
                    # capacity check: evict one if full
                    if len(self.cache) >= self.max_containers:
                        self._evict_lowest_priority(arrival)
                    inst = Container(fn, is_cached=True)
                    self.cache[fn] = inst
                else:
                    inst = Container(fn, is_cached=False)
            self.instances[fn] = inst
        else:
            # Instance exists (cached or ephemeral). If ephemeral and now admitted,
            # we can convert it to cached; otherwise keep ephemeral.
            if not inst.is_cached and fn not in self.cache:
                cand_prio = self._priority(fn)
                if cand_prio >= self.clock_value:
                    if len(self.cache) >= self.max_containers:
                        self._evict_lowest_priority(arrival)
                    inst.is_cached = True
                    self.cache[fn] = inst  # adopt current instance into cache

        # Determine if this invocation needs cold start
        need_cold = (inst.create_until is None)  # never created before

        # If the instance was previously ended (ephemeral finished) and still in dict,
        # treat as new (safety)
        if inst.end_lifespan is not None and arrival >= inst.end_lifespan:
            # revive as a new generation (ephemeral)
            need_cold = True
            inst.create_until = None
            inst.start_lifespan = None
            inst.end_lifespan = None
            inst.idle_since = None
            inst.busy_until = 0.0
            inst.is_cached = inst.func_name in self.cache  # keep cache flag consistent

        # If arrival is before the end of cold-start, it is still creating
        if need_cold:
            inst.create_until = arrival + self.cold_start
            exe_start = inst.create_until
            # mark container lifespan starting at the end of cold start
            inst.start_lifespan = inst.create_until
        else:
            # delayed warm start if busy
            exe_start = max(arrival, inst.busy_until)

        finish_time = exe_start + exe_time

        # lifecycle updates
        inst.busy_until = finish_time
        inst.last_used = finish_time
        inst.idle_since = finish_time  # becomes available after this run

        # For cold starts (newly created instance), we do NOT record deletion here.
        # Lifespan contribution is recorded at eviction or ephemeral destroy.

        # If instance is ephemeral (not cached), destroy right after finishing
        if not inst.is_cached:
            # lifespan = finish - start_lifespan
            if inst.start_lifespan is not None:
                life = max(0.0, finish_time - inst.start_lifespan)
                if self.on_lifespan:
                    self.on_lifespan(fn, float(life))
            # an ephemeral "deletion" event (not a cache eviction but a destruction)
            if self.on_deletion:
                self.on_deletion(fn)
            # waste for ephemeral is the idle period before destroy (0 here, since destroyed immediately)
            inst.end_lifespan = finish_time
            # remove active instance
            self.instances.pop(fn, None)

        return {
            'ID': int(req_id),
            'function_name': fn,
            'arrival_time': self.r5(arrival),
            'exe time (percentile 50)': self.r5(exe_time),
            'exe_start': self.r5(exe_start),
            'cold_start': int(1 if need_cold else 0),
            'finish_time': self.r5(finish_time),
        }


# ===================== STATUS SNAPSHOT =====================
def count_containers(sim: ServerlessFaasCache, snapshot_time: float):
    ava = busy = exist = 0
    for fn, c in sim.cache.items():
        st = c.state_at(snapshot_time)
        if st == 'available':
            ava += 1; exist += 1
        elif st == 'busy':
            busy += 1; exist += 1
        elif st == 'creating':
            exist += 1
    return ava, busy, exist

# ===================== DRIVER =====================
def run_faas_cache_stream(
    zip_file_path=INPUT_ZIP,
    output_file=INVOCATION_OUT,
    chunk_size=CHUNKSIZE,
    method_name=METHOD_NAME,
    output_dir=OUTPUT_DIR,
    gc_every_n_chunks=50
):
    # prep dirs/files
    os.makedirs(output_dir, exist_ok=True)
    _ensure_dir(output_file)
    if os.path.exists(output_file):
        os.remove(output_file)

    # reset minute handler cursor so runs don't interfere
    if hasattr(minute_stats_handler, "current_minute"):
        minute_stats_handler.current_minute = None

    # write headers once for the three metric files
    _append_rows(DELETION_OUT, ["Function name", "deletion number"], [])
    _append_rows(WASTE_OUT,     ["func name", "waste minutes"],      [])
    _append_rows(LIFESPAN_OUT,  ["func name", "sum lifeSpan"],       [])

    sim = ServerlessFaasCache(max_containers=MAX_CONTAINERS, cold_start=COLD_START_COST)

    first_chunk = True
    per_minute_stats = []

    # per-chunk accumulators (tiny; cleared each chunk)
    del_counts = defaultdict(int)     # fn -> count
    waste_rows = []                   # (fn, idle_minutes)
    life_sums  = defaultdict(float)   # fn -> sum(lifespan)

    # wire streaming callbacks
    def set_callbacks():
        sim.set_callbacks(
            on_deletion=lambda fn: del_counts.__setitem__(fn, del_counts.get(fn, 0) + 1),
            on_waste=lambda fn, idle: waste_rows.append((fn, float(idle))),
            on_lifespan=lambda fn, life: life_sums.__setitem__(fn, life_sums.get(fn, 0.0) + float(life))
        )
    set_callbacks()

    # minute emission cursor (carried across chunks)
    next_minute_to_emit = None  # first minute still to emit

    # Stream trace
    reader = iter_zip_csv_chunks(
        zip_file_path,
        chunk_size,
        usecols=USECOLS,
        dtype=DTYPES,
        engine="c",
        low_memory=False,
        na_filter=False
    )

    for chunk_num, chunk in enumerate(reader, 1):
        # normalize / round / sort
        chunk = chunk.copy()
        chunk["arrival time"] = chunk["arrival time"].astype(float).round(5)
        chunk["exe time (percentile 50)"] = chunk["exe time (percentile 50)"].astype(float).round(5)
        chunk = chunk.sort_values("arrival time", kind="mergesort")

        # clear per-chunk accumulators
        del_counts.clear(); waste_rows.clear(); life_sums.clear()

        results = []

        # minute-driven snapshots & request processing
        for _id, fname, at, exe in chunk.itertuples(index=False, name=None):
            row_minute = int(at)

            if next_minute_to_emit is None:
                next_minute_to_emit = row_minute

            # emit all pending minutes strictly before this row's minute
            while next_minute_to_emit < row_minute:
                t_snap = float(next_minute_to_emit)
                ava, busy, exist = count_containers(sim, t_snap)
                per_minute_stats.append([next_minute_to_emit, ava, busy, exist])

                # advance handler to (minute + 1) so it flushes minute 'next_minute_to_emit'
                per_minute_stats = minute_stats_handler.check_minute_change(
                    next_minute_to_emit + 1, per_minute_stats,
                    output_dir=output_dir, method_name=method_name
                )
                next_minute_to_emit += 1

            # also add a sample for this arrival's minute (so handler can average)
            ava_now, busy_now, exist_now = count_containers(sim, float(at))
            per_minute_stats.append([row_minute, ava_now, busy_now, exist_now])

            # process the request
            row = sim.process_request(int(_id), str(fname), float(at), float(exe))
            results.append(row)

        # write main per-invocation output (5 decimals)
        if results:
            df = pd.DataFrame(results)
            if first_chunk:
                df.to_csv(output_file, index=False, float_format="%.5f")
                first_chunk = False
            else:
                df.to_csv(output_file, mode="a", header=False, index=False, float_format="%.5f")
        del results  # free ASAP

        # write three metric files (rounded to 5 decimals for floats) and clear accumulators
        if del_counts:
            _append_rows(DELETION_OUT, ["Function name", "deletion number"],
                         [(fn, cnt) for fn, cnt in del_counts.items()])
            del_counts.clear()

        if waste_rows:
            rows = [(fn, round(idle, 5)) for fn, idle in waste_rows]
            _append_rows(WASTE_OUT, ["func name", "waste minutes"], rows, float_prec=5)
            waste_rows.clear()

        if life_sums:
            rows = [(fn, round(s, 5)) for fn, s in life_sums.items()]
            _append_rows(LIFESPAN_OUT, ["func name", "sum lifeSpan"], rows, float_prec=5)
            life_sums.clear()

        # periodic GC
        if (chunk_num % gc_every_n_chunks) == 0:
            max_t = float(chunk["arrival time"].max())
            sim._maybe_gc(max_t)


    # optional final snapshot for last open minute
    if next_minute_to_emit is not None:
        ava, busy, exist = count_containers(sim, float(next_minute_to_emit))
        per_minute_stats.append([next_minute_to_emit, ava, busy, exist])

    # final flush for minute stats
    _ = minute_stats_handler.finalize_minute_stats(
        per_minute_stats, output_dir=OUTPUT_DIR, method_name=METHOD_NAME
    )

    return pd.read_csv(output_file)


# ===================== MAIN =====================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _ensure_dir(INVOCATION_OUT)

    # Ensure clean headers for metrics
    _append_rows(DELETION_OUT, ["Function name", "deletion number"], [])
    _append_rows(WASTE_OUT,     ["func name", "waste minutes"],      [])
    _append_rows(LIFESPAN_OUT,  ["func name", "sum lifeSpan"],       [])

    run_faas_cache_stream()
