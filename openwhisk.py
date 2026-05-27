# openwhisk_stream_no_simpy.py
import os
import io
import csv
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd

import minute_stats_handler  # minute snapshots helper

# ===================== CONFIG =====================
OUTPUT_DIR = "."
INPUT_ZIP  = "/mydata/paper_warmFlex/complete_trace.zip"
CHUNKSIZE  = 10**3

USECOLS = ["ID", "function name", "arrival time", "exe time (percentile 50)"]
DTYPES  = {
    "ID": np.int64,
    "function name": "category",
    "arrival time": np.float64,
    "exe time (percentile 50)": np.float64,
}

INVOCATION_OUT = os.path.join(OUTPUT_DIR, "result/openwhisk_output.csv")
DELETION_OUT   = os.path.join(OUTPUT_DIR, "result/result_container_deletions_openwhisk.csv")
WASTE_OUT      = os.path.join(OUTPUT_DIR, "result/result_container_waste_openwhisk.csv")
LIFESPAN_OUT   = os.path.join(OUTPUT_DIR, "result/result_container_lifeSpan_openwhisk.csv")

METHOD_NAME = "openwhisk"  # used by minute_stats_handler

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
        yield from pd.read_csv(
            zip_path,
            compression="zip",
            chunksize=chunksize,
            **read_csv_kwargs
        )
        return
    except Exception:
        pass
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith((".csv", ".txt"))]
        if not names:
            raise FileNotFoundError("No CSV/TXT file found inside the ZIP.")
        inner_name = names[0]
        with zf.open(inner_name, "r") as raw:
            encoding = read_csv_kwargs.pop("encoding", "utf-8")
            with io.TextIOWrapper(raw, encoding=encoding, newline="") as text_fh:
                yield from pd.read_csv(
                    text_fh,
                    chunksize=chunksize,
                    **read_csv_kwargs
                )

# ===================== CONTAINER MODEL =====================
class Container:
    """
    Single container per function with explicit lifecycle times.
    State (derived at query-time): 'None', 'creating', 'busy', 'available', 'destroyed'
    """
    idle_time_minutes = 10.0  # OpenWhisk fixed keep-alive

    def __init__(self, function_name: str):
        self.function_name = function_name
        self.create_until: float | None = None   # when cold start completes
        self.busy_until: float = 0.0             # when current run finishes
        self.last_used: float = 0.0              # last finish time
        self.destroyed_at: float | None = None   # when idle window ends
        self.state_cached = 'None'

        # generation tracking for deletion/waste
        self.gen_id = 0
        self.gen_used_after_creation = False  # becomes True on first warm reuse after cold start

class ServerlessOpenWhisk:

    def __init__(self, cold_start_time=1/600, gc_horizon=60.0):
        self.Tp = float(cold_start_time)
        self.gc_horizon = float(gc_horizon)

        self.containers: dict[str, Container] = {}
        self.last_seen_time: dict[str, float] = {}

        # streaming callbacks (set by driver each chunk)
        self.on_deletion = None            # fn -> None
        self.on_waste = None               # (fn, idle_minutes) -> None
        self.on_lifespan = None            # (fn, lifespan_minutes) -> None

    # ---------- helpers ----------
    @staticmethod
    def r5(x):  # 5-decimal rounding for main CSV
        return round(float(x), 5)

    @staticmethod
    def get_state_at(c: Container | None, t: float) -> str:
        if c is None:
            return 'None'
        if c.create_until is not None and t < c.create_until:
            return 'creating'
        if t < c.busy_until:
            return 'busy'
        if c.destroyed_at is not None and t >= c.destroyed_at:
            return 'destroyed'
        return 'available'

    def set_callbacks(self, on_deletion=None, on_waste=None, on_lifespan=None):
        self.on_deletion = on_deletion
        self.on_waste = on_waste
        self.on_lifespan = on_lifespan

    def _maybe_gc(self, now: float):
        # Drop containers unseen and destroyed past the horizon
        to_del = []
        for fn, c in self.containers.items():
            destroyed_long_ago = (c.destroyed_at is not None) and (now - c.destroyed_at > self.gc_horizon)
            unseen_long_ago = (fn not in self.last_seen_time) or (now - self.last_seen_time[fn] > self.gc_horizon)
            if destroyed_long_ago and unseen_long_ago:
                to_del.append(fn)
        for fn in to_del:
            del self.containers[fn]
            self.last_seen_time.pop(fn, None)

    # ---------- core step ----------
    def process_request(self, req_id: int, fn: str, arrival: float, exe_time: float):
        self.last_seen_time[fn] = arrival

        extra_time = Container.idle_time_minutes  # fixed keep-alive

        c = self.containers.get(fn)

        # detect deletion/waste at next sighting after expiry
        if c is not None and c.busy_until <= arrival and c.destroyed_at is not None and arrival >= c.destroyed_at:
            if self.on_deletion:
                self.on_deletion(fn)
            if not c.gen_used_after_creation and self.on_waste:
                idle_minutes = c.destroyed_at - c.last_used
                self.on_waste(fn, float(idle_minutes))
            # keep object; new generation may start now

        need_cold = False
        if c is None:
            c = Container(fn)
            self.containers[fn] = c
            need_cold = True
        else:
            if c.destroyed_at is not None and arrival >= c.destroyed_at:
                need_cold = True
            elif c.busy_until == 0.0:
                need_cold = True
            else:
                # OpenWhisk fixed idle window: cold if we've exceeded busy_until + idle_time
                if arrival > c.busy_until + extra_time:
                    need_cold = True

        # choose exec start
        if need_cold:
            c.gen_id += 1
            c.gen_used_after_creation = False
            c.create_until = arrival + self.Tp
            exe_start = c.create_until
        else:
            exe_start = max(arrival, c.busy_until)  # must wait if busy

        finish_time = exe_start + exe_time

        # lifecycle update
        c.busy_until = finish_time
        c.last_used = finish_time
        c.destroyed_at = finish_time + extra_time

        # lifespan metric for cold starts: from creation (arrival) to finish
        if need_cold and self.on_lifespan:
            self.on_lifespan(fn, float(finish_time - arrival))

        # mark warm reuse
        if not need_cold and arrival >= c.busy_until - exe_time:
            c.gen_used_after_creation = True

        # return main row
        return {
            'ID': int(req_id),
            'function_name': fn,
            'arrival_time': self.r5(arrival),
            'exe time (percentile 50)': self.r5(exe_time),
            'exe_start': self.r5(exe_start),
            'cold_start': int(1 if need_cold else 0),
            'finish_time': self.r5(finish_time),
            'warmup_duration': self.r5(extra_time),
        }

# ===================== STATUS SNAPSHOT =====================
def count_containers(sim: ServerlessOpenWhisk, snapshot_time: float):
    ava = busy = exist = 0
    for fn, c in sim.containers.items():
        st = ServerlessOpenWhisk.get_state_at(c, snapshot_time)
        if st == 'available':
            ava += 1; exist += 1
        elif st == 'busy':
            busy += 1; exist += 1
        elif st == 'creating':
            exist += 1
    return ava, busy, exist

# ===================== DRIVER =====================
def run_openwhisk_stream(
    zip_file_path=INPUT_ZIP,
    output_file=INVOCATION_OUT,
    chunk_size=CHUNKSIZE,
    method_name=METHOD_NAME,
    output_dir=OUTPUT_DIR,
    gc_every_n_chunks=50
):
    # prep dirs/files
    _ensure_dir(output_file)
    _ensure_dir(os.path.join(output_dir, "dummy.txt"))
    if os.path.exists(output_file):
        os.remove(output_file)

    # write headers for the three per-chunk outputs once
    _append_rows(DELETION_OUT, ["Function name", "deletion number"], [])
    _append_rows(WASTE_OUT,     ["func name", "waste minutes"],      [])
    _append_rows(LIFESPAN_OUT,  ["func name", "sum lifeSpan"],       [])

    sim = ServerlessOpenWhisk()

    first_chunk = True

    per_minute_stats = []

    # per-chunk accumulators (cleared every chunk)
    del_counts = defaultdict(int)   # fn -> count
    waste_rows = []                 # (fn, idle_minutes)
    life_sums  = defaultdict(float) # fn -> sum(lifespan)

    def set_streaming_callbacks():
        sim.set_callbacks(
            on_deletion=lambda fn: del_counts.__setitem__(fn, del_counts.get(fn, 0) + 1),
            on_waste=lambda fn, idle: waste_rows.append((fn, float(idle))),
            on_lifespan=lambda fn, life: life_sums.__setitem__(fn, life_sums.get(fn, 0.0) + float(life)),
        )

    # minute emission cursor (carried across chunks)
    next_minute_to_emit = None  # first minute still to emit

    # stream from ZIP
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

        # reset per-chunk accumulators + callbacks
        del_counts.clear(); waste_rows.clear(); life_sums.clear()
        set_streaming_callbacks()

        results = []

        # --- emit per-minute snapshots BEFORE processing each row (minute boundary driven) ---
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

            # process current row
            row = sim.process_request(int(_id), str(fname), float(at), float(exe))
            results.append(row)

        # write main output for this chunk (5 decimals)
        if results:
            df = pd.DataFrame(results)
            if first_chunk:
                df.to_csv(output_file, index=False, float_format="%.5f")
                first_chunk = False
            else:
                df.to_csv(output_file, mode="a", header=False, index=False, float_format="%.5f")
        del results  # free asap

        # three per-chunk metric files (rounded to 5 decimals where floats)
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

        # periodic GC to keep memory flat
        if (chunk_num % gc_every_n_chunks) == 0:
            max_t = float(chunk["arrival time"].max())
            sim._maybe_gc(max_t)


    # flush the last open minute snapshot (start-of-minute view)
    if next_minute_to_emit is not None:
        ava, busy, exist = count_containers(sim, float(next_minute_to_emit))
        per_minute_stats.append([next_minute_to_emit, ava, busy, exist])

    # final flush of minute stats
    _ = minute_stats_handler.finalize_minute_stats(
        per_minute_stats, output_dir=output_dir, method_name=method_name
    )

    return pd.read_csv(output_file)

# ===================== MAIN =====================
if __name__ == "__main__":
    _ensure_dir(INVOCATION_OUT)
    run_openwhisk_stream()
