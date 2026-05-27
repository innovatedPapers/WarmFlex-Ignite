import os
import csv
import zipfile
from collections import defaultdict
import pandas as pd
import numpy as np
import io

import WarmUp_Fuzzification
import minute_stats_handler    # minute snapshots helper

# ===================== CONFIG =====================
INPUT_ZIP  = "/mydata/paper_warmFlex/complete_trace.zip"
CHUNKSIZE  = 10**3
OUTPUT_DIR = "result"

USECOLS = ["ID", "function name", "arrival time", "exe time (percentile 50)"]
DTYPES  = {
    "ID": "int64",
    "function name": "category",
    "arrival time": "float64",
    "exe time (percentile 50)": "float64",
}

INVOCATION_OUT = os.path.join(OUTPUT_DIR, "warmFlex_output.csv")
DELETION_OUT   = os.path.join(OUTPUT_DIR, "result_container_deletions_warmFlex.csv")
WASTE_OUT      = os.path.join(OUTPUT_DIR, "result_container_waste_warmFlex.csv")
LIFESPAN_OUT   = os.path.join(OUTPUT_DIR, "result_container_lifeSpan_warmFlex.csv")

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
    # Fast path: let pandas stream the CSV directly from the zip
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
    # Fallback: manual stream
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith((".csv", ".txt"))]
        if not names:
            raise FileNotFoundError("No CSV/TXT file found inside the ZIP.")
        inner_name = names[0]
        with zf.open(inner_name, "r") as raw:
            with io.TextIOWrapper(raw, encoding=read_csv_kwargs.get("encoding", "utf-8"), newline="") as text_fh:
                yield from pd.read_csv(
                    text_fh,
                    chunksize=chunksize,
                    **read_csv_kwargs
                )

# ===================== CONTAINER MODEL =====================
class Container:
    """
    Single container per function with explicit lifecycle times.
    States (query-time): 'None', 'creating', 'busy', 'available', 'destroyed'
    """
    def __init__(self, function_name):
        self.function_name = function_name
        self.create_until = None   # when cold start completes
        self.busy_until = 0.0      # last finish time of an invocation
        self.last_used = 0.0       # alias of busy_until (for clarity)
        self.destroyed_at = None   # scheduled deletion time (finish + keep-alive)

        # generation tracking for deletion/waste/lifespan
        self.gen_id = 0
        self.gen_used_after_creation = False  # becomes True on first warm reuse after cold start
        self.gen_created_at = None            # moment this generation became *available* (post cold start)

class ServerlessWarmFlex:

    def __init__(self, cold_start_time=1/600, gc_horizon=60.0):
        self.Tp = float(cold_start_time)  # 100ms -> minutes
        self.gc_horizon = float(gc_horizon)

        self.containers = {}                     # fn -> Container
        self.last_arrival_time = {}              # fn -> last arrival time (for inter-arrival)
        self.warmup_time = {}                    # fn -> extra keep-alive minutes (cached on first use)

        self.last_seen_time = {}                 # fn -> last arrival time seen

        # streaming callbacks (set by the driver each chunk)
        self.on_deletion = None
        self.on_waste = None
        self.on_lifespan = None

    # ---------- helpers ----------
    @staticmethod
    def r5(x):  # 5-decimal rounding for main CSV
        return round(float(x), 5)

    def get_state_at(self, c: Container, t: float) -> str:
        if c is None:
            return 'None'
        if c.create_until is not None and t < c.create_until:
            return 'creating'
        if t < c.busy_until:
            return 'busy'
        if c.destroyed_at is not None and t >= c.destroyed_at:
            return 'destroyed'
        if t >= c.busy_until and (c.destroyed_at is None or t < c.destroyed_at):
            return 'available'
        return 'None'

    def _compute_keepalive(self, fn: str, arrival: float) -> float:
        # Cache per function to keep it stable for a while (you can change if desired)
        if fn not in self.warmup_time:
            last = self.last_arrival_time.get(fn, None)
            # If first time, treat as long gap (normalize will cap anyway in your fuzzifier)
            inter_arrival = (arrival - last) if last is not None else 60.0
            keep = float(WarmUp_Fuzzification.fuzzy_conclusion(fn, float(inter_arrival)))
            if not np.isfinite(keep) or keep < 0:
                keep = 0.0
            self.warmup_time[fn] = keep
        return self.warmup_time[fn]

    def _maybe_gc(self, now: float):
        to_del = []
        for fn, c in self.containers.items():
            destroyed_long_ago = (c.destroyed_at is not None) and (now - c.destroyed_at > self.gc_horizon)
            unseen_long_ago = (fn not in self.last_seen_time) or (now - self.last_seen_time[fn] > self.gc_horizon)
            if destroyed_long_ago and unseen_long_ago:
                # Emit lifespan if not yet emitted (e.g., if there was no later arrival to trigger deletion reporting)
                if self.on_lifespan and c.gen_created_at is not None:
                    self.on_lifespan(fn, float(c.destroyed_at - c.gen_created_at))
                to_del.append(fn)

        for fn in to_del:
            del self.containers[fn]
            self.warmup_time.pop(fn, None)
            self.last_arrival_time.pop(fn, None)
            self.last_seen_time.pop(fn, None)

    # ---------- main processing ----------
    def process_request(self, _id: int, fn: str, arrival: float, exe_time: float):
 
        self.last_seen_time[fn] = arrival
        c = self.containers.get(fn, None)

        # If a previous generation already expired before this arrival, emit deletion (& metrics)
        if c is not None and c.busy_until <= arrival and c.destroyed_at is not None and arrival >= c.destroyed_at:
            if self.on_deletion:
                self.on_deletion(fn)
            # lifespan = destroyed_at - gen_created_at
            if self.on_lifespan and c.gen_created_at is not None:
                self.on_lifespan(fn, float(c.destroyed_at - c.gen_created_at))
            # If the generation was never reused after creation, its keep-alive time is wasted
            if not c.gen_used_after_creation and self.on_waste:
                idle_minutes = c.destroyed_at - c.last_used  # equals scheduled keep-alive
                self.on_waste(fn, float(idle_minutes))

        # compute keep-alive for this function
        extra_time = self._compute_keepalive(fn, arrival)

        # Determine if we need a cold start
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
                if arrival > c.busy_until + extra_time:
                    need_cold = True

        # Pick exe_start and update generation state
        if need_cold:
            c.gen_id += 1
            c.gen_used_after_creation = False
            c.create_until = arrival + self.Tp
            # --- LIFESPAN START POINT ---
            c.gen_created_at = c.create_until
            exe_start = c.create_until
            cold = 1
        else:
            # delayed warm start: can't start before last finish
            exe_start = max(arrival, c.busy_until)
            c.gen_used_after_creation = True
            cold = 0

        # finish and lifecycle
        finish_time = exe_start + exe_time
        c.busy_until = finish_time
        c.last_used = finish_time
        c.destroyed_at = finish_time + extra_time

        # track inter-arrival for next time
        self.last_arrival_time[fn] = arrival

        return [
            int(_id),
            str(fn),
            self.r5(arrival),
            self.r5(exe_time),
            self.r5(exe_start),
            self.r5(finish_time),
            cold,
            self.r5(extra_time),    # warmup_duration (keep-alive)
        ]

# ===================== SNAPSHOT HELPERS =====================
def count_containers(sim: ServerlessWarmFlex, snapshot_time: float):
    ava = busy = exist = 0
    for fn, c in sim.containers.items():
        st = sim.get_state_at(c, snapshot_time)
        if st == 'available':
            ava += 1; exist += 1
        elif st == 'busy':
            busy += 1; exist += 1
        elif st == 'creating':
            exist += 1
    return ava, busy, exist

# ===================== DRIVER =====================
def run_warmflex_stream(
    zip_file_path=INPUT_ZIP,
    output_file=INVOCATION_OUT,
    chunk_size=CHUNKSIZE,
    method_name="warmFlex", # used by minute_stats_handler
    output_dir=OUTPUT_DIR,
    gc_every_n_chunks=50
):
    # prep files
    _ensure_dir(output_file)
    _ensure_dir(os.path.join(output_dir, "dummy.txt"))
    if os.path.exists(output_file):
        os.remove(output_file)

    # write headers for the three per-chunk metric files
    for p, hdr in [
        (DELETION_OUT, ["Function name", "deletion number"]),
        (WASTE_OUT,    ["func name", "waste minutes"]),
        (LIFESPAN_OUT, ["func name", "sum lifeSpan"]),
    ]:
        if os.path.exists(p):
            os.remove(p)
        _append_rows(p, hdr, [])

    # simulator
    sim = ServerlessWarmFlex()

    # rolling minute snapshots to be flushed via minute_stats_handler
    per_minute_stats = []            # rows: [minute, available_count, busy_count, exist_count]
    next_minute_to_emit = None       # int minute cursor

    # per-chunk aggregations
    del_counts  = defaultdict(int)   # fn -> deletions in this chunk
    waste_rows  = []                 # list of (fn, idle_minutes) for this chunk
    life_sums   = defaultdict(float) # fn -> sum lifespan in this chunk

    # expose callbacks to sim
    def set_streaming_callbacks():
        sim.on_deletion = lambda fn: del_counts.__setitem__(fn, del_counts[fn] + 1)
        sim.on_waste    = lambda fn, idle: waste_rows.append((fn, float(idle)))
        sim.on_lifespan = lambda fn, life: life_sums.__setitem__(fn, life_sums[fn] + float(life))

    set_streaming_callbacks()

    first_chunk = True
    chunks_processed = 0

    # stream the dataset
    for chunk in iter_zip_csv_chunks(
        zip_file_path,
        chunksize=chunk_size,
        usecols=USECOLS,
        dtype=DTYPES
    ):
        if chunk.empty:
            continue

        # enforce arrival time order within chunk
        chunk = chunk.sort_values("arrival time", kind="mergesort")

        # lightweight GC periodically
        chunks_processed += 1
        if (chunks_processed % gc_every_n_chunks) == 0:
            try:
                max_t = float(chunk["arrival time"].max())
                sim._maybe_gc(max_t)
            except Exception:
                pass

        # prepare per-chunk accumulators
        del_counts.clear(); waste_rows.clear(); life_sums.clear()
        set_streaming_callbacks()

        results = []

        # --- stream rows and emit per-minute snapshots BEFORE processing each row
        for _id, fname, at, exe in chunk.itertuples(index=False, name=None):
            row_minute = int(at)

            # initialize cursor on the very first row seen overall
            if next_minute_to_emit is None:
                next_minute_to_emit = row_minute

            # emit all pending minutes strictly before this row's minute
            while next_minute_to_emit < row_minute:
                t_snap = float(next_minute_to_emit)
                ava, busy, exist = count_containers(sim, t_snap)
                per_minute_stats.append([next_minute_to_emit, ava, busy, exist])

                # advance the handler so it flushes minute 'next_minute_to_emit'
                per_minute_stats = minute_stats_handler.check_minute_change(
                    next_minute_to_emit + 1, per_minute_stats,
                    output_dir=output_dir, method_name=method_name
                )
                next_minute_to_emit += 1

            # now process the current row
            row = sim.process_request(int(_id), str(fname), float(at), float(exe))
            results.append(row)

        # write main output for this chunk (5 decimals)
        if results:
            df = pd.DataFrame(
                results,
                columns=[
                    "ID", "function name", "arrival time",
                    "exe time (percentile 50)", "exe_start",
                    "finish_time", "cold", "warmup_duration"
                ]
            )
            if first_chunk:
                df.to_csv(output_file, index=False, float_format="%.5f")
                first_chunk = False
            else:
                df.to_csv(output_file, mode="a", header=False, index=False, float_format="%.5f")
        del results  # free asap

        # --- three per-chunk metric files (with 5-dec rounding for waste & lifespan) ---
        if del_counts:
            _append_rows(DELETION_OUT, ["Function name", "deletion number"],
                         [(fn, cnt) for fn, cnt in del_counts.items()])  # counts are ints
            del_counts.clear()

        if waste_rows:
            rows = [(fn, round(idle, 5)) for fn, idle in waste_rows]
            _append_rows(WASTE_OUT, ["func name", "waste minutes"], rows, float_prec=5)
            waste_rows.clear()

        if life_sums:
            rows = [(fn, round(s, 5)) for fn, s in life_sums.items()]
            _append_rows(LIFESPAN_OUT, ["func name", "sum lifeSpan"], rows, float_prec=5)
            life_sums.clear()

        # final GC with the last timestamp of this chunk
        try:
            max_t = float(chunk["arrival time"].max())
            sim._maybe_gc(max_t)
        except Exception:
            pass

    # After all chunks, flush the last open minute
    if next_minute_to_emit is not None:
        ava, busy, exist = count_containers(sim, float(next_minute_to_emit))
        per_minute_stats.append([next_minute_to_emit, ava, busy, exist])

    # finalize minute stats
    _ = minute_stats_handler.finalize_minute_stats(
        per_minute_stats, output_dir=output_dir, method_name=method_name
    )
    return pd.read_csv(output_file)

# ===================== MAIN =====================
if __name__ == "__main__":
    run_warmflex_stream()
