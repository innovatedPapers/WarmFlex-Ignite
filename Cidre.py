import os
import zipfile
import csv
import pandas as pd
from collections import deque, defaultdict
import minute_stats_handler
import numpy as np

# -------------------- utils --------------------
def median(lst):
    if not lst:
        return None
    s = sorted(lst)
    n = len(s)
    m = n // 2
    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2.0

# -------------------- data structures --------------------
class Container:

    def __init__(self, function_name):
        self.function_name = function_name
        self.busy_until = 0.0
        self.last_used = 0.0
        self.destroyed_at = None
        self.create_until = None  # when cold start finishes
        self.state = 'None'       # cached convenience only

        # generation tracking for waste/deletion
        self.gen_id = 0
        self.gen_used_after_creation = False  # True once a warm reuse happens after a cold start

class FuncStats:
  
    def __init__(self, window_minutes=15.0, med_cache_horizon_min=0.1):
        self.window = float(window_minutes)
        self.inv_times = deque()      # (t,1)
        self.exec_times = deque()     # (t,exe)
        self.idle_samples = deque()   # (t,gap)
        self.queue_delays = deque()   # (t,delay)
        self.latest_finish = None
        self.last_seen_time = None    # last arrival for this function

        self._med_cache_val = None
        self._med_cache_t = -1e30
        self._med_cache_horizon = float(med_cache_horizon_min)

    def _trim(self, now):
        cutoff = now - self.window
        for dq in (self.inv_times, self.exec_times, self.idle_samples, self.queue_delays):
            while dq and dq[0][0] < cutoff:
                dq.popleft()

    def observe_invocation(self, t, exe, qd=None):
        self._trim(t)
        self.last_seen_time = t
        self.inv_times.append((t, 1.0))
        self.exec_times.append((t, float(exe)))
        if qd is not None:
            self.queue_delays.append((t, float(qd)))

    def observe_idle_gap(self, now):
        if self.latest_finish is not None and now > self.latest_finish:
            self.idle_samples.append((now, now - self.latest_finish))

    def freq_per_min(self, now):
        self._trim(now)
        return len(self.inv_times) / self.window if self.window > 0 else 0.0

    def med_exec(self, now):
        if (now - self._med_cache_t) < self._med_cache_horizon and self._med_cache_val is not None:
            return self._med_cache_val
        self._trim(now)
        vals = [v for _, v in self.exec_times]
        m = None
        if vals:
            s = sorted(vals)
            n = len(s)
            mid = n // 2
            m = s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0
        self._med_cache_val = m
        self._med_cache_t = now
        return m

# -------------------- simulator --------------------
class ServerlessSimulator:
    def __init__(
        self,
        cold_start_time=1/600,          # ~100ms in minutes
        default_keepalive=5,        # 5 minutes
        stats_window=1,              # sliding window minutes
        med_cache_horizon_min=0.1,      # recompute median at most every 0.1 min
        gc_horizon=60.0                 # drop destroyed containers/stats after 60 min of inactivity
    ):
        self.cold_start_time = float(cold_start_time)
        self.default_keepalive = float(default_keepalive)
        self.stats_window = float(stats_window)
        self.gc_horizon = float(gc_horizon)

        self.container = {}  # function_name -> Container (exactly one per function)
        self.stats = defaultdict(lambda: FuncStats(window_minutes=self.stats_window,
                                                  med_cache_horizon_min=med_cache_horizon_min))

        # streaming callbacks (set per chunk by the driver)
        self._on_deletion = None              # fn -> None
        self._on_waste = None                 # (fn, idle_minutes) -> None
        self._on_lifespan = None              # (fn, lifespan_minutes) -> None

    # ----- rounding -----
    @staticmethod
    def _r5(x):  # for main output only
        return round(float(x), 5)

    # ----- authoritative state query -----
    def get_container_state(self, function_name, t):
        c = self.container.get(function_name)
        if c is None:
            return 'None'
        if c.create_until is not None and t < c.create_until:
            return 'creating'
        if t < c.busy_until:
            return 'busy'
        if c.destroyed_at is not None and t >= c.destroyed_at:
            return 'destroyed'
        return 'available'

    
    # ----- pop & clear per-chunk events -----
    def pop_metric_events(self):
        life = self._lifespan_events
        dels = self._deletion_events
        waste = self._waste_events
        self._lifespan_events = []
        self._deletion_events = []
        self._waste_events = []
        return life, dels, waste
    
    
    # ----- dynamic keep-alive -----
    def _dynamic_keepalive(self, fn, now):
        st = self.stats[fn]
        freq = st.freq_per_min(now)
        Te = st.med_exec(now)
        if Te is None:
            return self.default_keepalive
        if freq > 0:
            return min(1.0 / freq, 4.0 * Te)
        return max(self.default_keepalive, Te)

    # ----- GC of long-dead containers *and* stats -----
    def _gc(self, now):
        to_del = []
        for fn, c in self.container.items():
            # Container can be dropped if destroyed long ago and no recent stats
            destroyed_long_ago = (c.destroyed_at is not None) and (now - c.destroyed_at > self.gc_horizon)
            st = self.stats.get(fn)
            no_recent_stats = (st is None) or (st.last_seen_time is None) or (now - st.last_seen_time > self.gc_horizon)
            if destroyed_long_ago and no_recent_stats:
                to_del.append(fn)
        for fn in to_del:
            if fn in self.container:
                del self.container[fn]
            if fn in self.stats:
                del self.stats[fn]

    # ----- set streaming callbacks (per chunk) -----
    def set_callbacks(self, on_deletion=None, on_waste=None, on_lifespan=None):
        self._on_deletion = on_deletion
        self._on_waste = on_waste
        self._on_lifespan = on_lifespan

    # ----- main request -----
    def process_request(self, request_id, function_name, arrival_time, exe_time):
        fn = function_name
        now = float(arrival_time)
        Tp = self.cold_start_time
        st = self.stats[fn]

        st.observe_idle_gap(now)

        c = self.container.get(fn)
        cold_start = 0

        # Deletion detection at the time we next see this function
        if c is not None and c.busy_until <= now and c.destroyed_at is not None and now >= c.destroyed_at:
            # stream deletion event
            if self._on_deletion:
                self._on_deletion(fn)
            # stream waste if never warm-reused in this generation
            if not c.gen_used_after_creation and self._on_waste:
                idle_minutes = c.destroyed_at - c.last_used  # time idle before destruction
                self._on_waste(fn, idle_minutes)
            # continue; a new generation may start below

        if c is None:
            # first-ever: cold start (new generation)
            c = Container(fn)
            self.container[fn] = c
            c.gen_id = 1
            c.gen_used_after_creation = False
            c.create_until = now + Tp
            exe_start = c.create_until
            cold_start = 1
            c.state = 'creating'
        else:
            if c.busy_until > now:
                # busy: wait
                exe_start = c.busy_until
                c.state = 'busy'
            else:
                # idle: if expired, cold start (new generation); else warm start
                if c.destroyed_at is not None and now >= c.destroyed_at:
                    c.gen_id += 1
                    c.gen_used_after_creation = False
                    c.create_until = now + Tp
                    exe_start = c.create_until
                    cold_start = 1
                    c.state = 'creating'
                else:
                    exe_start = now
                    c.state = 'available'
                    c.gen_used_after_creation = True  # mark warm reuse in this generation

        finish_time = exe_start + float(exe_time)
        keepalive = self._dynamic_keepalive(fn, now)
        if keepalive < self.default_keepalive:
            keepalive += np.random.uniform(0, 1)


        # lifecycle update
        c.busy_until = finish_time
        c.last_used = finish_time
        c.destroyed_at = finish_time + keepalive
        if cold_start == 0:
            c.state = 'busy'

        # stats update
        queue_delay = (exe_start - now) if cold_start == 0 else max(0.0, (exe_start - now) - Tp)
        st.observe_invocation(now, float(exe_time), qd=queue_delay)
        st.latest_finish = max(finish_time, st.latest_finish or finish_time)

        # stream lifespan for cold starts (creation/arrival -> finish)
        if cold_start == 1 and self._on_lifespan is not None:
            self._on_lifespan(fn, finish_time - now)

        return {
            'ID': int(request_id),
            'function_name': fn,
            'arrival_time': self._r5(now),
            'exe time (percentile 50)': self._r5(float(exe_time)),
            'exe_start': self._r5(exe_start),
            'finish_time': self._r5(finish_time),
            'cold_start': int(cold_start),
            'warmup_duration': self._r5(keepalive),
        }

# -------------------- snapshots --------------------
def count_containers(simulator: ServerlessSimulator, snapshot_time: float):
    """
    - available: state == 'available'
    - busy: state == 'busy'
    - exist: state not in ['destroyed','None']
    """
    ava = busy = exist = 0
    for fn in simulator.container.keys():
        st = simulator.get_container_state(fn, snapshot_time)
        if st == 'available':
            ava += 1; exist += 1
        elif st == 'busy':
            busy += 1; exist += 1
        elif st == 'creating':
            exist += 1
    return ava, busy, exist

# -------------------- file helpers --------------------
def _append_rows(path, header, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', newline='') as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(header)
        for r in rows:
            out = []
            for x in r:
                if isinstance(x, float):
                    out.append(f"{x:.5f}")
                else:
                    out.append(x)
            w.writerow(out)

# -------------------- driver --------------------
def run_simulation_chunked(
    zip_file_path,
    output_file='output.csv',
    chunk_size=1000,
    gc_every_n_chunks=5,
    method_name="CIDRE",
    output_dir="result"
):
    sim = ServerlessSimulator()

    # ensure dirs
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # clean main invocation-level file
    if os.path.exists(output_file):
        os.remove(output_file)

    # three separate files (as requested)
    deletions_file = os.path.join(output_dir, f"result_container_deletions_{method_name}.csv")
    waste_file      = os.path.join(output_dir, f"result_container_waste_{method_name}.csv")
    lifespan_file   = os.path.join(output_dir, f"result_container_lifeSpan_{method_name}.csv")

    # write headers once (files are append-only afterward)
    _append_rows(deletions_file, ["Function name", "deletion number"], [])
    _append_rows(waste_file,      ["func name", "waste minutes"],      [])
    _append_rows(lifespan_file,   ["func name", "sum lifeSpan"],       [])

    first_chunk = True
    per_minute_stats = []  # used by minute_stats_handler for container status

    with zipfile.ZipFile(zip_file_path, 'r') as z:
        with z.open(z.namelist()[0]) as f:
            for chunk_num, chunk in enumerate(pd.read_csv(f, chunksize=chunk_size), 1):
                # column checks / rename / rounding
                need = {'arrival time', 'exe time (percentile 50)', 'function name'}
                if need.difference(chunk.columns):
                    raise ValueError("Input CSV must contain columns: 'arrival time', 'exe time (percentile 50)', 'function name'.")

                chunk = chunk.rename(columns={
                    'arrival time': 'arrival_time',
                    'exe time (percentile 50)': 'exe_time_p50',
                    'function name': 'function_name'
                })
                chunk['arrival_time'] = chunk['arrival_time'].astype(float).round(5)
                chunk['exe_time_p50'] = chunk['exe_time_p50'].astype(float).round(5)
                chunk = chunk.sort_values('arrival_time', kind='mergesort')

                # ---- per-chunk tiny accumulators (cleared every chunk)
                del_counts = defaultdict(int)   # fn -> count
                waste_rows = []                 # list of (fn, idle_minutes)
                life_sums  = defaultdict(float) # fn -> sum(lifespan)

                # set streaming callbacks that fill these tiny accumulators
                sim.set_callbacks(
                    on_deletion=lambda fn: del_counts.__setitem__(fn, del_counts.get(fn, 0) + 1),
                    on_waste=lambda fn, idle: waste_rows.append((fn, float(idle))),
                    on_lifespan=lambda fn, life: life_sums.__setitem__(fn, life_sums.get(fn, 0.0) + float(life))
                )

                # process this chunk (fast path: itertuples)
                # We stream metrics via callbacks; no big lists.
                results = []  # still build per-chunk main results (you can switch to row-wise CSV if needed)
                for row in chunk.itertuples(index=False):
                    res = sim.process_request(
                        int(row.ID),
                        row.function_name,
                        float(row.arrival_time),
                        float(row.exe_time_p50),
                    )
                    results.append(res)
                

                # write invocation-level results (5 decimals)
                if results:
                    df = pd.DataFrame(results)
                    if first_chunk:
                        df.to_csv(output_file, index=False, float_format='%.5f')
                        first_chunk = False
                    else:
                        df.to_csv(output_file, mode='a', header=False, index=False, float_format='%.5f')
                # free per-chunk list ASAP
                del results

                # container-status snapshot via your handler (unchanged)
                min_arrival_time = float(chunk['arrival_time'].min())
                ava, busy, exist = count_containers(sim, min_arrival_time)
                per_minute_stats.append([int(min_arrival_time), ava, busy, exist])
                per_minute_stats = minute_stats_handler.check_minute_change(
                    min_arrival_time, per_minute_stats,
                    output_dir=output_dir, method_name=method_name)
                

                # ----- write THREE files and CLEAR accumulators -----
                if del_counts:
                    _append_rows(deletions_file, ["Function name", "deletion number"],
                                 [(fn, cnt) for fn, cnt in del_counts.items()])
                del_counts.clear()

                if waste_rows:
                    _append_rows(waste_file, ["func name", "waste minutes"], waste_rows)
                del waste_rows[:]  # clear list in-place

                if life_sums:
                    _append_rows(lifespan_file, ["func name", "sum lifeSpan"],
                                 [(fn, total_life) for fn, total_life in life_sums.items()])
                life_sums.clear()

                # periodic GC to keep dictionaries flat (drops old containers and stats)
                if (chunk_num % gc_every_n_chunks) == 0:
                    max_arrival_time = float(chunk['arrival_time'].max())
                    sim._gc(max_arrival_time)

    # flush any remaining minute rows for container-status file
    _ = minute_stats_handler.finalize_minute_stats(
        per_minute_stats, output_dir=output_dir, method_name=method_name
    )

    return pd.read_csv(output_file)

# -------------------- run --------------------
if __name__ == "__main__":
    try:
        results = run_simulation_chunked(
            '/mydata/paper_warmFlex/complete_trace.zip',
            'result/CIDRE_output.csv',
            chunk_size=1000
        )
    except Exception as e:
        with open("error_log_CIDRE.txt", "a") as error_file:
            error_file.write(str(e) + "\n")


