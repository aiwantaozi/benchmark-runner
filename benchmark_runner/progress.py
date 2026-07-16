import time
import aiohttp
from guidellm.benchmark.progress import (
    BenchmarkerProgress,
    GenerativeBenchmarkAccumulator,
    GenerativeBenchmark,
    SchedulerState,
    Profile,
    SchedulingStrategy,
)


class ServerBenchmarkerProgress(
    BenchmarkerProgress[GenerativeBenchmarkAccumulator, GenerativeBenchmark]
):
    def __init__(self, progress_url: str, progress_auth: str = None):
        super().__init__()
        self.progress_url = progress_url
        self.progress_auth = progress_auth
        self.session = None
        self._last_update_ts = 0
        self._last_progress = -1.0
        # Two-level normalization (see progress-design.md):
        #   overall = (run_index + run_local) / run_total
        #   run_local = (benchmarks_done + current_fraction) / benchmarks_total
        # run_index / run_total are set by the runner loop (stages);
        # benchmarks_* track multiple benchmarks within one guidellm run (sweep).
        self.run_index = 0
        self.run_total = 1
        self._bench_done = 0
        self._bench_total = 1

    def _ensure_session(self):
        # Multi-run (stages) reuses this progress object across
        # runs; each run's on_finalize closes the session, so recreate it when
        # None OR already closed — otherwise later stages hit "Session is closed".
        # Auto-tune drives progress directly (not via guidellm's on_initialize),
        # so _update_progress also relies on this to open the session on demand.
        if self.session is None or self.session.closed:
            headers = {}
            if self.progress_auth is not None:
                headers["Authorization"] = f"Bearer {self.progress_auth}"
            self.session = aiohttp.ClientSession(
                headers=headers, timeout=aiohttp.ClientTimeout(total=60)
            )

    async def on_initialize(self, profile: Profile):
        self._ensure_session()
        # New run: reset within-run counters; total = number of benchmarks this
        # run produces (1 for a single-rate stage, sweep_size for a sweep).
        self._bench_done = 0
        try:
            self._bench_total = max(1, len(profile.strategy_types))
        except Exception:
            self._bench_total = 1

    async def on_benchmark_start(self, strategy: SchedulingStrategy):
        await self._emit(0.0)

    async def on_benchmark_update(
        self,
        accumulator: GenerativeBenchmarkAccumulator,
        scheduler_state: SchedulerState,
    ):
        current_fraction = (
            (1.0 - scheduler_state.progress.remaining_fraction)
            if scheduler_state.progress.remaining_fraction is not None
            else 0.0
        )
        await self._emit(current_fraction)

    async def on_benchmark_complete(self, benchmark: GenerativeBenchmark):
        # This benchmark is done: count it; the next one (if any) starts at 0.
        self._bench_done += 1
        await self._emit(0.0)

    async def _emit(self, current_fraction: float):
        run_local = (self._bench_done + current_fraction) / self._bench_total
        overall = (self.run_index + run_local) / self.run_total * 100.0
        overall = min(100.0, max(0.0, overall))
        await self._update_progress(overall)

    async def on_finalize(self):
        if self.session is not None and not self.session.closed:
            await self.session.close()

    async def _update_progress(self, progress: float):
        now = time.time()
        should_update = (
            now - self._last_update_ts >= 1.0  # 1 seconds elapsed
            or progress >= 100.0
            or progress - self._last_progress >= 2.0
        )
        if not should_update:
            return

        # Open the session on demand (auto-tune writes without on_initialize).
        self._ensure_session()

        try:
            resp = await self.session.patch(
                f"{self.progress_url}", json={"progress": progress}
            )
            resp.raise_for_status()

            self._last_progress = progress
            self._last_update_ts = now

        except Exception as e:
            raise RuntimeError(f"Failed to update progress to server: {e}")
