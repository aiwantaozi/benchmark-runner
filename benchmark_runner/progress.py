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

    async def on_initialize(self, profile: Profile):
        if self.session is None:
            headers = {}
            if self.progress_auth is not None:
                headers["Authorization"] = f"Bearer {self.progress_auth}"
            self.session = aiohttp.ClientSession(
                headers=headers, timeout=aiohttp.ClientTimeout(total=60)
            )

    async def on_benchmark_start(self, strategy: SchedulingStrategy):
        await self._update_progress(0)

    async def on_benchmark_update(
        self,
        accumulator: GenerativeBenchmarkAccumulator,
        scheduler_state: SchedulerState,
    ):
        progress = (
            (1.0 - scheduler_state.progress.remaining_fraction) * 100
            if scheduler_state.progress.remaining_fraction is not None
            else 0.0
        )

        if progress and progress > 0:
            print(f"Progress: {progress}")
        if scheduler_state and scheduler_state.progress:
            print(
                f"Total requests: {scheduler_state.progress.total_requests}, Created requests: {scheduler_state.created_requests}, processed requests: {scheduler_state.processed_requests}"
            )

        await self._update_progress(progress)

    async def on_benchmark_complete(self, benchmark: GenerativeBenchmark):
        await self._update_progress(100)

    async def on_finalize(self):
        await self.session.close()

    async def _update_progress(self, progress: float):
        if self.session is None:
            return

        now = time.time()
        should_update = (
            now - self._last_update_ts >= 1.0  # 1 seconds elapsed
            or progress >= 100.0
            or progress - self._last_progress >= 2.0
        )
        if not should_update:
            return

        try:
            resp = await self.session.patch(
                f"{self.progress_url}", json={"progress": progress}
            )
            resp.raise_for_status()

            self._last_progress = progress
            self._last_update_ts = now

        except Exception as e:
            raise RuntimeError(f"Failed to update progress to server: {e}")
