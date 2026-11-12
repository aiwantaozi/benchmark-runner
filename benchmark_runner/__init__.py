__version__ = '0.0.0'
__git_commit__ = 'HEAD'

from . import custom_response_handler  # noqa: F401 # Register custom response handler
from . import output_dual_json  # noqa: F401 # Register output format
from . import openai_http_error_detail_backend  # noqa: F401 # Register custom backend

# guidellm 0.7.x: registering a new BackendArgs / BenchmarkOutputArgs subclass only
# rebuilds that base registry's own schema (register_decorator -> reload_schema),
# NOT the parent models that embed them as fields. Reload BenchmarkArgs and
# BenchmarkScenario so their polymorphic unions include our custom
# ``openai_http_error_detail`` backend args and ``dual_json`` output args.
from guidellm.benchmark import BenchmarkArgs, BenchmarkScenario  # noqa: E402

# Rebuild only the models embedding our extensions. In 0.8.1, recursively
# rebuilding every parent also traverses recursive trace schemas and can fail
# with an unresolved Pydantic reference during import.
from guidellm.benchmark.schemas import GenerativeBenchmarksReport  # noqa: E402

BenchmarkArgs.reload_schema()
BenchmarkScenario.reload_schema()
GenerativeBenchmarksReport.model_rebuild(force=True)
