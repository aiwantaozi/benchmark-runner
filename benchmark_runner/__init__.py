__version__ = '0.0.0'
__git_commit__ = 'HEAD'

from . import custom_response_handler  # noqa: F401 # Register custom response handler
from . import output_dual_json  # noqa: F401 # Register output format
from . import openai_http_error_detail_backend  # noqa: F401 # Register custom backend

# guidellm 0.7.1: registering a new BackendArgs / BenchmarkOutputArgs subclass only
# rebuilds that base registry's own schema (register_decorator -> reload_schema),
# NOT the parent models that embed them as fields. Reload BenchmarkArgs and
# BenchmarkScenario so their polymorphic unions include our custom
# ``openai_http_error_detail`` backend args and ``dual_json`` output args.
from guidellm.benchmark import BenchmarkArgs, BenchmarkScenario  # noqa: E402

# ``parents=True`` also rebuilds models that embed these (e.g.
# GenerativeBenchmarksReport.config) so their tagged-union serializers recognize
# our custom backend/output subclasses (avoids PydanticSerializationUnexpectedValue
# warnings when the report config is dumped).
BenchmarkArgs.reload_schema(parents=True)
BenchmarkScenario.reload_schema(parents=True)
