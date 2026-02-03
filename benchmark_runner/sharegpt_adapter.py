from pathlib import Path
from benchmark_runner.sharegpt_to_guidellm import convert_sharegpt_to_guidellm


def _count_lines(path: Path, chunk_size: int = 1024 * 1024) -> int:
    total = 0
    last_byte = None
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            total += chunk.count(b"\n")
            last_byte = chunk[-1]
    if last_byte is None:
        return 0
    if last_byte != 10:  # b"\n"
        total += 1
    return total


class ShareGPTAdapter:
    def supports(self, source: str) -> bool:
        return (
            source.endswith(".json") or source.endswith(".jsonl")
        ) and "sharegpt" in source.lower()

    def prepare(
        self,
        source: str,
        *,
        tokenizer: str,
        max_items: int | None,
    ) -> list[str]:
        source_path = Path(source)
        output = source_path.parent / f"converted_{source_path.stem}.jsonl"

        if not output.exists():
            if max_items is not None:
                max_items = int(max_items * 1.2)  # Convert more
            convert_sharegpt_to_guidellm(
                input_file=Path(source),
                output_file=output,
                tokenizer_name=tokenizer,
                max_items=max_items,
            )
        return [str(output)]


dataset_adapters = [
    ShareGPTAdapter(),
]


def _select_adapter(source: str) -> ShareGPTAdapter | None:
    for adapter in dataset_adapters:
        if adapter.supports(source):
            return adapter
    return None


def prepare_datasets(
    data: list[str],
    *,
    tokenizer: str,
    max_items: int | None,
) -> tuple[list[str], int | None]:
    prepared = []
    used_max_items = max_items

    for source in data:
        adapter = _select_adapter(source)
        if adapter is None:
            prepared.append(source)
            continue

        outputs = adapter.prepare(
            source,
            tokenizer=tokenizer,
            max_items=used_max_items,
        )
        prepared.extend(outputs)
        if used_max_items is None and outputs:
            used_max_items = 0
            for output in outputs:
                actual_items = _count_lines(Path(output))
                if actual_items > used_max_items:
                    used_max_items = actual_items

    return prepared, used_max_items
