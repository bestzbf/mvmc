"""Read complete native update records amid torchrun's rank-tagged output."""
import json
from pathlib import Path
import re


_STEP = re.compile(r'\{\s*"step"\s*:')


def training_rows(path):
    """Keep every update, tolerating transport text around complete JSON only.

    torchrun may append a rank tag to a valid record or join two records on
    one line. Decode each object, rather than treating its suffix as JSON.
    Broken update JSON still fails closed; callers check continuity/health.
    """
    decoder = json.JSONDecoder()
    rows = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        offset = 0
        while match := _STEP.search(line, offset):
            try:
                row, offset = decoder.raw_decode(line, match.start())
            except json.JSONDecodeError as error:
                raise ValueError(f'malformed native update in {path}:{number}: {error}') from error
            if type(row.get('step')) is not int:
                raise ValueError(f'noninteger native update in {path}:{number}')
            rows.append(row)
    return rows
