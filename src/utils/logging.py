"""Structured logging for the LGD prediction pipeline.

Every module in this project uses get_logger(__name__) rather than print
statements. That is not a stylistic preference — it is a practical requirement
for a model that operates in a regulatory context. When a model risk reviewer
asks what happened during a specific training run, or when an audit requires
a reconstruction of what the preprocessing pipeline did to a specific batch
of loans, structured logs with timestamps, module names, and function names
are the difference between an answer and an investigation.

JSON formatting is the default because it allows logs to be ingested directly
by any modern log aggregation system (Datadog, Splunk, CloudWatch, etc.)
without parsing. The fields are chosen to answer the first questions an
analyst will ask when debugging: when did this happen, at what severity,
in which module and function, and what was the message. Exception tracebacks
are included inline when present, which means a single log entry contains
everything needed to diagnose the error without cross-referencing multiple
log lines.

The json_format=False option exists specifically for interactive development
in notebooks and terminals, where the dense JSON output is harder to read
than a simple timestamped line. The underlying behavior is identical — it is
purely a readability switch.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from datetime import datetime, timezone
from typing import Any


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively replace NaN/Inf floats with None so json.dumps produces valid JSON.

    Python's json module serializes float('nan') as bare NaN and float('inf') as
    Infinity — both are invalid per the JSON spec (RFC 8259) and will cause jq and
    any strict JSON parser to fail. This sanitizer walks the object tree and replaces
    those values with null (None) before serialization.
    """
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


class JSONFormatter(logging.Formatter):
    """Format log records as single-line JSON.

    Each log record becomes one JSON object on one line, which is the format
    that log aggregation systems can ingest reliably. Multi-line log output
    creates parsing problems when log lines are streamed through container
    runtimes, which truncate or split lines unpredictably. A single JSON object
    per record avoids that entirely.

    The fields included are:
      - timestamp: ISO 8601 with UTC timezone, not local time. Local time zones
        create ambiguity when logs are compared across machines or across
        daylight saving transitions. UTC eliminates that class of confusion.
      - level: The log level as a string (INFO, WARNING, ERROR, etc.).
      - logger: The logger name, typically the __name__ of the calling module,
        which gives the full dotted module path (e.g., src.data.preprocess).
      - message: The formatted log message.
      - module, function, line: Source code location, which is the first thing
        needed when a log message alone is not enough to diagnose the issue.
      - exception: The formatted traceback, included only when an exception is
        being logged. Inline with the record rather than as a separate entry.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        # Exception information is included inline when present. A reviewer
        # should not have to correlate exception lines with message lines by
        # timestamp — that is fragile and fails when log volumes are high.
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        # The "extra" attribute allows callers to attach domain-specific context
        # to log records without requiring changes to the formatter. For example,
        # a preprocessing step could attach {"loan_count": 15000, "vintage": 2012}
        # to relevant log records for richer traceability.
        if hasattr(record, "extra"):
            log_obj.update(record.extra)
        return json.dumps(_sanitize_for_json(log_obj))


def get_logger(name: str, level: int = logging.INFO, json_format: bool = True) -> logging.Logger:
    """Return a configured logger, creating it if it does not already exist.

    This function is idempotent — calling it twice with the same name returns
    the same logger without adding duplicate handlers. That matters because
    Python's logging module attaches handlers cumulatively if you are not
    careful, which produces duplicate log lines that are both confusing and
    wasteful in a streaming log context.

    The logger.propagate = False setting prevents messages from bubbling up
    to the root logger, which avoids double-logging when the calling code
    also has a root logger configured. In a pipeline that chains multiple
    modules, each of which calls get_logger(), this behavior is important.

    Args:
        name: Logger name, typically __name__ of the calling module. Using
              __name__ produces a logger hierarchy that mirrors the module
              hierarchy, which means you can control log verbosity at any
              level of the package tree (e.g., set src.data to WARNING to
              suppress preprocessing chatter while keeping src.models at INFO).
        level: Minimum severity level to emit. Defaults to INFO. Set to
               logging.DEBUG during development when you need to see
               intermediate computation steps.
        json_format: When True, emit structured JSON (production default).
                     When False, emit a human-readable timestamped line,
                     which is easier to scan in a terminal during development.

    Returns:
        A configured Logger instance ready to use.
    """
    logger = logging.getLogger(name)

    # Guard against re-adding handlers when the same logger is requested
    # multiple times in the same process, which happens routinely in a pipeline
    # where multiple modules import get_logger at the top level.
    if logger.handlers:
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        # The human-readable format trades machine-parseability for readability.
        # The %-8s padding on levelname aligns the fields across log levels,
        # making it easier to scan a terminal for WARNING or ERROR entries.
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )

    logger.addHandler(handler)
    # Prevent propagation to the root logger to avoid duplicate output when
    # a root handler is also configured by the runtime environment.
    logger.propagate = False
    return logger
