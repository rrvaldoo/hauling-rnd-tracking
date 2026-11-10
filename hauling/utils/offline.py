"""Disable optional integrations and reject outbound Python socket operations."""

from contextlib import contextmanager
import os
from pathlib import Path
import socket
import sys
from unittest.mock import patch


class OfflineNetworkError(RuntimeError):
    pass


@contextmanager
def offline_runtime(output_dir: Path):
    settings = {
        "YOLO_OFFLINE": "true",
        "YOLO_AUTOINSTALL": "false",
        "YOLO_CONFIG_DIR": str((output_dir / ".ultralytics").resolve()),
        "YOLO_VERBOSE": "false",
        "WANDB_DISABLED": "true",
        "COMET_MODE": "DISABLED",
        "DO_NOT_TRACK": "1",
    }

    def deny(*args, **kwargs):
        raise OfflineNetworkError("Network access is disabled for offline inference")

    # Restore only this entry. Restoring the entire sys.modules mapping would
    # discard modules imported during inference and break native-module reloads.
    missing = object()
    previous_telemetry = sys.modules.get("openvino_telemetry", missing)
    sys.modules["openvino_telemetry"] = None
    try:
        with (
            patch.dict(os.environ, settings),
            patch.object(socket.socket, "connect", deny),
            patch.object(socket.socket, "connect_ex", deny),
            patch.object(socket.socket, "sendto", deny),
            patch.object(socket, "create_connection", deny),
            patch.object(socket, "getaddrinfo", deny),
        ):
            yield
    finally:
        if previous_telemetry is missing:
            sys.modules.pop("openvino_telemetry", None)
        else:
            sys.modules["openvino_telemetry"] = previous_telemetry
