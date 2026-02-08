"""WebSocket service for real-time training progress updates."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)

# Active WebSocket connections per training run
_connections: dict[int, list[WebSocket]] = {}


async def connect(run_id: int, websocket: WebSocket) -> None:
    """Register a WebSocket connection for a training run."""
    await websocket.accept()
    if run_id not in _connections:
        _connections[run_id] = []
    _connections[run_id].append(websocket)
    logger.info("WebSocket connected for run %d (total: %d)", run_id, len(_connections[run_id]))


async def disconnect(run_id: int, websocket: WebSocket) -> None:
    """Remove a WebSocket connection."""
    if run_id in _connections:
        _connections[run_id] = [ws for ws in _connections[run_id] if ws != websocket]
        if not _connections[run_id]:
            del _connections[run_id]


async def broadcast_progress(run_id: int, data: dict[str, Any]) -> None:
    """Send progress update to all connected clients for a training run."""
    if run_id not in _connections:
        return

    message = json.dumps(data)
    disconnected = []

    for ws in _connections[run_id]:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.append(ws)

    # Clean up disconnected clients
    for ws in disconnected:
        await disconnect(run_id, ws)


async def progress_monitor(run_id: int) -> None:
    """Periodically broadcast training progress for a run.

    Polls job_runner progress and sends updates via WebSocket.
    """
    from backend.services.job_runner import get_job_progress

    while True:
        progress = get_job_progress(run_id)
        await broadcast_progress(run_id, progress)

        if progress.get("status") in ("completed", "failed", "cancelled"):
            break

        await asyncio.sleep(1)
