# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pi session branch evidence and incremental history collection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel, JsonValue

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.connectors import (
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
    ConnectorSubmissionStatus,
)
from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

PI_EVENT_SOURCE_NAME = "pi-event"
PI_SESSION_CONNECTOR_NAME = "pi-session-history"


class PiSessionInput(BaseModel):
    """One complete Pi v3 JSONL archive and an optional selected leaf."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible event on the selected Pi session branch."""

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class PiEventSource(Source):
    """A captured Pi event with stable session-scoped identity."""

    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


def pi_event_source_id(session_id: str, event_id: str) -> str:
    """Return a delimiter-safe identity for one Pi event."""

    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must be a non-empty string")
    if not isinstance(event_id, str) or not event_id:
        raise ValueError("event_id must be a non-empty string")
    source_id = f"pi:{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"source_id must not exceed {MAX_SOURCE_ID_LENGTH} characters")
    return source_id


def _visible_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str) and text != "":
                parts.append(text)
    return "\n".join(parts)


def _tool_call_text(block: dict[str, Any]) -> str:
    name = block.get("name")
    call_id = block.get("id")
    arguments = block.get("arguments", {})
    compact = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"[tool_call {name if isinstance(name, str) else ''} {call_id if isinstance(call_id, str) else ''}] {compact}"


def _message_body_parts(content: object) -> list[str]:
    if isinstance(content, str):
        return [content] if content != "" else []
    if not isinstance(content, list):
        return []
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str) and text != "":
                parts.append(text)
        elif block_type == "toolCall":
            parts.append(_tool_call_text(block))
    return parts


def _visible_event(session_id: str, record: dict[str, JsonValue]) -> PiEventInput | None:
    event_type = record.get("type")
    event_id = record.get("id")
    if not isinstance(event_type, str) or not isinstance(event_id, str):
        raise ValueError("Pi event must contain string type and id")
    parent_value = record.get("parentId")
    if parent_value is None or parent_value == "":
        parent_id: str | None = None
    elif isinstance(parent_value, str):
        parent_id = parent_value
    else:
        raise ValueError("Pi event parentId must be a string or null")

    role: str
    body: str
    if event_type == "message":
        message = record.get("message")
        if not isinstance(message, dict):
            raise ValueError("Pi message event must contain a message object")
        message_role = message.get("role")
        if not isinstance(message_role, str) or not message_role:
            raise ValueError("Pi message event must contain a role")
        role = message_role
        if message_role == "toolResult":
            tool_name = message.get("toolName")
            tool_call_id = message.get("toolCallId")
            text = _visible_text(message.get("content"))
            body = (
                f"[tool_result {tool_name if isinstance(tool_name, str) else ''} "
                f"{tool_call_id if isinstance(tool_call_id, str) else ''}] {text}"
            )
        elif message_role == "bashExecution":
            command = message.get("command")
            output = message.get("output")
            command_text = command if isinstance(command, str) else ""
            output_text = output if isinstance(output, str) else ""
            body = f"$ {command_text}\n{output_text}"
        else:
            body_parts = _message_body_parts(message.get("content"))
            if not body_parts:
                return None
            body = "\n".join(body_parts)
    elif event_type in {"compaction", "branch_summary"}:
        summary = record.get("summary")
        if summary is None:
            return None
        if not isinstance(summary, str):
            raise ValueError("Pi summary event must contain a string summary")
        role = event_type
        body = summary
    elif event_type == "custom_message":
        body = _visible_text(record.get("content"))
        role = "custom"
        if body == "":
            return None
    else:
        return None

    return PiEventInput(
        session_id=session_id,
        event_id=event_id,
        parent_id=parent_id,
        event_type=event_type,
        role=role,
        content=f"{role}: {body}",
        record=record,
    )


def _parse_content(
    content: bytes, leaf_id: str | None
) -> tuple[str, tuple[PiEventInput, ...], dict[str, dict[str, JsonValue]]]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Pi session must be valid UTF-8") from error

    records: list[dict[str, JsonValue]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("Pi session contains invalid JSON") from error
        if not isinstance(value, dict):
            raise ValueError("Pi session records must be JSON objects")
        records.append(value)

    if not records:
        raise ValueError("Pi session is empty")
    header = records[0]
    if header.get("type") != "session" or header.get("version") != 3:
        raise ValueError("Pi session must start with a version 3 session header")
    session_id = header.get("id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("Pi session header must contain a non-empty id")

    nodes: dict[str, dict[str, JsonValue]] = {}
    parents: dict[str, str | None] = {}
    order: list[str] = []
    for record in records[1:]:
        event_id = record.get("id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("Pi event must contain a non-empty id")
        if event_id in nodes:
            raise ValueError("Pi event ids must be unique")
        parent_value = record.get("parentId")
        if parent_value is None or parent_value == "":
            parent_id: str | None = None
        elif isinstance(parent_value, str):
            parent_id = parent_value
        else:
            raise ValueError("Pi event parentId must be a string or null")
        if parent_id is not None and parent_id not in nodes:
            raise ValueError("Pi event parent must already exist")
        nodes[event_id] = record
        parents[event_id] = parent_id
        order.append(event_id)

    if leaf_id is None:
        leaf = order[-1] if order else None
    else:
        if leaf_id not in nodes:
            raise ValueError("Pi session leaf does not exist")
        leaf = leaf_id

    chain: list[str] = []
    seen: set[str] = set()
    current = leaf
    while current is not None:
        if current in seen:
            raise ValueError("Pi session contains a cycle")
        seen.add(current)
        chain.append(current)
        current = parents[current]
    chain.reverse()

    events = tuple(
        event
        for event_id in chain
        if (event := _visible_event(session_id, nodes[event_id])) is not None
    )
    return session_id, events, nodes


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Parse and validate the complete selected branch of a Pi session."""

    return _parse_content(value.content, value.leaf_id)[1]


def _record_digest(record: dict[str, JsonValue]) -> str:
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class _PiEventAdapter:
    input_class = PiEventInput
    name = PI_EVENT_SOURCE_NAME
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        source_id = pi_event_source_id(value.session_id, value.event_id)
        return PiEventSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class _PiEventTextEvidenceProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_NAME,
            source_id=source.source_id,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


PI_EVENT_SOURCE_DEFINITION: SourceDefinition[PiEventInput, PiEventSource, PiEventInput] = AdapterSourceDefinition(
    _PiEventAdapter(),
    projections=(_PiEventTextEvidenceProjection(),),
)


class PiSessionConnector:
    """Collect the selected Pi branch and commit durable import progress."""

    name = PI_SESSION_CONNECTOR_NAME
    version = "1"
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self.path = Path(path)
        self.leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        try:
            content = self.path.read_bytes()
        except OSError as error:
            raise ValueError("Pi session file could not be read") from error
        session_id, events, nodes = _parse_content(content, self.leaf_id)

        previous = session.checkpoint
        imported: dict[str, str] = {}
        if previous is not None:
            if not isinstance(previous, dict):
                raise ValueError("Pi connector checkpoint must be an object")
            if previous.get("session_id") != session_id:
                raise ValueError("Pi connector checkpoint belongs to another session")
            raw_events = previous.get("events")
            if not isinstance(raw_events, dict):
                raise ValueError("Pi connector checkpoint events must be an object")
            for event_id, digest in raw_events.items():
                if not isinstance(event_id, str) or not isinstance(digest, str):
                    raise ValueError("Pi connector checkpoint entries must be strings")
                imported[event_id] = digest

        for event_id, digest in imported.items():
            record = nodes.get(event_id)
            if record is not None and _record_digest(record) != digest:
                raise ValueError("Pi event changed after it was imported")

        digests = {event.event_id: _record_digest(event.record) for event in events}
        for event in events:
            if event.event_id in imported and imported[event.event_id] != digests[event.event_id]:
                raise ValueError("Pi event changed after it was imported")

        merged = dict(imported)
        for event in events:
            if event.event_id in imported:
                continue
            outcome = await session.submit(event.event_id, PI_EVENT_SOURCE_NAME, event)
            if outcome.status is ConnectorSubmissionStatus.ACCEPTED:
                merged[event.event_id] = digests[event.event_id]

        checkpoint = {"session_id": session_id, "events": dict(sorted(merged.items()))}
        return ConnectorRunCompletion(status=ConnectorRunStatus.COMPLETE, checkpoint=checkpoint)


__all__ = [
    "PI_EVENT_SOURCE_DEFINITION",
    "PI_EVENT_SOURCE_NAME",
    "PI_SESSION_CONNECTOR_NAME",
    "PiEventInput",
    "PiEventSource",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
    "pi_event_source_id",
]
