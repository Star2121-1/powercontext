# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Explicit Source dependency lifecycle for Memory and Topic Memory eligibility."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, field_validator
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory.models import MemoryCitation
from powercontext.sources import SourceRef

_SOURCE = "source"
_MEMORY = "memory"
_TOPIC = "topic"


class SourceLifecycleError(ValueError):
    """A lifecycle graph, event, or eligibility contract was violated."""


class SourceLifecycleEvent(BaseModel):
    """One immutable Source availability event."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: StrictStr
    revision: StrictInt
    sequence: StrictInt
    available: StrictBool
    reason: StrictStr

    @field_validator("node_id", "reason")
    @classmethod
    def non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("revision", "sequence")
    @classmethod
    def positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be positive")
        return value


class _LifecycleRow(BaseModel):
    scope_id: str
    node_id: str
    kind: Literal["source", "memory", "topic"]
    parents: tuple[str, ...]
    policy: Literal["all", "any"]
    revision: int | None
    sequence: int
    available: bool
    reason: str | None
    binding: dict[str, Any]


class ScopedSourceLifecycle:
    """One scope's explicit dependency DAG and source lifecycle events."""

    def __init__(self, runtime: Any, scope_id: str) -> None:
        self._runtime = runtime
        self.scope_id = scope_id
        self._database = runtime._provider.database
        self._ready = False

    async def _ensure(self) -> None:
        if self._ready:
            return
        async with self._database.transaction() as connection:
            await connection.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS pc_source_lifecycle_nodes (
                        scope_id VARCHAR(512) NOT NULL,
                        node_id VARCHAR(512) NOT NULL,
                        kind VARCHAR(16) NOT NULL,
                        parents TEXT NOT NULL,
                        policy VARCHAR(8) NOT NULL,
                        revision INTEGER,
                        sequence INTEGER NOT NULL,
                        available INTEGER NOT NULL,
                        reason TEXT,
                        binding TEXT NOT NULL,
                        PRIMARY KEY (scope_id, node_id)
                    )
                    """
                )
            )
        self._ready = True

    async def _load_all(self, connection: AsyncConnection) -> dict[str, _LifecycleRow]:
        rows = (
            await connection.execute(
                text("SELECT * FROM pc_source_lifecycle_nodes WHERE scope_id = :scope_id"),
                {"scope_id": self.scope_id},
            )
        ).mappings()
        return {str(row["node_id"]): self._decode(row) for row in rows}

    async def _load_one(self, connection: AsyncConnection, node_id: str) -> _LifecycleRow | None:
        row = (
            await connection.execute(
                text(
                    "SELECT * FROM pc_source_lifecycle_nodes WHERE scope_id = :scope_id AND node_id = :node_id"
                ),
                {"scope_id": self.scope_id, "node_id": node_id},
            )
        ).mappings().one_or_none()
        return None if row is None else self._decode(row)

    @staticmethod
    def _decode(row: Any) -> _LifecycleRow:
        return _LifecycleRow(
            scope_id=str(row["scope_id"]),
            node_id=str(row["node_id"]),
            kind=str(row["kind"]),
            parents=tuple(json.loads(row["parents"])),
            policy=str(row["policy"]),
            revision=None if row["revision"] is None else int(row["revision"]),
            sequence=int(row["sequence"]),
            available=bool(row["available"]),
            reason=None if row["reason"] is None else str(row["reason"]),
            binding=json.loads(row["binding"]),
        )

    async def _write(self, connection: AsyncConnection, row: _LifecycleRow) -> None:
        values = {
            "scope_id": row.scope_id,
            "node_id": row.node_id,
            "kind": row.kind,
            "parents": json.dumps(row.parents),
            "policy": row.policy,
            "revision": row.revision,
            "sequence": row.sequence,
            "available": 1 if row.available else 0,
            "reason": row.reason,
            "binding": json.dumps(row.binding, sort_keys=True),
        }
        existing = await self._load_one(connection, row.node_id)
        if existing is None:
            await connection.execute(
                text(
                    """
                    INSERT INTO pc_source_lifecycle_nodes
                    (scope_id, node_id, kind, parents, policy, revision, sequence, available, reason, binding)
                    VALUES (:scope_id, :node_id, :kind, :parents, :policy, :revision, :sequence, :available, :reason, :binding)
                    """
                ),
                values,
            )
        else:
            await connection.execute(
                text(
                    """
                    UPDATE pc_source_lifecycle_nodes
                    SET kind = :kind, parents = :parents, policy = :policy, revision = :revision,
                        sequence = :sequence, available = :available, reason = :reason, binding = :binding
                    WHERE scope_id = :scope_id AND node_id = :node_id
                    """
                ),
                values,
            )

    async def register_source(self, node_id: str, source_ref: SourceRef, *, revision: int) -> None:
        self._validate_node(node_id)
        if type(revision) is not int or revision < 1:
            raise SourceLifecycleError("source revision must be a positive integer")
        await self._ensure()
        row = _LifecycleRow(
            scope_id=self.scope_id,
            node_id=node_id,
            kind=_SOURCE,
            parents=(),
            policy="all",
            revision=revision,
            sequence=0,
            available=True,
            reason=None,
            binding={"source_ref": source_ref.model_dump(mode="json")},
        )
        async with self._database.transaction() as connection:
            existing = await self._load_one(connection, node_id)
            if existing is not None:
                if existing.kind == _SOURCE and existing.revision == revision and existing.binding == row.binding:
                    return
                raise SourceLifecycleError("source node registration conflicts with an existing node")
            await self._write(connection, row)

    async def register_memory(
        self,
        node_id: str,
        citation: MemoryCitation,
        *,
        parents: tuple[str, ...],
        policy: str,
    ) -> None:
        self._validate_node(node_id)
        self._validate_parents(node_id, parents, policy)
        await self._ensure()
        binding = {"citation": citation.model_dump(mode="json")}
        async with self._database.transaction() as connection:
            nodes = await self._load_all(connection)
            self._require_parents(node_id, parents, nodes)
            existing = nodes.get(node_id)
            if existing is not None:
                if (
                    existing.kind == _MEMORY
                    and existing.parents == parents
                    and existing.policy == policy
                    and existing.binding == binding
                ):
                    return
                raise SourceLifecycleError("memory node registration conflicts with an existing node")
            available = self._availability(node_id, parents, policy, nodes, {})
            row = _LifecycleRow(
                scope_id=self.scope_id,
                node_id=node_id,
                kind=_MEMORY,
                parents=parents,
                policy=policy,
                revision=None,
                sequence=0,
                available=available,
                reason=None,
                binding=binding,
            )
            await self._write(connection, row)
        if not available:
            await self._set_memory(citation, False)

    async def register_topic(
        self,
        node_id: str,
        reference: ArtifactRef,
        *,
        parents: tuple[str, ...],
        policy: str,
    ) -> None:
        self._validate_node(node_id)
        self._validate_parents(node_id, parents, policy)
        await self._ensure()
        binding = {"artifact_ref": reference.model_dump(mode="json")}
        async with self._database.transaction() as connection:
            nodes = await self._load_all(connection)
            self._require_parents(node_id, parents, nodes)
            existing = nodes.get(node_id)
            if existing is not None:
                if (
                    existing.kind == _TOPIC
                    and existing.parents == parents
                    and existing.policy == policy
                    and existing.binding == binding
                ):
                    return
                raise SourceLifecycleError("topic node registration conflicts with an existing node")
            row = _LifecycleRow(
                scope_id=self.scope_id,
                node_id=node_id,
                kind=_TOPIC,
                parents=parents,
                policy=policy,
                revision=None,
                sequence=0,
                available=self._availability(node_id, parents, policy, nodes, {}),
                reason=None,
                binding=binding,
            )
            await self._write(connection, row)

    async def apply(self, event: SourceLifecycleEvent) -> list[str]:
        await self._ensure()
        async with self._database.transaction() as connection:
            nodes = await self._load_all(connection)
        source = nodes.get(event.node_id)
        if source is None or source.kind != _SOURCE:
            raise SourceLifecycleError("event targets an unknown Source")
        if source.revision != event.revision:
            raise SourceLifecycleError("event revision does not match the Source")
        if event.sequence == source.sequence:
            if source.available == event.available and source.reason == event.reason:
                return self._active_ids(nodes)
            raise SourceLifecycleError("event sequence was reused with different content")
        if event.sequence < source.sequence:
            raise SourceLifecycleError("event sequence went backwards")
        source = source.model_copy(update={"sequence": event.sequence, "available": event.available, "reason": event.reason})
        nodes[source.node_id] = source
        for node in tuple(nodes.values()):
            if node.kind == _SOURCE:
                continue
            available = self._availability(node.node_id, node.parents, node.policy, nodes, {})
            nodes[node.node_id] = node.model_copy(update={"available": available})
            if node.kind == _MEMORY and available != node.available:
                citation = MemoryCitation.model_validate(node.binding["citation"])
                await self._set_memory(citation, available)
        async with self._database.transaction() as connection:
            for node in nodes.values():
                await self._write(connection, node)
        return self._active_ids(nodes)

    async def resolve(self, node_id: str, *, purpose: str) -> SourceRef | MemoryCitation | ArtifactRef:
        if purpose not in {"generation", "audit"}:
            raise SourceLifecycleError("unknown lifecycle purpose")
        await self._ensure()
        async with self._database.transaction() as connection:
            node = await self._load_one(connection, node_id)
        if node is None:
            raise SourceLifecycleError("unknown lifecycle node")
        if purpose == "generation" and not node.available:
            raise SourceLifecycleError("node is not available for generation")
        if node.kind == _SOURCE:
            return SourceRef.model_validate(node.binding["source_ref"])
        if node.kind == _MEMORY:
            return MemoryCitation.model_validate(node.binding["citation"])
        return ArtifactRef.model_validate(node.binding["artifact_ref"])

    async def filter_topic_hits(self, hits: tuple[Any, ...]) -> tuple[Any, ...]:
        inactive = await self._inactive_topic_ids()
        if not inactive:
            return hits
        return tuple(hit for hit in hits if hit.artifact_ref.artifact_id not in inactive)

    async def filter_topic_result(self, result: Any) -> Any:
        hits = await self.filter_topic_hits(result.hits)
        return result if hits is result.hits else result.model_copy(update={"hits": hits})

    async def filter_topic_items(self, items: tuple[Any, ...]) -> tuple[Any, ...]:
        inactive = await self._inactive_topic_ids()
        if not inactive:
            return items
        return tuple(item for item in items if item.artifact_ref.artifact_id not in inactive)

    async def topic_available(self, artifact_id: str) -> bool:
        return artifact_id not in await self._inactive_topic_ids()

    async def _inactive_topic_ids(self) -> set[str]:
        await self._ensure()
        async with self._database.transaction() as connection:
            rows = (
                await connection.execute(
                    text(
                        "SELECT binding FROM pc_source_lifecycle_nodes "
                        "WHERE scope_id = :scope_id AND kind = 'topic' AND available = 0"
                    ),
                    {"scope_id": self.scope_id},
                )
            ).scalars()
            return {str(json.loads(row)["artifact_ref"]["artifact_id"]) for row in rows}

    async def _set_memory(self, citation: MemoryCitation, active: bool) -> None:
        async with self._runtime._context(self.scope_id) as context, self._runtime._locked(self.scope_id):
            service = context.artifacts.memory
            current = await service.head(citation.memory_ref.artifact_id)
            entries = await service.entries(current)
            entry = next((item for item in entries if item.entry_id == citation.entry_id), None)
            manifest_entry = next(
                (item for item in current.content.manifest.entries if item.entry_id == citation.entry_id),
                None,
            )
            if entry is None or manifest_entry is None:
                raise SourceLifecycleError("registered Memory entry no longer exists")
            if (manifest_entry.state == "active") == active:
                return
            if active:
                await service.reactivate(current, entries=(entry,), reason="source lifecycle")
            else:
                await service.forget(current, entries=(entry,), reason="source lifecycle")

    @staticmethod
    def _active_ids(nodes: dict[str, _LifecycleRow]) -> list[str]:
        return sorted(node_id for node_id, node in nodes.items() if node.available)

    @staticmethod
    def _availability(
        node_id: str,
        parents: tuple[str, ...],
        policy: str,
        nodes: dict[str, _LifecycleRow],
        memo: dict[str, bool],
    ) -> bool:
        values = [_node_available(parent, nodes, memo) for parent in parents]
        return all(values) if policy == "all" else any(values)

    @staticmethod
    def _validate_node(node_id: str) -> None:
        if not isinstance(node_id, str) or not node_id.strip() or node_id != node_id.strip():
            raise SourceLifecycleError("node id must be nonempty and trimmed")

    @staticmethod
    def _validate_parents(node_id: str, parents: tuple[str, ...], policy: str) -> None:
        if policy not in {"all", "any"}:
            raise SourceLifecycleError("policy must be 'all' or 'any'")
        if not parents:
            raise SourceLifecycleError("derived nodes need at least one parent")
        if len(set(parents)) != len(parents):
            raise SourceLifecycleError("parents must be unique")
        if node_id in parents:
            raise SourceLifecycleError("a node cannot depend on itself")

    @staticmethod
    def _require_parents(node_id: str, parents: tuple[str, ...], nodes: dict[str, _LifecycleRow]) -> None:
        for parent in parents:
            if parent not in nodes:
                raise SourceLifecycleError("parent must be registered before its child")
        seen: set[str] = set()
        pending = list(parents)
        while pending:
            parent = pending.pop()
            if parent == node_id:
                raise SourceLifecycleError("dependency cycle")
            if parent in seen:
                continue
            seen.add(parent)
            pending.extend(nodes[parent].parents)


def _node_available(node_id: str, nodes: dict[str, _LifecycleRow], memo: dict[str, bool]) -> bool:
    if node_id in memo:
        return memo[node_id]
    node = nodes[node_id]
    if node.kind == _SOURCE:
        memo[node_id] = node.available
    else:
        values = [_node_available(parent, nodes, memo) for parent in node.parents]
        memo[node_id] = all(values) if node.policy == "all" else any(values)
    return memo[node_id]


class SourceLifecycleApplication:
    """Select a scoped source lifecycle service."""

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    def for_scope(self, scope_id: str, /) -> ScopedSourceLifecycle:
        return ScopedSourceLifecycle(self._runtime, scope_id)


__all__ = [
    "SourceLifecycleApplication",
    "SourceLifecycleError",
    "SourceLifecycleEvent",
    "ScopedSourceLifecycle",
]
