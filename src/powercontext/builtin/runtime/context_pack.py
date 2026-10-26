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

"""Deterministic Context Pack and prefix-delta wire protocol."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from dataclasses import dataclass
from typing import Iterable

CONTEXT_PACK_SCHEMA = "powercontext.context-pack.v1"
CONTEXT_PACK_DELTA_SCHEMA = "powercontext.context-pack-delta.v1"


class ContextPackError(ValueError):
    """Protocol, chunking, or integrity validation failure."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _encode_items(items: Iterable[bytes]) -> list[str]:
    encoded: list[str] = []
    for item in items:
        if not isinstance(item, bytes):
            raise ContextPackError("context pack items must be bytes")
        encoded.append(base64.b64encode(item).decode("ascii"))
    return encoded


def _decode_items(values: object) -> tuple[bytes, ...]:
    if not isinstance(values, list):
        raise ContextPackError("context pack items must be a list")
    decoded: list[bytes] = []
    for value in values:
        if not isinstance(value, str):
            raise ContextPackError("context pack items must be base64 strings")
        try:
            decoded.append(base64.b64decode(value, validate=True))
        except (binascii.Error, ValueError) as error:
            raise ContextPackError("context pack items must be valid base64") from error
    return tuple(decoded)


def _pack_digest(items: tuple[bytes, ...]) -> str:
    unsigned = {"schema": CONTEXT_PACK_SCHEMA, "items": _encode_items(items)}
    return "sha256:" + hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def _pack_payload(items: tuple[bytes, ...], digest: str, byte_count: int) -> dict[str, object]:
    return {
        "schema": CONTEXT_PACK_SCHEMA,
        "items": _encode_items(items),
        "digest": digest,
        "byte_count": byte_count,
    }


@dataclass(frozen=True, slots=True)
class ContextPack:
    """One complete context payload split into deterministic UTF-8 chunks."""

    items: tuple[bytes, ...]
    digest: str
    byte_count: int

    @classmethod
    def build(cls, items: Iterable[bytes], /) -> ContextPack:
        item_tuple = tuple(items)
        digest = _pack_digest(item_tuple)
        byte_count = 0
        for _ in range(16):
            data = _canonical_json(_pack_payload(item_tuple, digest, byte_count))
            if len(data) == byte_count:
                return cls(items=item_tuple, digest=digest, byte_count=byte_count)
            byte_count = len(data)
        raise ContextPackError("context pack byte_count did not converge")

    @classmethod
    def from_content(cls, content: str | bytes, *, chunk_bytes: int = 512) -> ContextPack:
        if type(chunk_bytes) is not int or chunk_bytes <= 0:
            raise ContextPackError("chunk_bytes must be a positive integer")
        if isinstance(content, bytes):
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ContextPackError("context content must be valid UTF-8") from error
        elif isinstance(content, str):
            text = content
        else:
            raise ContextPackError("context content must be text or bytes")

        chunks: list[bytes] = []
        current: list[str] = []
        current_bytes = 0
        for character in text:
            encoded = character.encode("utf-8")
            size = len(encoded)
            if size > chunk_bytes:
                raise ContextPackError("a single character exceeds the chunk byte limit")
            if current and current_bytes + size > chunk_bytes:
                chunks.append("".join(current).encode("utf-8"))
                current = []
                current_bytes = 0
            current.append(character)
            current_bytes += size
        if current:
            chunks.append("".join(current).encode("utf-8"))
        return cls.build(chunks)

    def to_bytes(self) -> bytes:
        return _canonical_json(_pack_payload(self.items, self.digest, self.byte_count))

    @classmethod
    def from_bytes(cls, data: bytes, /) -> ContextPack:
        if not isinstance(data, bytes):
            raise ContextPackError("context pack wire value must be bytes")
        try:
            payload = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ContextPackError("context pack must be valid UTF-8 JSON") from error
        if not isinstance(payload, dict) or set(payload) != {"schema", "items", "digest", "byte_count"}:
            raise ContextPackError("context pack must contain exactly schema, items, digest, byte_count")
        if payload["schema"] != CONTEXT_PACK_SCHEMA:
            raise ContextPackError("unsupported context pack schema")
        items = _decode_items(payload["items"])
        digest = payload["digest"]
        byte_count = payload["byte_count"]
        if not isinstance(digest, str) or digest != _pack_digest(items):
            raise ContextPackError("context pack digest does not match its items")
        if type(byte_count) is not int or byte_count != len(data):
            raise ContextPackError("context pack byte_count does not match its wire bytes")
        return cls(items=items, digest=digest, byte_count=byte_count)


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    """A prefix of the base pack followed by the target suffix."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack, /) -> ContextPackDelta:
        if not isinstance(base, ContextPack) or not isinstance(target, ContextPack):
            raise ContextPackError("delta endpoints must be ContextPack values")
        prefix = 0
        limit = min(len(base.items), len(target.items))
        while prefix < limit and base.items[prefix] == target.items[prefix]:
            prefix += 1
        return cls(
            base_digest=base.digest,
            target_digest=target.digest,
            prefix_items=prefix,
            appended=target.items[prefix:],
        )

    def to_bytes(self) -> bytes:
        return _canonical_json(
            {
                "schema": CONTEXT_PACK_DELTA_SCHEMA,
                "base_digest": self.base_digest,
                "target_digest": self.target_digest,
                "prefix_items": self.prefix_items,
                "appended": _encode_items(self.appended),
            }
        )

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    @classmethod
    def from_bytes(cls, data: bytes, /) -> ContextPackDelta:
        if not isinstance(data, bytes):
            raise ContextPackError("context pack delta wire value must be bytes")
        try:
            payload = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ContextPackError("context pack delta must be valid UTF-8 JSON") from error
        expected = {"schema", "base_digest", "target_digest", "prefix_items", "appended"}
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ContextPackError("context pack delta must contain exactly the protocol fields")
        if payload["schema"] != CONTEXT_PACK_DELTA_SCHEMA:
            raise ContextPackError("unsupported context pack delta schema")
        base_digest = payload["base_digest"]
        target_digest = payload["target_digest"]
        prefix = payload["prefix_items"]
        if not isinstance(base_digest, str) or not isinstance(target_digest, str):
            raise ContextPackError("delta digests must be strings")
        if type(prefix) is not int or prefix < 0:
            raise ContextPackError("prefix_items must be a non-negative integer")
        return cls(
            base_digest=base_digest,
            target_digest=target_digest,
            prefix_items=prefix,
            appended=_decode_items(payload["appended"]),
        )


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta, /) -> ContextPack:
    """Validate a base and delta, then rebuild the exact target pack."""

    if not isinstance(base, ContextPack) or not isinstance(delta, ContextPackDelta):
        raise ContextPackError("delta application requires a ContextPack and ContextPackDelta")
    if base.digest != delta.base_digest:
        raise ContextPackError("delta base digest does not match the supplied base pack")
    if type(delta.prefix_items) is not int or delta.prefix_items < 0:
        raise ContextPackError("delta prefix_items must be a non-negative integer")
    if delta.prefix_items > len(base.items):
        raise ContextPackError("delta prefix_items exceeds the base pack length")
    target_items = base.items[: delta.prefix_items] + delta.appended
    target = ContextPack.build(target_items)
    if target.digest != delta.target_digest:
        raise ContextPackError("delta target digest does not match the rebuilt pack")
    return target


__all__ = [
    "CONTEXT_PACK_DELTA_SCHEMA",
    "CONTEXT_PACK_SCHEMA",
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "apply_context_pack_delta",
]
