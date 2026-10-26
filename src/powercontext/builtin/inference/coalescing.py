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

"""Cross-call Embedding coalescing with a bounded success-only LRU cache."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass

from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel


@dataclass(frozen=True, slots=True)
class _BatchUsage:
    usage: InferenceUsage


class CoalescingEmbeddingModel:
    """Share in-flight text embeddings and cache successful vectors."""

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if type(max_batch_size) is not int or max_batch_size <= 0:
            raise ValueError("max_batch_size must be a positive integer")
        if not isinstance(coalesce_ms, (int, float)) or isinstance(coalesce_ms, bool) or coalesce_ms < 0:
            raise ValueError("coalesce_ms must be a non-negative number")
        if type(cache_size) is not int or cache_size < 0:
            raise ValueError("cache_size must be a non-negative integer")
        self._model = model
        self.profile = model.profile
        self.max_batch_size = max_batch_size
        self.coalesce_ms = float(coalesce_ms)
        self.cache_size = cache_size
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self._inflight: dict[str, asyncio.Future[tuple[EmbeddingVector, _BatchUsage]]] = {}
        self._pending: set[str] = set()
        self._flusher: asyncio.Task[None] | None = None
        self._batches: set[asyncio.Task[None]] = set()

    async def embed(self, texts: tuple[str, ...] | list[str], /) -> EmbeddingResult:
        values = tuple(texts)
        if not values:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))

        unique = tuple(dict.fromkeys(values))
        vectors: dict[str, EmbeddingVector] = {}
        waiters: dict[str, asyncio.Future[tuple[EmbeddingVector, _BatchUsage]]] = {}
        loop = asyncio.get_running_loop()

        for text in unique:
            cached = self._cache.get(text)
            if cached is not None:
                self._cache.move_to_end(text)
                vectors[text] = cached
                continue
            future = self._inflight.get(text)
            if future is None:
                future = loop.create_future()
                self._inflight[text] = future
                self._pending.add(text)
                self._ensure_flusher()
            waiters[text] = future

        batches: dict[int, _BatchUsage] = {}
        for text, future in waiters.items():
            vector, batch = await asyncio.shield(future)
            vectors[text] = vector
            batches[id(batch)] = batch

        usage = _sum_usage(tuple(batches.values()))
        return EmbeddingResult(vectors=tuple(vectors[text] for text in values), usage=usage)

    def _ensure_flusher(self) -> None:
        if self._flusher is None or self._flusher.done():
            self._flusher = asyncio.create_task(self._flush())

    async def _flush(self) -> None:
        try:
            if self.coalesce_ms > 0:
                await asyncio.sleep(self.coalesce_ms / 1000.0)
            else:
                await asyncio.sleep(0)
            while self._pending:
                batch_texts = tuple(list(self._pending)[: self.max_batch_size])
                self._pending.difference_update(batch_texts)
                task = asyncio.create_task(self._run_batch(batch_texts))
                self._batches.add(task)
        finally:
            self._flusher = None

    async def _run_batch(self, texts: tuple[str, ...]) -> None:
        try:
            result = await self._model.embed(texts)
            if len(result.vectors) != len(texts):
                raise ValueError("embedding provider returned the wrong number of vectors")
            batch = _BatchUsage(usage=result.usage)
            for text, vector in zip(texts, result.vectors, strict=True):
                if self.cache_size > 0:
                    self._cache[text] = vector
                    self._cache.move_to_end(text)
                    while len(self._cache) > self.cache_size:
                        self._cache.popitem(last=False)
                future = self._inflight.pop(text, None)
                if future is not None and not future.done():
                    future.set_result((vector, batch))
        except BaseException as error:
            for text in texts:
                future = self._inflight.pop(text, None)
                if future is not None and not future.done():
                    future.set_exception(error)
        finally:
            task = asyncio.current_task()
            if task is not None:
                self._batches.discard(task)


def _sum_usage(batches: tuple[_BatchUsage, ...]) -> InferenceUsage:
    if not batches:
        return InferenceUsage(requests=0)
    requests = sum(batch.usage.requests for batch in batches)
    input_values = [batch.usage.input_tokens for batch in batches]
    output_values = [batch.usage.output_tokens for batch in batches]
    input_tokens = None if all(value is None for value in input_values) else sum(value or 0 for value in input_values)
    output_tokens = (
        None if all(value is None for value in output_values) else sum(value or 0 for value in output_values)
    )
    return InferenceUsage(requests=requests, input_tokens=input_tokens, output_tokens=output_tokens)


__all__ = ["CoalescingEmbeddingModel"]
