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

"""HTTP cross-encoder reranking for coarse Memory search candidates."""

from __future__ import annotations

import math
from typing import Any

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage

DEFAULT_HTTP_RERANK_POLICY_ID = "http-cross-encoder-v1"


class HTTPMemoryReranker:
    """Score a complete candidate pool through a batched HTTP service."""

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = DEFAULT_HTTP_RERANK_POLICY_ID,
    ) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("endpoint must be a non-empty string")
        if not isinstance(policy_id, str) or not policy_id.strip():
            raise ValueError("policy_id must be a non-empty string")
        self.endpoint = endpoint
        self.client = client
        self.batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Score every candidate once and select a global stable top-limit."""

        if not candidates:
            raise ValueError("rerank requires at least one candidate")
        if limit < 1 or limit > len(candidates):
            raise ValueError("rerank limit must be between one and the candidate count")

        batches = [
            tuple(enumerate(candidates[start : start + self.batch_size], start=start + 1))
            for start in range(0, len(candidates), self.batch_size)
        ]
        scores: dict[int, float] = {}
        for batch in batches:
            payload = {
                "query": query,
                "candidates": [
                    {"rank": rank, "text": candidate.text} for rank, candidate in batch
                ],
            }
            response = await self.client.post(self.endpoint, json=payload)
            response.raise_for_status()
            try:
                body = response.json()
            except ValueError as error:
                raise ValueError("rerank response is not valid JSON") from error
            if not isinstance(body, dict) or set(body) != {"scores"}:
                raise ValueError("rerank response must contain exactly the scores field")
            raw_scores = body["scores"]
            if not isinstance(raw_scores, list):
                raise ValueError("rerank response scores must be a list")
            expected = {rank for rank, _ in batch}
            seen: set[int] = set()
            for item in raw_scores:
                if not isinstance(item, dict) or set(item) != {"rank", "score"}:
                    raise ValueError("rerank score entries must contain exactly rank and score")
                rank = item["rank"]
                score = item["score"]
                if not isinstance(rank, int) or isinstance(rank, bool):
                    raise ValueError("rerank score rank must be an integer")
                if not isinstance(score, (int, float)) or isinstance(score, bool):
                    raise ValueError("rerank score must be a finite number")
                if not math.isfinite(float(score)):
                    raise ValueError("rerank score must be a finite number")
                if rank not in expected:
                    raise ValueError("rerank score rank is outside the requested batch")
                if rank in seen:
                    raise ValueError("rerank response contains a duplicate rank")
                seen.add(rank)
                scores[rank] = float(score)
            if seen != expected:
                raise ValueError("rerank response must cover every requested rank exactly once")

        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        selected = tuple(rank for rank, _ in ordered[:limit])
        return MemoryRerankDecision(
            selected_ranks=selected,
            usage=InferenceUsage(requests=len(batches)),
        )


__all__ = ["DEFAULT_HTTP_RERANK_POLICY_ID", "HTTPMemoryReranker"]
