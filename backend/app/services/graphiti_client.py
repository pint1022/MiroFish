"""
Local Graphiti/Neo4j adapter.

Graphiti's HTTP API handles ingest/search, while Neo4j HTTP is used for graph
reads that the existing MiroFish UI expects (list nodes, list edges, node detail).
"""

import base64
import json
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..config import Config


class GraphitiClient:
    """Small stdlib HTTP client for local Graphiti and Neo4j."""

    def __init__(
        self,
        graphiti_base_url: Optional[str] = None,
        neo4j_http_url: Optional[str] = None,
        neo4j_database: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
    ):
        self.graphiti_base_url = (graphiti_base_url or Config.GRAPHITI_BASE_URL).rstrip("/")
        self.neo4j_http_url = (neo4j_http_url or Config.NEO4J_HTTP_URL).rstrip("/")
        self.neo4j_database = neo4j_database or Config.NEO4J_DATABASE
        self.neo4j_user = neo4j_user or Config.NEO4J_USER
        self.neo4j_password = neo4j_password or Config.NEO4J_PASSWORD

        if not self.neo4j_password:
            raise ValueError("NEO4J_PASSWORD 未配置")

    def _request_json(
        self,
        url: str,
        payload: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        method: Optional[str] = None,
        timeout: float = 30.0,
    ) -> Any:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request_headers = {"Content-Type": "application/json"}
        if headers:
            request_headers.update(headers)
        req = urllib.request.Request(url, data=data, headers=request_headers, method=method)

        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
            if not raw:
                return None
            return json.loads(raw.decode("utf-8"))

    def _neo4j_auth_header(self) -> Dict[str, str]:
        token = base64.b64encode(f"{self.neo4j_user}:{self.neo4j_password}".encode("utf-8")).decode("ascii")
        return {"Authorization": f"Basic {token}"}

    def cypher(self, statement: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        url = f"{self.neo4j_http_url}/db/{self.neo4j_database}/tx/commit"
        payload = {
            "statements": [
                {
                    "statement": statement,
                    "parameters": parameters or {},
                }
            ]
        }
        response = self._request_json(url, payload, headers=self._neo4j_auth_header())
        errors = response.get("errors") or []
        if errors:
            message = "; ".join(error.get("message", str(error)) for error in errors)
            raise RuntimeError(f"Neo4j query failed: {message}")

        result = (response.get("results") or [{}])[0]
        columns = result.get("columns") or []
        rows = []
        for item in result.get("data") or []:
            row = item.get("row") or []
            rows.append(dict(zip(columns, row)))
        return rows

    def healthcheck(self) -> Dict[str, Any]:
        return self._request_json(f"{self.graphiti_base_url}/healthcheck", method="GET")

    def create_graph(self, name: str) -> str:
        return f"mirofish_{uuid.uuid4().hex[:16]}"

    def add_texts(self, graph_id: str, texts: List[str], source_description: str = "MiroFish") -> List[str]:
        now = datetime.now(timezone.utc).isoformat()
        episode_ids = [str(uuid.uuid4()) for _ in texts]
        messages = [
            {
                "uuid": episode_id,
                "name": f"{source_description} {index + 1}",
                "content": text,
                "role_type": "user",
                "role": source_description,
                "timestamp": now,
                "source_description": source_description,
            }
            for index, (episode_id, text) in enumerate(zip(episode_ids, texts))
        ]
        self._request_json(
            f"{self.graphiti_base_url}/messages",
            {
                "group_id": graph_id,
                "messages": messages,
            },
            timeout=Config.LLM_TIMEOUT_SECONDS,
        )
        return episode_ids

    def wait_for_episodes(self, graph_id: str, expected_count: int, timeout: int = 600):
        if expected_count <= 0:
            return

        effective_timeout = min(timeout, Config.GRAPHITI_INGEST_TIMEOUT_SECONDS)
        deadline = time.time() + effective_timeout
        while time.time() < deadline:
            try:
                episodes = self._request_json(
                    f"{self.graphiti_base_url}/episodes/{graph_id}?last_n={expected_count}",
                    method="GET",
                    timeout=15,
                )
                if isinstance(episodes, list) and len(episodes) >= expected_count:
                    return
            except Exception:
                pass
            time.sleep(3)

        node_count = self.cypher(
            "MATCH (n) WHERE n.group_id = $graph_id RETURN count(n) AS count",
            {"graph_id": graph_id},
        )[0]["count"]
        raise TimeoutError(
            f"Graphiti accepted {expected_count} messages for group {graph_id}, "
            f"but no completed episodes were visible within {effective_timeout}s "
            f"(Neo4j nodes for group: {node_count}). Check the Graphiti API worker logs; "
            "the /messages async ingest worker may not be persisting to Neo4j."
        )

    def search(self, graph_id: str, query: str, limit: int = 10, scope: str = "edges") -> Dict[str, Any]:
        try:
            response = self._request_json(
                f"{self.graphiti_base_url}/search",
                {
                    "group_ids": [graph_id],
                    "query": query,
                    "max_facts": limit,
                },
                timeout=Config.LLM_TIMEOUT_SECONDS,
            )
            facts = self._extract_facts(response)
            return {"facts": facts[:limit], "edges": [], "nodes": [], "query": query}
        except Exception:
            return self.local_search(graph_id, query, limit, scope)

    def _extract_facts(self, response: Any) -> List[str]:
        if isinstance(response, list):
            items = response
        elif isinstance(response, dict):
            items = response.get("facts") or response.get("results") or response.get("edges") or []
        else:
            items = []

        facts = []
        for item in items:
            if isinstance(item, str):
                facts.append(item)
            elif isinstance(item, dict):
                fact = item.get("fact") or item.get("content") or item.get("summary")
                if fact:
                    facts.append(str(fact))
        return facts

    def get_all_nodes(self, graph_id: str) -> List[Dict[str, Any]]:
        rows = self.cypher(
            """
            MATCH (n:Entity)
            WHERE n.group_id = $graph_id
            RETURN coalesce(n.uuid, elementId(n)) AS uuid,
                   coalesce(n.name, '') AS name,
                   labels(n) AS labels,
                   coalesce(n.summary, '') AS summary,
                   properties(n) AS attributes,
                   toString(n.created_at) AS created_at
            ORDER BY name
            """,
            {"graph_id": graph_id},
        )
        return [self._normalize_node(row) for row in rows]

    def get_node(self, node_uuid: str) -> Optional[Dict[str, Any]]:
        rows = self.cypher(
            """
            MATCH (n:Entity)
            WHERE n.uuid = $node_uuid OR elementId(n) = $node_uuid
            RETURN coalesce(n.uuid, elementId(n)) AS uuid,
                   coalesce(n.name, '') AS name,
                   labels(n) AS labels,
                   coalesce(n.summary, '') AS summary,
                   properties(n) AS attributes,
                   toString(n.created_at) AS created_at
            LIMIT 1
            """,
            {"node_uuid": node_uuid},
        )
        return self._normalize_node(rows[0]) if rows else None

    def get_all_edges(self, graph_id: str) -> List[Dict[str, Any]]:
        rows = self.cypher(
            """
            MATCH (source:Entity)-[r:RELATES_TO]->(target:Entity)
            WHERE source.group_id = $graph_id AND target.group_id = $graph_id
            RETURN coalesce(r.uuid, elementId(r)) AS uuid,
                   coalesce(r.name, type(r)) AS name,
                   coalesce(r.fact, '') AS fact,
                   coalesce(source.uuid, elementId(source)) AS source_node_uuid,
                   coalesce(target.uuid, elementId(target)) AS target_node_uuid,
                   coalesce(source.name, '') AS source_node_name,
                   coalesce(target.name, '') AS target_node_name,
                   properties(r) AS attributes,
                   toString(r.created_at) AS created_at,
                   toString(r.valid_at) AS valid_at,
                   toString(r.invalid_at) AS invalid_at,
                   toString(r.expired_at) AS expired_at
            ORDER BY created_at
            """,
            {"graph_id": graph_id},
        )
        return [self._normalize_edge(row) for row in rows]

    def get_node_edges(self, node_uuid: str) -> List[Dict[str, Any]]:
        rows = self.cypher(
            """
            MATCH (source:Entity)-[r:RELATES_TO]-(target:Entity)
            WHERE source.uuid = $node_uuid OR target.uuid = $node_uuid
               OR elementId(source) = $node_uuid OR elementId(target) = $node_uuid
            RETURN coalesce(r.uuid, elementId(r)) AS uuid,
                   coalesce(r.name, type(r)) AS name,
                   coalesce(r.fact, '') AS fact,
                   coalesce(source.uuid, elementId(source)) AS source_node_uuid,
                   coalesce(target.uuid, elementId(target)) AS target_node_uuid,
                   coalesce(source.name, '') AS source_node_name,
                   coalesce(target.name, '') AS target_node_name,
                   properties(r) AS attributes,
                   toString(r.created_at) AS created_at,
                   toString(r.valid_at) AS valid_at,
                   toString(r.invalid_at) AS invalid_at,
                   toString(r.expired_at) AS expired_at
            """,
            {"node_uuid": node_uuid},
        )
        return [self._normalize_edge(row) for row in rows]

    def local_search(self, graph_id: str, query: str, limit: int = 10, scope: str = "edges") -> Dict[str, Any]:
        query_lower = query.lower()
        facts: List[str] = []
        edges: List[Dict[str, Any]] = []
        nodes: List[Dict[str, Any]] = []

        if scope in ("edges", "both"):
            for edge in self.get_all_edges(graph_id):
                text = f"{edge.get('name', '')} {edge.get('fact', '')}".lower()
                if query_lower in text:
                    facts.append(edge.get("fact", ""))
                    edges.append(edge)
                    if len(edges) >= limit:
                        break

        if scope in ("nodes", "both") and len(nodes) < limit:
            for node in self.get_all_nodes(graph_id):
                text = f"{node.get('name', '')} {node.get('summary', '')}".lower()
                if query_lower in text:
                    if node.get("summary"):
                        facts.append(f"[{node.get('name')}]: {node.get('summary')}")
                    nodes.append(node)
                    if len(nodes) >= limit:
                        break

        return {"facts": [fact for fact in facts if fact][:limit], "edges": edges, "nodes": nodes, "query": query}

    def delete_graph(self, graph_id: str):
        self._request_json(f"{self.graphiti_base_url}/group/{graph_id}", method="DELETE", timeout=30)

    def _normalize_node(self, row: Dict[str, Any]) -> Dict[str, Any]:
        attributes = dict(row.get("attributes") or {})
        for key in ("uuid", "name", "summary", "group_id", "created_at"):
            attributes.pop(key, None)
        return {
            "uuid": row.get("uuid") or "",
            "name": row.get("name") or "",
            "labels": row.get("labels") or [],
            "summary": row.get("summary") or "",
            "attributes": attributes,
            "created_at": row.get("created_at"),
        }

    def _normalize_edge(self, row: Dict[str, Any]) -> Dict[str, Any]:
        attributes = dict(row.get("attributes") or {})
        for key in ("uuid", "name", "fact", "created_at", "valid_at", "invalid_at", "expired_at"):
            attributes.pop(key, None)
        return {
            "uuid": row.get("uuid") or "",
            "name": row.get("name") or "",
            "fact": row.get("fact") or "",
            "source_node_uuid": row.get("source_node_uuid") or "",
            "target_node_uuid": row.get("target_node_uuid") or "",
            "source_node_name": row.get("source_node_name") or "",
            "target_node_name": row.get("target_node_name") or "",
            "attributes": attributes,
            "created_at": row.get("created_at"),
            "valid_at": row.get("valid_at"),
            "invalid_at": row.get("invalid_at"),
            "expired_at": row.get("expired_at"),
            "episodes": attributes.get("episodes") or attributes.get("episode_ids") or [],
        }
