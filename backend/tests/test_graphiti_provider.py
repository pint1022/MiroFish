from app.config import Config
from app.services.graphiti_client import GraphitiClient
from app.services import graph_builder
from app.services import zep_entity_reader
from app.services import zep_graph_memory_updater
from app.services import zep_tools


class FakeGraphitiClient:
    def __init__(self):
        self.added_text = []

    def create_graph(self, name):
        return "mirofish_test"

    def add_texts(self, graph_id, texts, source_description="MiroFish"):
        self.added_text.append((graph_id, texts, source_description))
        return ["episode-1"]

    def get_all_nodes(self, graph_id):
        return [
            {
                "uuid": "node-1",
                "name": "Alice",
                "labels": ["Entity", "Person"],
                "summary": "A person",
                "attributes": {"role": "analyst"},
            }
        ]

    def get_all_edges(self, graph_id):
        return []

    def get_node(self, node_uuid):
        return {
            "uuid": node_uuid,
            "name": "Alice",
            "labels": ["Entity", "Person"],
            "summary": "A person",
            "attributes": {},
        }

    def get_node_edges(self, node_uuid):
        return []

    def search(self, graph_id, query, limit=10, scope="edges"):
        return {"facts": ["Alice knows Graphiti"], "edges": [], "nodes": [], "query": query}


def _enable_graphiti(monkeypatch):
    monkeypatch.setattr(Config, "GRAPH_PROVIDER", "graphiti", raising=False)
    monkeypatch.setattr(Config, "ZEP_API_KEY", None)


def test_graph_builder_uses_graphiti_without_zep_key(monkeypatch):
    _enable_graphiti(monkeypatch)
    monkeypatch.setattr(graph_builder, "GraphitiClient", FakeGraphitiClient, raising=False)

    service = graph_builder.GraphBuilderService()

    assert service.create_graph("Test") == "mirofish_test"


def test_graph_builder_waits_with_graph_id_in_graphiti_mode(monkeypatch):
    _enable_graphiti(monkeypatch)

    class WaitingGraphitiClient(FakeGraphitiClient):
        def __init__(self):
            super().__init__()
            self.wait_args = None

        def wait_for_episodes(self, graph_id, expected_count, timeout=600):
            self.wait_args = (graph_id, expected_count, timeout)

    fake_client = WaitingGraphitiClient()
    monkeypatch.setattr(graph_builder, "GraphitiClient", lambda: fake_client, raising=False)

    service = graph_builder.GraphBuilderService()
    service._wait_for_episodes("mirofish_test", ["episode-1"], lambda *_: None, timeout=7)

    assert fake_client.wait_args == ("mirofish_test", 1, 7)


def test_graphiti_wait_reports_unpersisted_messages(monkeypatch):
    monkeypatch.setattr(Config, "GRAPHITI_INGEST_TIMEOUT_SECONDS", 0, raising=False)
    monkeypatch.setattr(Config, "NEO4J_PASSWORD", "test", raising=False)

    client = GraphitiClient(neo4j_password="test")
    monkeypatch.setattr(client, "_request_json", lambda *_, **__: [])
    monkeypatch.setattr(client, "cypher", lambda *_, **__: [{"count": 0}])

    try:
        client.wait_for_episodes("mirofish_test", 1, timeout=10)
    except TimeoutError as exc:
        assert "async ingest worker" in str(exc)
    else:
        raise AssertionError("Expected Graphiti ingest timeout")


def test_entity_reader_uses_graphiti_without_zep_key(monkeypatch):
    _enable_graphiti(monkeypatch)
    monkeypatch.setattr(zep_entity_reader, "GraphitiClient", FakeGraphitiClient, raising=False)

    reader = zep_entity_reader.ZepEntityReader()
    result = reader.filter_defined_entities("mirofish_test")

    assert result.filtered_count == 1
    assert result.entities[0].name == "Alice"


def test_tools_search_uses_graphiti_without_zep_key(monkeypatch):
    _enable_graphiti(monkeypatch)
    monkeypatch.setattr(zep_tools, "GraphitiClient", FakeGraphitiClient, raising=False)

    service = zep_tools.ZepToolsService()
    result = service.search_graph("mirofish_test", "Alice")

    assert result.facts == ["Alice knows Graphiti"]


def test_memory_updater_sends_graphiti_messages(monkeypatch):
    _enable_graphiti(monkeypatch)
    fake_client = FakeGraphitiClient()
    monkeypatch.setattr(zep_graph_memory_updater, "GraphitiClient", lambda: fake_client, raising=False)

    updater = zep_graph_memory_updater.ZepGraphMemoryUpdater("mirofish_test")
    updater._send_batch_activities(
        [
            zep_graph_memory_updater.AgentActivity(
                platform="twitter",
                agent_id=1,
                agent_name="Alice",
                action_type="CREATE_POST",
                action_args={"content": "hello"},
                round_num=1,
                timestamp="2026-01-01T00:00:00",
            )
        ],
        "twitter",
    )

    assert fake_client.added_text
    assert fake_client.added_text[0][0] == "mirofish_test"
