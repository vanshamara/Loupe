import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VIZ_ROOT = REPO_ROOT / "viz"


class GraphExplorerTest(unittest.TestCase):
    def test_sample_trace_has_required_shape(self):
        trace = json.loads((VIZ_ROOT / "sample-trace.json").read_text())

        self.assertEqual(trace["version"], 1)
        self.assertGreaterEqual(len(trace["nodes"]), 8)
        self.assertGreaterEqual(len(trace["edges"]), 8)
        self.assertGreaterEqual(len(trace["per_plan_paths"]), 3)
        self.assertFalse(trace["truncated"])

        statuses = {node["status"] for node in trace["nodes"]}
        self.assertIn("Query", statuses)
        self.assertIn("Returned", statuses)
        self.assertIn("Matched", statuses)
        self.assertIn("FilteredOut", statuses)
        self.assertIn("Tombstoned", statuses)

        node_ids = {node["id"] for node in trace["nodes"]}
        for edge in trace["edges"]:
            self.assertIn(edge["from"], node_ids)
            self.assertIn(edge["to"], node_ids)
        for plan in trace["per_plan_paths"]:
            self.assertTrue(plan["ordered_events"])
            for node_id in plan["ordered_events"]:
                self.assertIn(node_id, node_ids)

    def test_viewer_uses_three_and_plan_overlays(self):
        html = (VIZ_ROOT / "index.html").read_text()
        app = (VIZ_ROOT / "src" / "app.js").read_text()

        self.assertIn("trace-canvas", html)
        self.assertIn("three@0.183.2", app)
        self.assertIn("OrbitControls", app)
        self.assertIn("per_plan_paths", app)
        self.assertIn("ordered_events", app)
        self.assertIn("Plan Overlay", html)

    def test_viewer_does_not_embed_raw_vectors(self):
        text = "\n".join(
            [
                (VIZ_ROOT / "sample-trace.json").read_text(),
                (VIZ_ROOT / "src" / "app.js").read_text(),
            ]
        )

        self.assertNotIn('"vector"', text)
        self.assertNotIn('"embedding"', text)


if __name__ == "__main__":
    unittest.main()
