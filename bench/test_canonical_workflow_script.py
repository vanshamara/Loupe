import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run-canonical-workflow.sh"


class CanonicalWorkflowScriptTest(unittest.TestCase):
    def test_script_is_valid_bash(self):
        subprocess.run(["bash", "-n", str(SCRIPT_PATH)], check=True)

    def test_script_keeps_generated_data_out_of_repo(self):
        text = SCRIPT_PATH.read_text()

        self.assertIn("LOUPE_WORKFLOW_DIR", text)
        self.assertIn("mktemp -d", text)
        self.assertIn("base.jsonl", text)
        self.assertIn("metrics.prom", text)
        self.assertIn("run_loupe metrics", text)
        self.assertNotIn("datasets/arxiv", text)


if __name__ == "__main__":
    unittest.main()
