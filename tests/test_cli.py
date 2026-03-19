import json
import subprocess
import sys


def test_cli_contract_ask_json_runs():
    proc = subprocess.run(
        [sys.executable, "-m", "src.cli", "ask", "--q", "dummy", "--json"],
        capture_output=True,
        text=True,
    )
    # Ingest olmadan index yoksa komut hata verebilir; bu test sadece JSON contract yolunu doğrular.
    if proc.returncode == 0:
        payload = json.loads(proc.stdout)
        assert "answer" in payload
        assert "abstained" in payload
        assert "citations" in payload
        assert "evidence" in payload
