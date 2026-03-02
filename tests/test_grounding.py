from src.agent import build_answer


def test_abstain_when_no_evidence():
    packet = build_answer("irrelevant", [])
    assert packet.abstained is True
    assert packet.citations == []
