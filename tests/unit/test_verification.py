from ml.verification.claim_extractor import Claim
from ml.verification.evidence_extractor import EvidencePassage
from ml.verification.verifier import verify_claim


def test_verify_claim_supports_non_contradictory_evidence():
    claim = Claim("claim-001", "The city has 3 hospitals.")
    evidence = [EvidencePassage("https://example.org", "The city has 3 hospitals.")]

    assert verify_claim(claim, evidence).status == "supported"


def test_verify_claim_detects_explicit_contradiction():
    claim = Claim("claim-001", "The city has 3 hospitals.")
    evidence = [EvidencePassage("https://example.org", "That claim is false.")]

    assert verify_claim(claim, evidence).status == "contradicted"


def test_verify_claim_returns_insufficient_without_evidence():
    claim = Claim("claim-001", "The city has 3 hospitals.")

    assert verify_claim(claim, []).status == "insufficient"
