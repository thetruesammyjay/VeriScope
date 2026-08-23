from ml.verification.claim_extractor import extract_claims


def test_extract_claims_preserves_checkable_sentence_and_id():
    claims = extract_claims(
        "This is an opinion. The city has 3 hospitals. Is this correct?"
    )

    assert len(claims) == 1
    assert claims[0].claim_id == "claim-001"
    assert claims[0].text == "The city has 3 hospitals."
    assert claims[0].article_span is not None
