from __future__ import annotations

import pytest

from app.services.url_normalize import (
    InvalidLinkedInURL,
    extract_public_identifier,
    normalize_linkedin_url,
    try_normalize,
)

CANON = "https://www.linkedin.com/in/john-smith"


@pytest.mark.parametrize(
    "raw",
    [
        "https://linkedin.com/in/john-smith/",
        "https://www.linkedin.com/in/john-smith",
        "https://www.linkedin.com/in/john-smith/?trk=abc",
        "https://linkedin.com/in/john-smith?utm_source=test",
        "http://www.linkedin.com/in/John-Smith",
        "www.linkedin.com/in/john-smith",
        "linkedin.com/in/john-smith#section",
        "https://www.linkedin.com/in/john-smith/overlay/contact-info/",
    ],
)
def test_all_variants_map_to_one_canonical(raw):
    assert normalize_linkedin_url(raw) == CANON


def test_extract_public_identifier():
    assert extract_public_identifier("https://www.linkedin.com/in/jane-smith/") == "jane-smith"
    assert extract_public_identifier("https://www.linkedin.com/in/%C3%A9lodie") == "élodie"


@pytest.mark.parametrize(
    "bad",
    ["", "   ", None, "https://example.com/in/john", "https://www.linkedin.com/company/google", "not a url"],
)
def test_unusable_urls(bad):
    assert try_normalize(bad) == (None, None)
    with pytest.raises(InvalidLinkedInURL):
        normalize_linkedin_url(bad or "")


def test_pub_style_url():
    assert extract_public_identifier("https://www.linkedin.com/pub/john-smith/1/2/3") == "john-smith"
