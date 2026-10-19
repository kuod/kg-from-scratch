from src.pdf_convert import _strip_repeated_headers_footers


def test_table_separators_survive_header_footer_stripping():
    md = "|a|b|\n|---|---|\n|1|2|\n\n|---|---|\n|3|4|\n|---|---|\n|---|---|"
    assert _strip_repeated_headers_footers(md).count("|---|---|") == 4


def test_running_header_is_still_removed():
    md = "\n".join(["Journal of X 2024", "body one", "Journal of X 2024", "body two",
                    "Journal of X 2024", "body three", "Journal of X 2024"])
    out = _strip_repeated_headers_footers(md)
    assert "Journal of X" not in out and "body two" in out
