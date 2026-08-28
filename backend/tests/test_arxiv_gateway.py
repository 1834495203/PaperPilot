from app.infrastructure.tools.arxiv import ArxivPaperSearchGateway


def test_parse_feed_returns_typed_paper() -> None:
    feed = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2401.12345v1</id>
        <updated>2024-01-20T10:00:00Z</updated>
        <published>2024-01-18T10:00:00Z</published>
        <title>  A Useful   Research Paper </title>
        <summary> First line.\n Second line. </summary>
        <author><name>Ada Lovelace</name></author>
        <link title="pdf" href="https://arxiv.org/pdf/2401.12345v1" />
      </entry>
    </feed>"""

    papers = ArxivPaperSearchGateway._parse_feed(feed)

    assert len(papers) == 1
    assert papers[0].arxiv_id == "2401.12345v1"
    assert papers[0].title == "A Useful Research Paper"
    assert papers[0].summary == "First line. Second line."
    assert papers[0].authors == ["Ada Lovelace"]

