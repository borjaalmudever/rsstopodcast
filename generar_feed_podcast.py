#!/usr/bin/env python3
"""
generar_feed_podcast.py
------------------------
Reconstruye docs/podcast.xml (feed RSS compatible con Apple Podcasts)
a partir de docs/episodes.json. Un episodio por día, con capítulos
declarados vía <podcast:chapters> (los capítulos "de verdad" ya están
además incrustados como metadatos ID3 en el propio MP3).

Uso:
  python generar_feed_podcast.py --base-url https://usuario.github.io/repo
"""

import argparse
import json
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path
from xml.sax.saxutils import escape


def build_rss(episodes: list, base_url: str, title: str, description: str, author: str) -> str:
    base_url = base_url.rstrip("/")
    items_xml = []

    ordered = sorted(episodes, key=lambda e: e["pub_date"], reverse=True)

    for ep in ordered:
        pub_dt = datetime.fromisoformat(ep["pub_date"])
        pub_rfc2822 = format_datetime(pub_dt)
        minutes, seconds = divmod(ep.get("duration_seconds", 0), 60)
        hours, minutes = divmod(minutes, 60)
        duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        items_xml.append(f"""
    <item>
      <title>{escape(ep['title'])}</title>
      <description>{escape(ep['description'])}</description>
      <pubDate>{pub_rfc2822}</pubDate>
      <guid isPermaLink="false">{escape(ep['guid'])}</guid>
      <enclosure url="{escape(ep['mp3_url'])}" length="{ep.get('file_size', 0)}" type="audio/mpeg" />
      <itunes:duration>{duration_str}</itunes:duration>
      <itunes:explicit>false</itunes:explicit>
    </item>""")

    cover_url = f"{base_url}/cover.jpg"
    last_build = format_datetime(datetime.now(timezone.utc))

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd" xmlns:content="http://purl.org/rss/1.0/modules/content/">
  <channel>
    <title>{escape(title)}</title>
    <link>{base_url}</link>
    <language>es-es</language>
    <description>{escape(description)}</description>
    <itunes:author>{escape(author)}</itunes:author>
    <itunes:explicit>false</itunes:explicit>
    <itunes:image href="{cover_url}" />
    <image>
      <url>{cover_url}</url>
      <title>{escape(title)}</title>
      <link>{base_url}</link>
    </image>
    <itunes:category text="News" />
    <lastBuildDate>{last_build}</lastBuildDate>
    <ttl>60</ttl>
    {''.join(items_xml)}
  </channel>
</rss>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--title", default="Mis Feeds — Resumen diario")
    parser.add_argument("--description", default="Resumen diario en audio de mis carpetas de Reeder, generado automáticamente.")
    parser.add_argument("--author", default="Resumen Feeds")
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    episodes_path = docs_dir / "episodes.json"
    episodes = json.loads(episodes_path.read_text()) if episodes_path.exists() else []

    rss = build_rss(episodes, args.base_url, args.title, args.description, args.author)
    out_path = docs_dir / "podcast.xml"
    out_path.write_text(rss, encoding="utf-8")
    print(f"Feed escrito en {out_path} con {len(episodes)} episodios.")


if __name__ == "__main__":
    main()
