import feedparser, resumen_feeds as rf

folders = rf.parse_opml_by_folder("iCloud_RSS.opml")
print("Carpetas encontradas:", list(folders.keys()))

for nombre, feeds in folders.items():
    for titulo, url in feeds:
        p = feedparser.parse(url, agent=rf.USER_AGENT_RSS)
        status = getattr(p, "status", "sin status")
        print(f"[{nombre}] {titulo}: HTTP {status}, {len(p.entries)} entradas, bozo={p.bozo}")
