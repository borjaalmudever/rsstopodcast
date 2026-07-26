#!/usr/bin/env python3
"""
resumen_feeds.py
-----------------
Genera un episodio de podcast por cada carpeta de un OPML exportado desde Reeder.

Para cada carpeta:
  1. Descarga las entradas recientes de sus feeds (feedparser).
  2. Pide a Claude que escriba un guion hablado y natural.
  3. Convierte ese guion a audio (MP3) con Fish Audio.
  4. Añade el episodio a docs/episodes.json (histórico, usado luego por
     generar_feed_podcast.py para reconstruir el podcast.xml).

Pensado para correr dentro de GitHub Actions, pero funciona igual en local.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

import feedparser
import requests

try:
    import anthropic
except ImportError:
    anthropic = None


# ---------- 1. Parsear el OPML por carpetas ----------

def parse_opml_by_folder(opml_path: str) -> dict:
    tree = ET.parse(opml_path)
    body = tree.getroot().find("body")

    folders = {}
    for outline in body.findall("outline"):
        children = outline.findall("outline")
        if children:
            folder_name = outline.get("title") or outline.get("text") or "Sin nombre"
            feeds = []
            for child in children:
                xml_url = child.get("xmlUrl")
                title = child.get("title") or child.get("text") or xml_url
                if xml_url:
                    feeds.append((title, xml_url))
            folders[folder_name] = feeds
        else:
            xml_url = outline.get("xmlUrl")
            if xml_url:
                folders.setdefault("Sin carpeta", []).append(
                    (outline.get("title") or xml_url, xml_url)
                )
    return folders


# ---------- 2. Descargar entradas recientes ----------

def strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()


def fetch_recent_entries(feeds: list, hours: int) -> list:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    entries = []
    for feed_title, url in feeds:
        try:
            parsed = feedparser.parse(url)
        except Exception as e:
            print(f"  [aviso] no se pudo leer {feed_title}: {e}")
            continue

        for entry in parsed.entries:
            published = None
            for key in ("published_parsed", "updated_parsed"):
                if getattr(entry, key, None):
                    published = datetime(*entry[key][:6], tzinfo=timezone.utc)
                    break
            if published and published < cutoff:
                continue

            summary = strip_html(getattr(entry, "summary", "") or getattr(entry, "description", ""))
            entries.append({
                "feed": feed_title,
                "title": entry.get("title", "Sin título"),
                "summary": summary[:600],
                "published": published,
            })
    entries.sort(key=lambda e: e["published"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return entries


# ---------- 3. Resumen hablado con Claude ----------

def build_script_with_claude(folder_name: str, entries: list, client) -> str:
    if not entries:
        return f"No hay novedades nuevas en la carpeta {folder_name} en las últimas horas."

    articles_text = "\n\n".join(
        f"- {e['title']} ({e['feed']}): {e['summary']}" for e in entries[:25]
    )

    prompt = f"""Eres un locutor de radio que prepara un briefing de noticias hablado en español.

Carpeta de feeds: "{folder_name}"

Aquí tienes los artículos recientes (título, fuente y resumen):

{articles_text}

Escribe un guion para ser LEÍDO EN VOZ ALTA (no un texto para leer con los ojos):
- Empieza con una frase breve tipo "Esto es lo destacado en {folder_name}".
- Agrupa temas relacionados, no leas la lista uno por uno de forma mecánica.
- Dirígete al oyente en segunda persona del SINGULAR ("lo que tienes que saber", "te cuento",
  "esto te interesa"). NUNCA uses la segunda persona del plural ("tenéis", "os cuento").
- Menciona explícitamente la fuente de cada noticia dentro de la frase, de forma natural,
  por ejemplo "según publica El País..." o "The Verge cuenta que...". No dejes ninguna noticia
  sin decir de dónde sale.
- Tono natural, conversacional, como un briefing de podcast de 2-4 minutos (300-500 palabras).
- No inventes datos que no estén en los resúmenes.
- No uses markdown, listas ni asteriscos: solo texto plano para voz.
- Termina con una frase de cierre breve.
"""

    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text").strip()


# ---------- 4. Texto -> Audio con Fish Audio ----------

def text_to_speech_fish(text: str, out_path: Path, api_key: str,
                         model: str = "s2.1-pro-free", reference_id: str = None):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "model": model,
    }
    payload = {"text": text, "format": "mp3"}
    if reference_id:
        payload["reference_id"] = reference_id

    resp = requests.post("https://api.fish.audio/v1/tts", headers=headers, json=payload, timeout=180)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)


def get_duration_seconds(mp3_path: Path) -> int:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(mp3_path)],
            capture_output=True, text=True, check=True,
        )
        return int(float(result.stdout.strip()))
    except Exception:
        return 0


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Genera episodios de podcast por carpeta de Reeder")
    parser.add_argument("opml_path", help="Ruta al archivo .opml exportado de Reeder")
    parser.add_argument("--hours", type=int, default=24, help="Ventana de artículos a incluir (horas)")
    parser.add_argument("--docs-dir", default="docs", help="Carpeta servida por GitHub Pages")
    parser.add_argument("--base-url", required=True,
                         help="URL pública base de GitHub Pages, ej: https://usuario.github.io/repo")
    parser.add_argument("--fish-reference-id", default=None,
                         help="ID de la voz de Fish Audio a usar (opcional)")
    parser.add_argument("--fish-model", default="s2.1-pro-free",
                         help="Modelo de Fish Audio a usar")
    parser.add_argument("--folders", nargs="*", default=None,
                         help="Procesar solo estas carpetas. Por defecto: todas.")
    args = parser.parse_args()

    if anthropic is None:
        sys.exit("Falta instalar el SDK: pip install anthropic")
    for var in ("ANTHROPIC_API_KEY", "FISH_API_KEY"):
        if var not in os.environ:
            sys.exit(f"Falta la variable de entorno {var}")

    docs_dir = Path(args.docs_dir)
    audio_dir = docs_dir / "audio"
    transcripts_dir = docs_dir / "transcripts"
    audio_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    episodes_path = docs_dir / "episodes.json"
    episodes = json.loads(episodes_path.read_text()) if episodes_path.exists() else []

    client = anthropic.Anthropic()
    folders = parse_opml_by_folder(args.opml_path)
    if args.folders:
        folders = {k: v for k, v in folders.items() if k in args.folders}

    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")

    for folder_name, feeds in folders.items():
        print(f"\n== {folder_name} ({len(feeds)} feeds) ==")
        entries = fetch_recent_entries(feeds, args.hours)
        print(f"  {len(entries)} artículos nuevos en las últimas {args.hours}h")

        if not entries:
            print("  Sin novedades, se omite el episodio de hoy para esta carpeta.")
            continue

        script_text = build_script_with_claude(folder_name, entries, client)

        safe_name = re.sub(r"[^\w\-]+", "_", folder_name.strip())
        mp3_filename = f"{safe_name}_{date_str}.mp3"
        mp3_path = audio_dir / mp3_filename

        txt_filename = f"{safe_name}_{date_str}.txt"
        (transcripts_dir / txt_filename).write_text(script_text, encoding="utf-8")
        transcript_url = f"{args.base_url.rstrip('/')}/transcripts/{txt_filename}"

        try:
            text_to_speech_fish(script_text, mp3_path, os.environ["FISH_API_KEY"],
                                 model=args.fish_model, reference_id=args.fish_reference_id)
        except Exception as e:
            print(f"  [error] no se pudo generar audio con Fish Audio: {e}")
            continue

        duration = get_duration_seconds(mp3_path)
        file_size = mp3_path.stat().st_size

        episodes.append({
            "guid": str(uuid.uuid4()),
            "title": f"{folder_name} — {date_str}",
            "folder": folder_name,
            "description": script_text,
            "transcript_url": transcript_url,
            "pub_date": now.isoformat(),
            "mp3_url": f"{args.base_url.rstrip('/')}/audio/{mp3_filename}",
            "file_size": file_size,
            "duration_seconds": duration,
        })
        print(f"  Episodio generado: {mp3_filename}")

    episodes_path.write_text(json.dumps(episodes, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nListo. {len(episodes)} episodios en total en {episodes_path}")


if __name__ == "__main__":
    main()
