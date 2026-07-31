#!/usr/bin/env python3
"""
resumen_feeds.py
-----------------
Genera UN ÚNICO episodio de podcast al día que recorre todas las carpetas
de un OPML exportado desde Reeder, con capítulos (uno por carpeta) dentro
del propio MP3.

Flujo:
  1. Lee el OPML y agrupa los feeds por carpeta.
  2. Para cada carpeta con novedades, descarga sus entradas recientes y pide
     a Claude un segmento hablado, seleccionando las noticias más relevantes
     (no solo las más recientes) dentro de un presupuesto de palabras.
  3. Genera una intro (saludo + fecha + efeméride real del día, vía Wikipedia)
     y un cierre breve.
  4. Convierte cada segmento a audio con Fish Audio y los concatena con ffmpeg.
  5. Escribe capítulos ID3 (uno por segmento) en el MP3 final.
  6. Genera un título llamativo y una descripción de 3 frases con Claude.
  7. Añade el episodio a docs/episodes.json (un registro por día).

Pensado para correr en GitHub Actions, pero funciona igual en local.
"""

import argparse
import json
import math
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

try:
    from mutagen.id3 import ID3, ID3NoHeaderError, CHAP, CTOC, TIT2, CTOCFlags
except ImportError:
    ID3 = None


MESES_ES = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
]

TRANSICIONES = [
    "Empezamos con la sección de {folder}.",
    "Vamos ahora con {folder}.",
    "Pasamos a {folder}.",
    "Toca hablar de {folder}.",
    "Seguimos con {folder}.",
]

PORTADA_TRANSICION = (
    "Vamos a repasar las últimas noticias: lo que llevan hoy en portada los "
    "principales medios."
)

# Nombres de fuente que se pronuncian mal por defecto en TTS: se sustituyen
# por una versión fonética antes de pasarlos a Claude, así el guion ya los
# cita correctamente.
PRONUNCIACIONES = {
    "jenesaispop.com": "Yé Né Sé Pop",
    "jenesaispop": "Yé Né Sé Pop",
    "variety.com": "Varáyeti",
    "variety": "Varáyeti",
}


def aplicar_pronunciaciones(texto: str) -> str:
    for original, pronunciacion in PRONUNCIACIONES.items():
        texto = re.sub(re.escape(original), pronunciacion, texto, flags=re.IGNORECASE)
    return texto


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


def fetch_recent_entries(feeds: list, hours: int, max_entries: int = 40) -> list:
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
                "feed": aplicar_pronunciaciones(feed_title),
                "title": entry.get("title", "Sin título"),
                "summary": summary[:600],
                "published": published,
            })
    entries.sort(key=lambda e: e["published"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    # Se recorta solo para no reventar el tamaño del prompt; la SELECCIÓN de
    # cuáles importan de verdad la hace Claude, no este corte por fecha.
    return entries[:max_entries]


def fetch_portada_articles(api_key: str, hours: int = 24, per_source: int = 10) -> list:
    """Trae, vía NewsAPI.ai, las noticias de las últimas `hours` horas de
    El País y El Mundo, quedándose con las `per_source` mejores de cada
    medio según su socialScore."""
    if not api_key:
        return []

    fuentes = {"El País": "elpais.com", "El Mundo": "elmundo.es"}
    date_start = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%d")
    entries = []

    for nombre_medio, source_uri in fuentes.items():
        try:
            arts = _consultar_newsapi_ai(api_key, source_uri, date_start, per_source,
                                          location_uri="http://en.wikipedia.org/wiki/Spain")
            if len(arts) < 3:
                # El filtro geográfico ha dejado muy pocos resultados: repetimos
                # sin filtrar por ubicación para no dejar la sección casi vacía.
                arts = _consultar_newsapi_ai(api_key, source_uri, date_start, per_source)

            for a in arts:
                published = None
                try:
                    published = datetime.fromisoformat(
                        a.get("dateTimePub", "").replace("Z", "+00:00")
                    )
                except Exception:
                    pass
                entries.append({
                    "feed": nombre_medio,
                    "title": a.get("title", "Sin título"),
                    "summary": strip_html(a.get("body", ""))[:600],
                    "published": published,
                })
        except Exception as e:
            print(f"  [aviso] no se pudo obtener portada de {nombre_medio}: {e}")

    return entries


def _consultar_newsapi_ai(api_key: str, source_uri: str, date_start: str,
                           per_source: int, location_uri: str = None) -> list:
    payload = {
        "action": "getArticles",
        "sourceUri": source_uri,
        "dateStart": date_start,
        "lang": "spa",
        "articlesSortBy": "socialScore",
        "articlesCount": per_source,
        "includeArticleSocialScore": True,
        "resultType": "articles",
        "apiKey": api_key,
    }
    if location_uri:
        payload["locationUri"] = location_uri

    resp = requests.post(
        "https://eventregistry.org/api/v1/article/getArticles",
        json=payload, timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("articles", {}).get("results", [])


# ---------- 3. Efeméride real del día (Wikipedia) ----------

def get_efemeride(dt: datetime, max_events: int = 6) -> list:
    headers = {"User-Agent": "ResumenFeedsPodcast/1.0 (uso personal, sin fines comerciales)"}
    url = f"https://api.wikimedia.org/feed/v1/wikipedia/es/onthisday/selected/{dt.month:02d}/{dt.day:02d}"
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        events = resp.json().get("selected", [])[:max_events]
        return [
            {"year": ev.get("year"), "text": strip_html(ev.get("text", ""))}
            for ev in events if ev.get("text")
        ]
    except Exception as e:
        print(f"  [aviso] no se pudo obtener efeméride: {e}")
    return []


# ---------- 4. Normalización de texto para TTS ----------

def normalize_for_tts(text: str) -> str:
    """Red de seguridad: convierte símbolos problemáticos a palabras,
    por si el modelo deja alguno sin transcribir a texto natural."""
    text = aplicar_pronunciaciones(text)
    text = text.replace("%", " por ciento")
    text = re.sub(r"(\d)\s*€", r"\1 euros", text)
    text = text.replace("€", "euros")
    text = re.sub(r"\$\s*(\d)", r"\1 dólares", text)
    text = text.replace("&", " y ")
    text = text.replace("#", " almohadilla ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------- 5. Guiones con Claude ----------

def build_intro_script(dt: datetime, efemerides: list, client) -> str:
    fecha_natural = f"{dt.day} de {MESES_ES[dt.month - 1]}"
    if efemerides:
        opciones = "\n".join(f"- Año {e['year']}: {e['text']}" for e in efemerides)
        efemeride_txt = f"""Además, elige UNA (y solo una) de estas efemérides reales de hoy —
la que consideres más interesante o llamativa PARA UNA AUDIENCIA DE ESPAÑA—
y redáctala con tus propias palabras de forma natural:

{opciones}

No inventes ninguna efeméride que no esté en esta lista, y no menciones
más de una."""
    else:
        efemeride_txt = "No incluyas ninguna efeméride: simplemente da los buenos días y la fecha."

    prompt = f"""Escribe la introducción hablada de un podcast diario en español, para ser LEÍDA EN VOZ ALTA.

Debe:
- Empezar saludando "Buenos días" y decir que hoy es {fecha_natural}.
- Dirigirte al oyente en segunda persona del SINGULAR ("tú", "tienes"), nunca en plural ("vosotros", "tenéis").
- {efemeride_txt}
- No inventes ningún dato histórico que no te haya dado yo explícitamente arriba.
- Ser breve: 2 a 4 frases en total.
- Todos los números y símbolos deben escribirse en palabras (ej. "quince por ciento", nunca "15%"). Los años sí se pueden decir con cifras normales si suena mejor en voz alta (ej "mil novecientos ochenta").
- No uses markdown ni asteriscos, solo texto plano para voz.
"""
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return normalize_for_tts("".join(b.text for b in response.content if b.type == "text").strip())


def build_folder_segment(folder_name: str, entries: list, client, word_budget: int) -> str:
    if not entries:
        return ""

    articles_text = "\n\n".join(
        f"- {e['title']} ({e['feed']}): {e['summary']}" for e in entries
    )

    instruccion_audiencias = ""
    if folder_name.strip().upper() == "TV":
        instruccion_audiencias = """
- Si entre los artículos hay datos de audiencias de televisión (normalmente
  del día anterior), EMPIEZA el segmento por ahí, antes de cualquier otra
  noticia de televisión.
- Dentro de esos datos de audiencia, si se menciona la cifra (share o
  espectadores) de alguno de estos programas, cítala explícitamente:
  "Y ahora Sonsoles", "YAS Verano", "Directo al grano", "Malas lenguas".
- MUY IMPORTANTE: solo menciones la cifra de un programa si aparece
  literalmente en los resúmenes de abajo. Si un programa de esa lista no
  tiene dato disponible, simplemente no lo menciones — no inventes ni
  estimes ninguna cifra.
"""
    elif folder_name.strip() == "Portada":
        instruccion_audiencias = """
- Prioriza noticias de ESPAÑA. Si tienes que elegir entre una noticia
  centrada en España y otra centrada en Latinoamérica de importancia
  similar, elige la de España.
"""

    prompt = f"""Eres un locutor de radio que prepara un segmento hablado en español para un podcast diario.

Sección: "{folder_name}"

Artículos disponibles (título, fuente, resumen) — hay más de los que caben en el tiempo asignado:

{articles_text}

Escribe un guion para ser LEÍDO EN VOZ ALTA (no un texto para leer con los ojos):
- Tienes un presupuesto de aproximadamente {word_budget} palabras. NO tienes que
  mencionar todos los artículos: ELIGE solo los que consideres más relevantes
  o importantes para el oyente, no simplemente los más recientes. Ignora sin
  problema los artículos menores, repetitivos o de relleno.
- Agrupa temas relacionados, no leas la lista uno por uno de forma mecánica.
- Dirígete al oyente en segunda persona del SINGULAR ("lo que tienes que saber",
  "te cuento", "esto te interesa"). NUNCA uses la segunda persona del plural
  ("tenéis", "os cuento").
- Menciona explícitamente la fuente de cada noticia dentro de la frase, de
  forma natural, por ejemplo "según publica El País..." o "The Verge cuenta
  que...". No dejes ninguna noticia sin decir de dónde sale.
- Todos los números, porcentajes, precios y símbolos deben escribirse en
  palabras para que se puedan leer en voz alta (ej. "quince por ciento" en
  vez de "15%", "treinta euros" en vez de "30€").
- Si varias noticias tratan sobre personas (artistas, famosos, deportistas,
  etc.), prioriza y ordena primero a los más conocidos o reconocibles para
  el público general, y deja para el final (o descarta si no hay espacio)
  a los perfiles emergentes o menos conocidos.
- No inventes datos que no estén en los resúmenes.
- No uses markdown, listas ni asteriscos: solo texto plano para voz.
- No incluyas saludos ni despedidas, ni menciones el nombre de la sección al
  principio (eso ya lo dice la transición previa) — empieza directo con el
  contenido.
{instruccion_audiencias}"""
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return normalize_for_tts(text)


def build_outro_script(client) -> str:
    prompt = """Escribe el cierre de un podcast diario de noticias en español, en segunda
persona del singular ("nos vemos mañana", nunca "vemos mañana" en plural).
Debe ser una sola frase breve de despedida, natural, sin markdown."""
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    )
    return normalize_for_tts("".join(b.text for b in response.content if b.type == "text").strip())


def build_title_and_description(date_str: str, guion_final: str, client) -> dict:
    prompt = f"""Este es el guion COMPLETO y DEFINITIVO del episodio de hoy ({date_str})
de un podcast de resumen de noticias — es exactamente lo que se va a leer en
voz alta, palabra por palabra:

---
{guion_final}
---

Devuelve ÚNICAMENTE un JSON válido (sin texto adicional antes ni después, sin
markdown, sin bloques de código) con esta forma exacta:
{{"titular": "...", "descripcion": "..."}}

Donde:
- "titular": un titular llamativo y breve (6 a 10 palabras) que resuma lo más
  interesante del episodio de hoy.
- "descripcion": un resumen de lo más llamativo del episodio en un MÁXIMO de
  3 frases, en español, sin markdown.

REGLA MÁS IMPORTANTE: el titular y la descripción SOLO pueden mencionar cosas
que aparezcan literalmente en el guion de arriba. Está terminantemente
prohibido mencionar cualquier noticia, dato o nombre que no esté en el texto,
aunque te parezca relevante o lo conozcas de otra forma. Si tienes dudas
sobre si algo se cuenta en el guion o no, no lo menciones.
"""
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = "".join(b.text for b in response.content if b.type == "text").strip()
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else json.loads(raw)
    except Exception:
        data = {"titular": "Resumen del día", "descripcion": "Resumen de noticias del día."}
    return data


# ---------- 6. Texto -> Audio con Fish Audio ----------

def text_to_speech_fish(text: str, out_path: Path, api_key: str,
                         model: str = "s2.1-pro-free", reference_id: str = None):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "model": model,
    }
    payload = {
        "text": text,
        "format": "mp3",
        "max_new_tokens": 8192,
    }
    if reference_id:
        payload["reference_id"] = reference_id

    resp = requests.post("https://api.fish.audio/v1/tts", headers=headers, json=payload, timeout=180)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)


def get_duration_seconds(mp3_path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(mp3_path)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def concatenate_segments(segment_paths: list, out_path: Path):
    inputs = []
    for p in segment_paths:
        inputs += ["-i", str(p)]
    n = len(segment_paths)
    filter_complex = "".join(f"[{i}:a:0]" for i in range(n)) + f"concat=n={n}:v=0:a=1[out]"
    cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", filter_complex,
           "-map", "[out]", "-c:a", "libmp3lame", "-b:a", "96k", str(out_path)]
    subprocess.run(cmd, check=True, capture_output=True)


def write_chapters(mp3_path: Path, chapters: list):
    """chapters: lista de dicts {title, start_ms, end_ms}"""
    if ID3 is None:
        print("  [aviso] mutagen no disponible, se omiten los capítulos")
        return
    try:
        tags = ID3(str(mp3_path))
    except ID3NoHeaderError:
        tags = ID3()

    child_ids = []
    for i, ch in enumerate(chapters):
        elem_id = f"chp{i}"
        child_ids.append(elem_id)
        tags.add(CHAP(
            element_id=elem_id,
            start_time=int(ch["start_ms"]),
            end_time=int(ch["end_ms"]),
            start_offset=0xFFFFFFFF,
            end_offset=0xFFFFFFFF,
            sub_frames=[TIT2(text=[ch["title"]])],
        ))
    tags.add(CTOC(
        element_id="toc",
        flags=CTOCFlags.TOP_LEVEL | CTOCFlags.ORDERED,
        child_element_ids=child_ids,
        sub_frames=[TIT2(text=["Capítulos"])],
    ))
    tags.save(str(mp3_path), v2_version=3)


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Genera un único episodio diario de podcast a partir de las carpetas de Reeder")
    parser.add_argument("opml_path")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--fish-reference-id", default=None)
    parser.add_argument("--fish-model", default="s2.1-pro-free")
    parser.add_argument("--target-minutes", type=float, default=15.0,
                         help="Duración objetivo del episodio en minutos")
    parser.add_argument("--folders", nargs="*", default=None)
    args = parser.parse_args()

    if anthropic is None:
        sys.exit("Falta instalar el SDK: pip install anthropic")
    for var in ("ANTHROPIC_API_KEY", "FISH_API_KEY"):
        if var not in os.environ:
            sys.exit(f"Falta la variable de entorno {var}")

    docs_dir = Path(args.docs_dir)
    audio_dir = docs_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    # Las transcripciones son de uso interno: se guardan FUERA de docs/, así
    # no se publican en GitHub Pages.
    transcripts_dir = Path("transcripts")
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path("tmp_audio")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    episodes_path = docs_dir / "episodes.json"
    episodes = json.loads(episodes_path.read_text()) if episodes_path.exists() else []

    client = anthropic.Anthropic()
    folders = parse_opml_by_folder(args.opml_path)
    if args.folders:
        folders = {k: v for k, v in folders.items() if k in args.folders}

    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")

    # --- Recopilar entradas por carpeta ---
    folder_entries = {}

    newsapi_key = os.environ.get("NEWSAPI_KEY")
    print("== Portada (El País + El Mundo, NewsAPI.ai) ==")
    if newsapi_key:
        try:
            portada_entries = fetch_portada_articles(newsapi_key, hours=args.hours)
            print(f"  {len(portada_entries)} artículos recuperados")
            if portada_entries:
                folder_entries["Portada"] = portada_entries
        except Exception as e:
            print(f"  [aviso] fallo recuperando Portada: {e}")
    else:
        print("  [aviso] NEWSAPI_KEY no configurada, se omite la sección Portada")

    for folder_name, feeds in folders.items():
        print(f"== {folder_name} ({len(feeds)} feeds) ==")
        entries = fetch_recent_entries(feeds, args.hours)
        print(f"  {len(entries)} artículos disponibles en las últimas {args.hours}h")
        if entries:
            folder_entries[folder_name] = entries

    if not folder_entries:
        print("Sin novedades en ninguna carpeta hoy. No se genera episodio.")
        return

    # --- Presupuesto de palabras por carpeta (proporcional, no solo recencia) ---
    WPM = 160
    total_budget = int(args.target_minutes * WPM)
    intro_budget, outro_budget = 60, 30
    remaining = max(total_budget - intro_budget - outro_budget, 300)

    weights = {f: math.sqrt(len(e)) for f, e in folder_entries.items()}
    total_weight = sum(weights.values())
    word_budgets = {}
    for f in folder_entries:
        budget = int(remaining * weights[f] / total_weight)
        word_budgets[f] = max(120, min(budget, 600))

    # --- Efeméride + intro ---
    efemeride = get_efemeride(now)
    print("Generando introducción...")
    intro_text = build_intro_script(now, efemeride, client)

    # --- Segmentos por carpeta ---
    segment_texts = {}
    rss_idx = 0
    for folder_name, entries in folder_entries.items():
        print(f"Generando segmento: {folder_name} (~{word_budgets[folder_name]} palabras)")
        if folder_name == "Portada":
            transicion = PORTADA_TRANSICION
        else:
            transicion = TRANSICIONES[rss_idx % len(TRANSICIONES)].format(folder=folder_name)
            rss_idx += 1
        cuerpo = build_folder_segment(folder_name, entries, client, word_budgets[folder_name])
        segment_texts[folder_name] = f"{transicion} {cuerpo}"

    outro_text = build_outro_script(client)

    # --- Título y descripción (basados en el guion REAL ya escrito, no en
    # las noticias en bruto antes de seleccionar, para que nunca mencionen
    # algo que luego no se cuenta) ---
    print("Generando título y descripción del episodio...")
    guion_noticias = "\n\n".join(segment_texts.values())
    meta = build_title_and_description(date_str, guion_noticias, client)
    episode_title = f"{now.day} de {MESES_ES[now.month - 1]} — {meta.get('titular', 'Resumen del día')}"
    episode_description = meta.get("descripcion", "Resumen de noticias del día.")

    # --- Guardar transcripción interna completa ---
    full_transcript = intro_text + "\n\n" + "\n\n".join(segment_texts.values()) + "\n\n" + outro_text
    (transcripts_dir / f"{date_str}.txt").write_text(full_transcript, encoding="utf-8")

    # --- Generar audio por segmento y concatenar ---
    print("Generando audio con Fish Audio...")
    segment_order = ["__intro__"] + list(segment_texts.keys()) + ["__outro__"]
    segment_audio_paths = {}
    for key in segment_order:
        text = {"__intro__": intro_text, "__outro__": outro_text}.get(key, segment_texts.get(key))
        if not text:
            continue
        safe = re.sub(r"[^\w\-]+", "_", key.strip())
        path = tmp_dir / f"{safe}.mp3"
        text_to_speech_fish(text, path, os.environ["FISH_API_KEY"],
                             model=args.fish_model, reference_id=args.fish_reference_id)
        segment_audio_paths[key] = path

    ordered_keys = [k for k in segment_order if k in segment_audio_paths]
    final_mp3 = audio_dir / f"episodio_{date_str}.mp3"
    concatenate_segments([segment_audio_paths[k] for k in ordered_keys], final_mp3)

    # --- Capítulos (offsets calculados sobre los audios individuales) ---
    chapters = []
    cursor_ms = 0.0
    for key in ordered_keys:
        dur = get_duration_seconds(segment_audio_paths[key]) * 1000
        title = "Introducción" if key == "__intro__" else ("Cierre" if key == "__outro__" else key)
        chapters.append({"title": title, "start_ms": cursor_ms, "end_ms": cursor_ms + dur})
        cursor_ms += dur
    write_chapters(final_mp3, chapters)

    duration_seconds = get_duration_seconds(final_mp3)
    file_size = final_mp3.stat().st_size

    episodes.append({
        "guid": str(uuid.uuid4()),
        "title": episode_title,
        "description": episode_description,
        "pub_date": now.isoformat(),
        "mp3_url": f"{args.base_url.rstrip('/')}/audio/{final_mp3.name}",
        "file_size": file_size,
        "duration_seconds": int(duration_seconds),
        "chapters": [{"title": c["title"], "start_ms": int(c["start_ms"])} for c in chapters],
    })
    episodes_path.write_text(json.dumps(episodes, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nListo. Episodio generado: {final_mp3.name} ({duration_seconds/60:.1f} min)")


if __name__ == "__main__":
    main()
