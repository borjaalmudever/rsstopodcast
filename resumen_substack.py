#!/usr/bin/env python3
"""
resumen_substack.py
--------------------
Genera la edición especial de los sábados: repaso semanal de las
publicaciones de Substack suscritas, con tres secciones (NOTICIAS DE
INTERNET, HERRAMIENTAS ONLINE, REFLEXIONES DIGITALES) y, en la última, un
diálogo a dos voces por cada fuente.

Reutiliza el pipeline ya construido en resumen_feeds.py (Gemini con Claude
de respaldo, Fish Audio, mezcla con ffmpeg, capítulos ID3, episodes.json)
importándolo como módulo, en vez de duplicar su lógica: como ese script
solo ejecuta main() bajo `if __name__ == "__main__"`, importarlo aquí no
dispara nada.

Pensado para correr como SEGUNDO episodio del sábado (además del episodio
diario de siempre), en un workflow de GitHub Actions aparte — ver
.github/workflows/saturday_substack.yml. Esta primera versión no lleva
música de fondo bajo las secciones (solo cabecera y cierre llevan
música); añadirla más adelante es un cambio local a este fichero.
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import feedparser

import resumen_feeds as rf

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

try:
    import anthropic
except ImportError:
    anthropic = None


SUBSTACK_SECTION_ORDER = ["NOTICIAS DE INTERNET", "HERRAMIENTAS ONLINE", "REFLEXIONES DIGITALES"]

# Pausa de silencio entre bloques de voz (y entre música y voz) al encadenar
# el episodio. Sin música de fondo bajo las secciones esta primera versión,
# así que aquí no hay crossfade musical posible: se usan pausas cortas de
# silencio real en vez de la envolvente de ducking que sí usa el diario.
MUSIC_PAUSE_SEC = 0.6
DIALOGUE_TURN_PAUSE_SEC = 0.5

SYSTEM_DIALOGO = """Eres el guionista de la sección REFLEXIONES DIGITALES de un podcast semanal
en español de España que repasa publicaciones de Substack.

Escribes un DIÁLOGO entre dos voces que comentan una única publicación:
- Voz A es la locutora/el locutor habitual del podcast.
- Voz B es una persona invitada que coanfitriona esta sección.

Todo lo que escribes se va a leer EN VOZ ALTA con un sintetizador de voz, así que:

FORMATO
- Cada turno es una frase o dos, en prosa continua, sin títulos, viñetas, asteriscos, markdown ni emojis.
- Alternáis turnos de forma natural, como una conversación real: os hacéis eco el uno del otro, os
  contestáis, no os limitáis a leer datos por turnos.
- Escribís SIEMPRE en español. No mezcláis palabras sueltas en inglés (ni siquiera para números o
  cantidades), salvo nombres propios o títulos que no tengan traducción habitual.

TRATO AL OYENTE
- Habláis entre vosotros dos, no os dirigís directamente al oyente.

CIFRAS Y SÍMBOLOS
- Los números, porcentajes y símbolos van escritos con letras, tal y como se leen.
- Los años SIEMPRE van en cifras, nunca escritos con letras.

CONTENIDO
- Resumís lo esencial de la publicación, destacáis lo más interesante y añadís una reflexión
  personal breve, siempre a partir de lo que dice el texto que se os entrega.
- Solo afirmáis aquello que aparezca de forma explícita en ese texto; si algo no está, no lo
  inventáis.
- No repetís la misma idea con otras palabras: cada turno aporta algo nuevo a la conversación."""


def _build_esquema_dialogo():
    if types is None:
        return None
    return types.Schema(
        type=types.Type.OBJECT,
        properties={
            "turnos": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "voz": types.Schema(type=types.Type.STRING, description="A o B"),
                        "texto": types.Schema(type=types.Type.STRING),
                    },
                    required=["voz", "texto"],
                ),
            ),
        },
        required=["turnos"],
    )


# ---------- Substacks: última publicación por fuente, con autor y texto largo ----------

def fetch_substack_entries(feeds: list, hours: int) -> list:
    """Para cada fuente, la entrada más reciente publicada dentro de `hours`
    (si no hay ninguna, esa fuente se omite). A diferencia de
    rf.fetch_recent_entries (pensada para resúmenes cortos multi-noticia),
    aquí interesa UNA sola publicación por fuente con su autor y el texto
    más completo posible, para poder sostener un diálogo sobre ella."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    results = []
    for feed_title, url in feeds:
        try:
            parsed = feedparser.parse(url)
        except Exception as e:
            print(f"  [aviso] no se pudo leer {feed_title}: {e}")
            continue

        mejor = None
        for entry in parsed.entries:
            published = None
            for key in ("published_parsed", "updated_parsed"):
                if getattr(entry, key, None):
                    published = datetime(*entry[key][:6], tzinfo=timezone.utc)
                    break
            if not published or published < cutoff:
                continue
            if mejor is None or published > mejor["published"]:
                content_list = getattr(entry, "content", None)
                if content_list:
                    contenido = rf.strip_html(content_list[0].get("value", ""))
                else:
                    contenido = rf.strip_html(getattr(entry, "summary", "") or getattr(entry, "description", ""))
                autor = entry.get("author") or None
                mejor = {
                    "feed": rf.aplicar_pronunciaciones(feed_title),
                    "title": entry.get("title", "Sin título"),
                    "author": rf.aplicar_pronunciaciones(autor) if autor else None,
                    "content": contenido[:4000],
                    "published": published,
                }
        if mejor:
            results.append(mejor)
        else:
            print(f"  [info] sin publicaciones de {feed_title} en las últimas {hours}h")
    return results


# ---------- Guiones ----------

def build_bienvenida_substack(dt: datetime, client, claude_client=None) -> str:
    fecha_natural = f"{rf.DIAS_ES[dt.weekday()]} {dt.day} de {rf.MESES_ES[dt.month - 1]}"
    prompt = f"""ENCARGO
Escribe la bienvenida hablada de la edición especial de los sábados de este podcast, que repasa
las publicaciones de Substack de la semana.

- Saluda y di que hoy es {fecha_natural}.
- Anuncia que toca poner los Substack al día (puedes decirlo con esas palabras o una variante muy
  cercana, pero el sentido tiene que ser exactamente ese).
- Extensión: una o dos frases en total.
- Termina justo ahí, sin despedida ni cierre: esto es solo la apertura del episodio.

Responde solo con el texto de la bienvenida."""
    text, _ = rf.generate_script(client, claude_client, prompt, system_instruction=rf.SYSTEM_LOCUTOR,
                                  max_output_tokens=300)
    text = rf.recortar_a_frase_completa(text)
    return rf.normalize_for_tts(text)


def build_dialogue_segment(source_name: str, entry: dict, client, claude_client=None) -> list:
    """Devuelve una lista de turnos [{"voz": "A"|"B", "texto": "..."}] a
    partir del contenido de una publicación. Ante cualquier fallo de
    formato, degrada a un turno único de voz A en vez de reventar el
    episodio entero por una sola fuente."""
    prompt = f"""PUBLICACIÓN A COMENTAR
Fuente: {source_name}
Título: {entry['title']}

Texto:
{entry['content']}

ENCARGO
Escribid el diálogo de esta sección sobre la publicación de arriba: entre cuatro y ocho turnos en
total, alternando voz A y voz B de forma natural (no tiene que ser estrictamente ABAB si la
conversación pide otra cosa).

Devuelve ÚNICAMENTE un JSON válido (sin texto adicional antes ni después, sin markdown, sin
bloques de código) con esta forma exacta:
{{"turnos": [{{"voz": "A", "texto": "..."}}, {{"voz": "B", "texto": "..."}}]}}"""

    try:
        raw, _ = rf.generate_script(client, claude_client, prompt, system_instruction=SYSTEM_DIALOGO,
                                     max_output_tokens=1400, response_schema=_build_esquema_dialogo())
        data = json.loads(raw)
        turnos = data.get("turnos") or []
        limpios = []
        for t in turnos:
            voz = str(t.get("voz", "")).strip().upper()
            texto = rf.normalize_for_tts(str(t.get("texto", "")).strip())
            if voz in ("A", "B") and texto:
                limpios.append({"voz": voz, "texto": texto})
        if not limpios:
            raise ValueError("respuesta sin turnos válidos")
        return limpios
    except Exception as e:
        print(f"  [aviso] no se pudo generar el diálogo de '{source_name}' ({str(e)[:120]}); se usa un resumen de reserva")
        return [{"voz": "A", "texto": rf.normalize_for_tts(f"Esta semana {source_name} ha publicado {entry['title']}.")}]


# ---------- Audio: silencio y encadenado con pausas ----------

def _make_silence(duration_sec: float, out_path: Path):
    cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
           "-t", str(duration_sec), "-c:a", "libmp3lame", "-b:a", "96k", str(out_path)]
    subprocess.run(cmd, check=True, capture_output=True)


def concat_with_pauses(paths: list, pause_sec: float, out_path: Path, tmp_dir: Path):
    """Concatena varios mp3 intercalando un silencio corto entre cada uno,
    para una pausa natural entre tramos de voz (o entre música y voz) sin
    necesidad de crossfade musical, que aquí no aplica al no haber música
    de fondo bajo las secciones."""
    if len(paths) == 1:
        rf.concatenate_segments(paths, out_path)
        return
    silence_path = tmp_dir / "_silencio.mp3"
    _make_silence(pause_sec, silence_path)
    interleaved = []
    for i, p in enumerate(paths):
        interleaved.append(p)
        if i < len(paths) - 1:
            interleaved.append(silence_path)
    rf.concatenate_segments(interleaved, out_path)


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Genera la edición especial de los sábados (Substacks)")
    parser.add_argument("opml_path")
    parser.add_argument("--hours", type=int, default=168, help="Ventana de publicaciones a considerar (por defecto una semana)")
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--fish-reference-id", default=None, help="Voz A: la locutora/el locutor habitual del podcast")
    parser.add_argument("--fish-reference-id-2", default="a57f25d2318c4765a5b9be7e7f34617c",
                         help="Voz B: coanfitrión/a del diálogo de REFLEXIONES DIGITALES")
    parser.add_argument("--fish-model", default="s2.1-pro-free")
    parser.add_argument("--music-dir", default="assets/music", help="Carpeta con la cabecera y el cierre")
    parser.add_argument("--target-minutes", type=float, default=15.0,
                         help="Duración objetivo de NOTICIAS DE INTERNET + HERRAMIENTAS ONLINE")
    parser.add_argument("--episode-image", default="substack_cover.jpg",
                         help="Fichero de portada (episode art) dentro de --docs-dir, servido en "
                              "{base-url}/{episode-image} y declarado como <itunes:image> del episodio")
    args = parser.parse_args()

    if genai is None or types is None:
        sys.exit("Falta instalar el SDK: pip install google-genai")
    for var in ("GEMINI_API_KEY", "FISH_API_KEY"):
        if var not in os.environ:
            sys.exit(f"Falta la variable de entorno {var}")

    docs_dir = Path(args.docs_dir)
    audio_dir = docs_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir = Path("transcripts")
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path("tmp_audio_substack")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    music_dir = Path(args.music_dir)
    required_music_files = [rf.CABECERA_FILENAME, rf.CIERRE_FILENAME]
    missing_music = [f for f in required_music_files if not (music_dir / f).exists()]
    if missing_music:
        sys.exit(f"Faltan ficheros de música en {music_dir}: {', '.join(missing_music)}")

    # Episode art: todo episodio de esta edición lleva la misma portada,
    # distinta de la del canal (docs/cover.jpg), declarada como
    # <itunes:image> propia del <item> en el feed.
    episode_image_path = docs_dir / args.episode_image
    if not episode_image_path.exists():
        sys.exit(f"Falta el episode art en {episode_image_path}")
    episode_image_url = f"{args.base_url.rstrip('/')}/{args.episode_image}"

    episodes_path = docs_dir / "episodes.json"
    episodes = json.loads(episodes_path.read_text()) if episodes_path.exists() else []

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    claude_client = None
    if anthropic is not None and os.environ.get("ANTHROPIC_API_KEY"):
        claude_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    else:
        print("  [aviso] ANTHROPIC_API_KEY no configurada: sin respaldo si Gemini se satura")
    print(f"Modelo de guion: {rf.GEMINI_MODEL} (temperatura {rf.GEMINI_TEMPERATURE})")

    folders = rf.parse_opml_by_folder(args.opml_path)
    folders = {k: v for k, v in folders.items() if k in SUBSTACK_SECTION_ORDER}

    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")

    # --- NOTICIAS DE INTERNET / HERRAMIENTAS ONLINE: como una sección normal del diario ---
    folder_entries = {}
    for folder_name in ("NOTICIAS DE INTERNET", "HERRAMIENTAS ONLINE"):
        feeds = folders.get(folder_name, [])
        if not feeds:
            continue
        print(f"== {folder_name} ({len(feeds)} feeds) ==")
        entries = rf.fetch_recent_entries(feeds, args.hours)
        print(f"  {len(entries)} artículos disponibles en las últimas {args.hours}h")
        if entries:
            folder_entries[folder_name] = entries

    # --- REFLEXIONES DIGITALES: última publicación por fuente ---
    reflexiones_feeds = folders.get("REFLEXIONES DIGITALES", [])
    reflexiones_entries = []
    if reflexiones_feeds:
        print(f"== REFLEXIONES DIGITALES ({len(reflexiones_feeds)} feeds) ==")
        reflexiones_entries = fetch_substack_entries(reflexiones_feeds, args.hours)
        print(f"  {len(reflexiones_entries)} fuentes con publicación nueva en las últimas {args.hours}h")

    if not folder_entries and not reflexiones_entries:
        print("Sin novedades en ninguna sección esta semana. No se genera la edición Substack.")
        return

    # --- Presupuesto de palabras (solo para las 2 secciones normales; el
    # diálogo de REFLEXIONES DIGITALES se gestiona por número de turnos) ---
    WPM = 160
    total_budget = int(args.target_minutes * WPM)
    word_budgets = {}
    if folder_entries:
        weights = {f: math.sqrt(len(e)) for f, e in folder_entries.items()}
        total_weight = sum(weights.values())
        for f in folder_entries:
            budget = int(total_budget * weights[f] / total_weight)
            # Tope más alto que en el diario (que reparte 15 min entre 4
            # secciones): aquí como mucho hay 2 secciones de texto
            # (NOTICIAS DE INTERNET/HERRAMIENTAS ONLINE) compitiendo por el
            # mismo presupuesto de 15 min, así que cada una puede necesitar
            # bastante más margen para no quedarse corta.
            word_budgets[f] = max(120, min(budget, 900))

    print("Generando bienvenida...")
    bienvenida_text = build_bienvenida_substack(now, client, claude_client)

    transicion_fija = {
        "NOTICIAS DE INTERNET": "Vamos con las noticias de internet.",
        "HERRAMIENTAS ONLINE": "Seguimos con las herramientas online.",
        "REFLEXIONES DIGITALES": "Y vamos con las reflexiones digitales.",
    }
    primera_seccion = True

    def _abre_seccion(folder_name):
        nonlocal primera_seccion
        if primera_seccion:
            primera_seccion = False
            return f"Empezamos con las novedades sobre {folder_name.lower()}."
        return transicion_fija[folder_name]

    # --- Segmentos de texto: NOTICIAS DE INTERNET / HERRAMIENTAS ONLINE ---
    segment_texts = {}
    for folder_name in ("NOTICIAS DE INTERNET", "HERRAMIENTAS ONLINE"):
        if folder_name not in folder_entries:
            continue
        print(f"Generando segmento: {folder_name} (~{word_budgets[folder_name]} palabras)")
        cuerpo = rf.build_folder_segment(folder_name, folder_entries[folder_name], client,
                                          word_budgets[folder_name], dt=now, claude_client=claude_client)
        if not cuerpo.strip():
            print(f"  [aviso] segmento vacío para '{folder_name}', se omite la sección")
            continue
        transicion = _abre_seccion(folder_name)
        segment_texts[folder_name] = f"{transicion} {cuerpo}"

    # --- REFLEXIONES DIGITALES: apertura + diálogo por fuente ---
    reflexiones_clip_texts = []
    reflexiones_voice_specs = []  # lista de (texto, voz)
    if reflexiones_entries:
        apertura_seccion = _abre_seccion("REFLEXIONES DIGITALES")
        reflexiones_voice_specs.append((apertura_seccion, "A"))
        reflexiones_clip_texts.append(apertura_seccion)
        for entry in reflexiones_entries:
            fuente = entry["feed"]
            autor = entry.get("author")
            if autor:
                frase_apertura = rf.normalize_for_tts(f"Vamos con la última publicación de {fuente}, de {autor}.")
            else:
                frase_apertura = rf.normalize_for_tts(f"Vamos con la última publicación de {fuente}.")
            reflexiones_voice_specs.append((frase_apertura, "A"))
            reflexiones_clip_texts.append(frase_apertura)

            print(f"  Generando diálogo: {fuente}")
            turnos = build_dialogue_segment(fuente, entry, client, claude_client)
            for turno in turnos:
                reflexiones_voice_specs.append((turno["texto"], turno["voz"]))
                reflexiones_clip_texts.append(f"[{turno['voz']}] {turno['texto']}")

    if not segment_texts and not reflexiones_voice_specs:
        print("No se pudo generar ningún segmento hoy. No se genera la edición Substack.")
        return

    # --- Título y descripción ---
    print("Generando descripción del episodio...")
    partes_transcripcion = [bienvenida_text] + list(segment_texts.values()) + reflexiones_clip_texts
    guion_texto = "\n\n".join(partes_transcripcion)
    meta = rf.build_title_and_description(date_str, guion_texto, client, claude_client)
    episode_title = f"{now.day} de {rf.MESES_ES[now.month - 1]} - EDICIÓN SUBSTACK"
    episode_description = meta.get("descripcion", "Resumen de las publicaciones de Substack de la semana.")

    full_transcript = "\n\n".join(partes_transcripcion)
    (transcripts_dir / f"{date_str}_substack.txt").write_text(full_transcript, encoding="utf-8")

    # --- Voz ---
    print("Generando audio con Fish Audio...")
    fish_key = os.environ["FISH_API_KEY"]
    voice_ids = {"A": args.fish_reference_id, "B": args.fish_reference_id_2}

    def _synth(text, voz, idx, tag):
        safe_tag = re.sub(r"[^\w\-]+", "_", tag)
        raw_path = tmp_dir / f"{safe_tag}_{idx}_raw.mp3"
        norm_path = tmp_dir / f"{safe_tag}_{idx}.mp3"
        rf.text_to_speech_fish(text, raw_path, fish_key, model=args.fish_model, reference_id=voice_ids[voz])
        rf.normalize_voice_audio(raw_path, norm_path)
        return norm_path

    bienvenida_path = _synth(bienvenida_text, "A", 0, "bienvenida")

    section_clip_paths = {}
    for folder_name, texto in segment_texts.items():
        section_clip_paths[folder_name] = [_synth(texto, "A", 0, folder_name)]

    if reflexiones_voice_specs:
        clips = [_synth(texto, voz, i, "reflexiones") for i, (texto, voz) in enumerate(reflexiones_voice_specs)]
        section_clip_paths["REFLEXIONES DIGITALES"] = clips

    ordered_sections_present = [f for f in SUBSTACK_SECTION_ORDER if f in section_clip_paths]
    section_merged_paths = {}
    for folder_name in ordered_sections_present:
        clips = section_clip_paths[folder_name]
        safe = re.sub(r"[^\w\-]+", "_", folder_name)
        merged_path = tmp_dir / f"seccion_{safe}.mp3"
        concat_with_pauses(clips, DIALOGUE_TURN_PAUSE_SEC, merged_path, tmp_dir)
        section_merged_paths[folder_name] = merged_path

    print("  Encadenando cabecera, bienvenida, secciones y cierre...")
    cabecera_path = tmp_dir / "cabecera_norm.mp3"
    cierre_path = tmp_dir / "cierre_norm.mp3"
    rf.normalize_jingle_audio(music_dir / rf.CABECERA_FILENAME, cabecera_path)
    rf.normalize_jingle_audio(music_dir / rf.CIERRE_FILENAME, cierre_path)

    all_blocks = [cabecera_path, bienvenida_path] + [section_merged_paths[f] for f in ordered_sections_present] + [cierre_path]
    final_mp3 = audio_dir / f"episodio_substack_{date_str}.mp3"
    concat_with_pauses(all_blocks, MUSIC_PAUSE_SEC, final_mp3, tmp_dir)

    # --- Capítulos ---
    pause_ms = MUSIC_PAUSE_SEC * 1000
    cabecera_dur = rf.get_duration_seconds(cabecera_path) * 1000
    bienvenida_dur = rf.get_duration_seconds(bienvenida_path) * 1000
    intro_end = cabecera_dur + pause_ms + bienvenida_dur

    chapters = [{"title": "Introducción", "start_ms": 0.0, "end_ms": intro_end}]
    cursor_ms = intro_end
    for folder_name in ordered_sections_present:
        cursor_ms += pause_ms
        dur = rf.get_duration_seconds(section_merged_paths[folder_name]) * 1000
        chapters.append({"title": folder_name, "start_ms": cursor_ms, "end_ms": cursor_ms + dur})
        cursor_ms += dur
    cursor_ms += pause_ms
    cierre_dur = rf.get_duration_seconds(cierre_path) * 1000
    chapters.append({"title": "Cierre", "start_ms": cursor_ms, "end_ms": cursor_ms + cierre_dur})

    rf.write_chapters(final_mp3, chapters)

    duration_seconds = rf.get_duration_seconds(final_mp3)
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
        "image_url": episode_image_url,
    })
    episodes_path.write_text(json.dumps(episodes, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nListo. Edición Substack generada: {final_mp3.name} ({duration_seconds/60:.1f} min)")


if __name__ == "__main__":
    main()
