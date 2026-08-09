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
     a Gemini un segmento hablado, seleccionando las noticias más relevantes
     (no solo las más recientes) dentro de un presupuesto de palabras.
  3. Genera una intro (saludo + fecha + efeméride real del día, vía Wikipedia),
     seguida de la cabecera musical, y una música de fondo por sección.
  4. Convierte cada segmento a audio con Fish Audio, lo mezcla con su música
     de fondo y encadena cabecera + secciones + cierre con crossfade.
  5. Escribe capítulos ID3 (uno por segmento) en el MP3 final.
  6. Genera un título llamativo y una descripción de 3 frases con Gemini.
  7. Añade el episodio a docs/episodes.json (un registro por día).

Pensado para correr en GitHub Actions, pero funciona igual en local.

Las llamadas a Gemini van centralizadas en `gemini_generate()`, con system
instruction, temperatura baja, tope de tokens, ajustes de seguridad aptos
para contenido informativo, salida estructurada donde procede y reintentos
con espera exponencial. Si Gemini agota esos reintentos (rachas largas de
503 "high demand" del modelo -lite gratuito), `generate_script()` cae a
Claude como respaldo, siempre que haya ANTHROPIC_API_KEY configurada.

Variables de entorno opcionales:
  GEMINI_MODEL             modelo a usar (por defecto gemini-3.5-flash-lite)
  GEMINI_TEMPERATURE       temperatura de generación (por defecto 0.35)
  GEMINI_SAFETY_THRESHOLD  umbral de los filtros (por defecto BLOCK_ONLY_HIGH)
  ANTHROPIC_API_KEY        si está presente, habilita el respaldo con Claude
  ANTHROPIC_MODEL          modelo de respaldo a usar (por defecto claude-sonnet-5)
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

import feedparser
import requests

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


# El alias "-latest" apunta siempre al último modelo de la familia y puede
# cambiar de un día para otro sin que toques una línea de código. En un cron
# diario conviene FIJAR una versión concreta y actualizarla a mano cuando se
# haya probado. Se puede sobreescribir con GEMINI_MODEL.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

# La temperatura por defecto de la API es 1.0. Para un guion de noticias cuya
# premisa es "no inventes nada", bajarla es la mejora más barata que existe:
# reduce la deriva creativa y hace mucho más consistente el cumplimiento de
# las reglas de formato.
GEMINI_TEMPERATURE = float(os.environ.get("GEMINI_TEMPERATURE", "0.35"))

# Reintentos: un 429 o un 503 puntual de Gemini no puede tumbar el episodio
# del día entero en GitHub Actions. Los modelos "-lite" gratuitos tienen
# rachas de alta demanda (503 "This model is currently experiencing high
# demand") que pueden durar más de treinta segundos, así que el presupuesto
# de reintentos tiene que aguantar minuto y medio largo, no solo segundos.
GEMINI_MAX_RETRIES = 6
GEMINI_RETRY_BASE_SEC = 4

# Respaldo cuando Gemini agota sus reintentos (rachas de 503 más largas que
# el presupuesto de arriba): se reintenta el guion con Claude, tal y como se
# hacía antes de adoptar Gemini. Solo se activa si hay ANTHROPIC_API_KEY.
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

# Se descubre en la primera llamada del proceso si GEMINI_MODEL admite
# thinking_config, y se recuerda para el resto de llamadas: evita repetir la
# misma petición fallida (y su reintento) en cada una de las ~7 llamadas que
# hace un episodio completo, ya que el modelo no cambia durante la ejecución.
_thinking_config_soportado = True

# En cuanto una llamada agota los reintentos de Gemini y cae a Claude, se da
# por caído para el resto de la ejecución: una racha de 503 "high demand"
# dura minutos, no segundos, así que es casi seguro que las llamadas
# siguientes del mismo episodio también fallarán. Sin esto, cada una de las
# ~7 llamadas repetiría el ciclo completo de reintentos (hasta ~124s de
# espera) antes de caer a Claude, en vez de ir directa.
_gemini_caido = False

_RETRYABLE_MARKERS = (
    "429", "500", "502", "503", "504",
    "RESOURCE_EXHAUSTED", "UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED",
    "overloaded", "timeout", "Timeout",
)

# Esto es un pipeline de NOTICIAS: sucesos, guerra, tribunales. Con los
# umbrales por defecto, Gemini puede bloquear una respuesta perfectamente
# legítima y devolver texto vacío, que acabaría llegando mudo al TTS.
# BLOCK_ONLY_HIGH es el umbral más permisivo disponible sin permisos
# especiales (si tu proyecto tiene habilitado OFF/BLOCK_NONE, puedes usarlo).
_SAFETY_THRESHOLD = os.environ.get("GEMINI_SAFETY_THRESHOLD", "BLOCK_ONLY_HIGH")
_SAFETY_CATEGORIES = (
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
)


class GeminiEmptyResponse(RuntimeError):
    """La API respondió pero sin texto utilizable (bloqueo de seguridad,
    corte por tokens antes de emitir nada, etc.)."""


def _safety_settings():
    return [
        types.SafetySetting(category=c, threshold=_SAFETY_THRESHOLD)
        for c in _SAFETY_CATEGORIES
    ]


def _build_config(system_instruction, max_output_tokens, response_schema, sin_thinking):
    kwargs = {
        "temperature": GEMINI_TEMPERATURE,
        "safety_settings": _safety_settings(),
    }
    if system_instruction:
        kwargs["system_instruction"] = system_instruction
    if max_output_tokens:
        kwargs["max_output_tokens"] = max_output_tokens
    if response_schema is not None:
        # Salida estructurada NATIVA: garantiza JSON válido por contrato, en
        # vez de pedirlo con palabras y rescatarlo luego con una expresión
        # regular (que era el patrón necesario con la API anterior).
        kwargs["response_mime_type"] = "application/json"
        kwargs["response_schema"] = response_schema
    if sin_thinking:
        # Los modelos 2.5 Flash-Lite ya vienen sin "thinking", pero si algún
        # día se apunta a un Flash/Pro el razonamiento consume presupuesto de
        # salida y puede provocar cortes con texto vacío. Si el modelo no
        # admite este parámetro, la llamada se reintenta sin él (ver abajo).
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    return types.GenerateContentConfig(**kwargs)


def _extract_text(response) -> str:
    """Saca el texto ignorando las partes de razonamiento, sin que un
    candidato vacío haga saltar una excepción del SDK."""
    try:
        if response.text:
            return response.text.strip()
    except Exception:
        pass
    trozos = []
    for cand in (response.candidates or []):
        content = getattr(cand, "content", None)
        for part in (getattr(content, "parts", None) or []):
            if getattr(part, "thought", False):
                continue
            if getattr(part, "text", None):
                trozos.append(part.text)
    return "".join(trozos).strip()


def _motivo_vacio(response) -> str:
    feedback = getattr(response, "prompt_feedback", None)
    if feedback is not None and getattr(feedback, "block_reason", None):
        return f"prompt bloqueado ({feedback.block_reason})"
    try:
        cand = response.candidates[0]
        return f"finish_reason={cand.finish_reason}, safety={getattr(cand, 'safety_ratings', None)}"
    except Exception:
        return "sin candidatos en la respuesta"


def gemini_generate(client, prompt: str, system_instruction: str = None,
                    max_output_tokens: int = None, response_schema=None):
    """Devuelve (texto, truncado_por_limite_de_tokens).

    Añade sobre la llamada pelada: system instruction, temperatura baja,
    tope real de tokens, ajustes de seguridad aptos para noticias, salida
    estructurada opcional y reintentos con espera exponencial.
    """
    global _thinking_config_soportado
    sin_thinking = _thinking_config_soportado
    ultimo_error = None

    intento = 0
    while intento < GEMINI_MAX_RETRIES:
        try:
            config = _build_config(system_instruction, max_output_tokens,
                                   response_schema, sin_thinking)
            response = client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt, config=config,
            )
        except Exception as e:
            mensaje = str(e)
            # Algunos modelos (los de la serie 3) rechazan thinking_budget,
            # pero no siempre lo dicen en el mensaje de error: gemini-3.5
            # -flash-lite, por ejemplo, devuelve un genérico "400
            # INVALID_ARGUMENT: Request contains an invalid argument" sin
            # mencionar "thinking" en ningún sitio. Así que, ante CUALQUIER
            # fallo del primer intento con thinking_config activo, se
            # reintenta una vez sin ese parámetro antes de aplicar la lógica
            # de reintentos normal (no se puede confiar en el texto del
            # mensaje para detectarlo). Este sondeo no cuenta como intento:
            # no puede restarle presupuesto de reintentos a un fallo temporal
            # real (p.ej. un 503 de sobrecarga del modelo).
            if sin_thinking:
                print(f"  [aviso] fallo con thinking_config activo ({mensaje[:120]}), se repite sin ese ajuste")
                sin_thinking = False
                _thinking_config_soportado = False
                continue
            ultimo_error = e
            if any(m in mensaje for m in _RETRYABLE_MARKERS) and intento < GEMINI_MAX_RETRIES - 1:
                espera = GEMINI_RETRY_BASE_SEC * (2 ** intento)
                print(f"  [aviso] fallo temporal de Gemini ({mensaje[:120]}). Reintento en {espera}s")
                time.sleep(espera)
                intento += 1
                continue
            raise

        text = _extract_text(response)
        truncated = False
        try:
            truncated = "MAX_TOKENS" in str(response.candidates[0].finish_reason)
        except Exception:
            pass

        if text:
            return text, truncated

        motivo = _motivo_vacio(response)
        ultimo_error = GeminiEmptyResponse(motivo)
        if intento < GEMINI_MAX_RETRIES - 1:
            espera = GEMINI_RETRY_BASE_SEC * (2 ** intento)
            print(f"  [aviso] Gemini devolvió texto vacío ({motivo}). Reintento en {espera}s")
            time.sleep(espera)
            intento += 1
            continue
        # Fallar de forma RUIDOSA: un texto vacío que siguiera adelante
        # acabaría como un tramo mudo dentro del mp3 sin que nadie se entere.
        raise GeminiEmptyResponse(f"Gemini no devolvió texto utilizable: {motivo}")

    raise ultimo_error or RuntimeError("Gemini: fallo desconocido")


def claude_generate(client, prompt: str, system_instruction: str = None,
                    max_output_tokens: int = None):
    """Igual forma que gemini_generate() (texto, truncado) pero contra
    Claude. Sin reintentos propios: se usa como respaldo puntual cuando
    Gemini ya agotó los suyos, no como ruta principal."""
    kwargs = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": max_output_tokens or 2000,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_instruction:
        kwargs["system"] = system_instruction
    response = client.messages.create(**kwargs)
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    truncated = response.stop_reason == "max_tokens"
    return text, truncated


def generate_script(gemini_client, claude_client, prompt: str, system_instruction: str = None,
                    max_output_tokens: int = None, response_schema=None):
    """Genera el guion con Gemini y, si agota sus reintentos (típicamente una
    racha larga de 503 "high demand" del modelo -lite gratuito), cae a Claude
    para no perder el episodio del día entero. Sin ANTHROPIC_API_KEY
    configurada (claude_client=None), el fallo de Gemini se propaga igual
    que antes. En cuanto Gemini se confirma caído (ver _gemini_caido), las
    llamadas siguientes de la misma ejecución van directas a Claude."""
    global _gemini_caido
    if not (claude_client is not None and _gemini_caido):
        try:
            return gemini_generate(gemini_client, prompt, system_instruction=system_instruction,
                                   max_output_tokens=max_output_tokens, response_schema=response_schema)
        except Exception as e:
            if claude_client is None:
                raise
            print(f"  [aviso] Gemini agotó sus reintentos ({str(e)[:120]}); se usa Claude como respaldo")
            _gemini_caido = True
    else:
        print("  [aviso] Gemini ya se confirmó caído en esta ejecución; se usa Claude directamente")
    return claude_generate(claude_client, prompt, system_instruction=system_instruction,
                           max_output_tokens=max_output_tokens)

try:
    from mutagen.id3 import ID3, ID3NoHeaderError, CHAP, CTOC, TIT2, CTOCFlags
except ImportError:
    ID3 = None


MESES_ES = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
]

# Índice: datetime.weekday() (0 = lunes) -> nombre del día.
DIAS_ES = [
    "lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo",
]

TRANSICIONES = [
    "Vamos ahora con {folder}.",
    "Pasamos a {folder}.",
    "Toca hablar de {folder}.",
    "Seguimos con {folder}.",
]

# Nombres de fuente que se pronuncian mal por defecto en TTS: se sustituyen
# por una versión fonética antes de pasarlos a Gemini, así el guion ya los
# cita correctamente.
PRONUNCIACIONES = {
    "jenesaispop.com": "Yé-Né-Sé Pop",
    "jenesaispop": "Yé-Né-Sé Pop",
    "variety.com": "Varáieti",
    "variety": "Varáieti",
    "xataka": "Shataka",
    "antena 3 noticias 1": "Antena tres noticias uno",
    "la 1": "La Uno",
    "grand prix": "Gran Priks",
    # El modelo no siempre reproduce estas palabras fonéticas inventadas tal
    # cual (no son vocabulario real, así que puede variarlas ligeramente al
    # generar texto): estas entradas corrigen las variantes observadas en
    # transcripciones reales, además de la forma "correcta" de arriba.
    "varáyeti": "Varáieti",
    "yé né sé pop": "Yé-Né-Sé Pop",
}


def aplicar_pronunciaciones(texto: str) -> str:
    # Los límites de palabra (\b) evitan que entradas con dígitos sueltos
    # (p.ej. "la 1") se cuelen dentro de otro número más largo como "la 10".
    for original, pronunciacion in PRONUNCIACIONES.items():
        texto = re.sub(r"\b" + re.escape(original) + r"\b", pronunciacion,
                       texto, flags=re.IGNORECASE)
    return texto


# Música de fondo por sección: nombre de carpeta OPML -> mp3 dentro de
# --music-dir. Cualquier carpeta que no aparezca aquí (p.ej. "MEDIA TECH")
# queda excluida del episodio.
SECTION_MUSIC = {
    "TELEVISIÓN": "02_TV.mp3",
    "GEEK": "03_GEEK.mp3",
    "POPCORN": "04_POPCORN.mp3",
    "CULTURA POP": "05_CULTURA.mp3",
}

# Orden fijo del episodio, independiente del orden de las carpetas dentro del
# OPML (que puede cambiar si alguien reordena o edita el fichero a mano).
SECTION_ORDER = ["TELEVISIÓN", "GEEK", "CULTURA POP", "POPCORN"]

CABECERA_FILENAME = "00_CABECERA.mp3"
CIERRE_FILENAME = "06_CIERRE.mp3"

# Envolvente de volumen de la música de fondo bajo cada sección: entra a un
# golpe de entrada ("sting", volumen normal de la pista, similar a la
# cabecera) con una subida rápida ("con fuerza"), y luego baja ("duck") al
# nivel de fondo de forma progresiva y NATURAL: el descenso empieza un poco
# antes de que arranque la voz pero no termina hasta un rato después de que
# la locución ya ha empezado (se solapan), en vez de cortar en seco justo
# antes de la voz. Se mantiene baja mientras se habla, y se desvanece a
# silencio en una cola corta tras terminar la locución.
#
# MUSIC_REF_LUFS es el nivel al que se normaliza cada música (con `loudnorm`)
# ANTES de aplicar la envolvente: es, por tanto, el volumen real del "sting".
# Antes la voz nunca se normalizaba (cada guion salía de Fish Audio con el
# volumen "natural" que le diera la gana, distinto segmento a segmento) y
# cabecera/cierre se usaban tal cual sin pasar por loudnorm, así que
# MUSIC_REF_LUFS solo se había calibrado cerca del nivel crudo de la
# cabecera, nunca contra la voz. Ahora TODO tramo de audio del episodio pasa
# por una única pasada de `loudnorm`: la voz (intro y cada sección) a
# VOICE_REF_LUFS, y la música (golpe de entrada de cada sección, cabecera y
# cierre) a MUSIC_REF_LUFS, calibrado por debajo de VOICE_REF_LUFS para que
# el "sting" no tape la voz. MUSIC_DUCK_DB es cuántos dB por debajo de
# MUSIC_REF_LUFS cae el fondo mientras se habla (la envolvente solo ATENÚA
# desde ese punto, nunca vuelve a normalizar, para no acabar recortando dos
# veces el volumen).
VOICE_REF_LUFS = -16.0          # objetivo de volumen para toda locución (intro y secciones)
MUSIC_PRE_ROLL_SEC = 1.5        # duración de pre-roll antes de que arranque la voz
MUSIC_FADE_IN_SEC = 0.3         # subida rápida hasta el golpe de entrada
MUSIC_DUCK_FADE_SEC = 1.3       # duración de la bajada al nivel de fondo (progresiva)
MUSIC_DUCK_OVERLAP_SEC = 0.6    # cuánto continúa bajando la música tras arrancar la voz
MUSIC_TAIL_SEC = 1.2            # cola tras la voz, antes de silencio (corta)
MUSIC_REF_LUFS = -19.0          # ~3 dB por debajo de VOICE_REF_LUFS
# Baja en la misma proporción (4 dB) que MUSIC_REF_LUFS respecto a su valor
# anterior (-15.0), para que el nivel ABSOLUTO del fondo bajo la voz no
# cambie (-15-26 = -41 antes, -19-22 = -41 ahora): ese nivel de fondo ya se
# dio por bueno en una ronda de ajuste anterior y no debía tocarse — solo
# baja el golpe de entrada/sting, no el fondo bajo la voz.
MUSIC_DUCK_DB = 22.0
# Solape entre pistas consecutivas al encadenarlas (cabecera -> secciones ->
# cierre): cae dentro de las zonas de pre-roll/tail, que son solo música
# (nunca voz), así que el crossfade no puede pisar dos locuciones.
MUSIC_CROSSFADE_SEC = 0.8
# Comprime el rango dinámico de la música ANTES de normalizar/atenuar: sin
# esto, pasajes internos ya de por sí más flojos de la propia canción (no el
# punto de bucle) pueden quedar casi en silencio al sumarles el "duck" y,
# como se repiten en cada vuelta del loop, se notan como un corte periódico.
MUSIC_COMPRESSOR = "acompressor=threshold=-20dB:ratio=4:attack=20:release=250:makeup=2"


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
    # cuáles importan de verdad la hace Gemini, no este corte por fecha.
    return entries[:max_entries]


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

_UNIDADES = ["", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"]
_DIEZ_A_DIECINUEVE = ["diez", "once", "doce", "trece", "catorce", "quince",
                       "dieciséis", "diecisiete", "dieciocho", "diecinueve"]
_DECENAS = ["", "", "veinte", "treinta", "cuarenta", "cincuenta", "sesenta", "setenta", "ochenta", "noventa"]
# 21-29 no son "veinti" + unidad a pelo: veintidós/veintitrés/veintiséis
# llevan tilde que no sale de una concatenación simple.
_VEINTIUNO_A_VEINTINUEVE = ["veintiuno", "veintidós", "veintitrés", "veinticuatro", "veinticinco",
                            "veintiséis", "veintisiete", "veintiocho", "veintinueve"]
# Centenas SIEMPRE en masculino: se usan para leer años ("mil ochocientos
# noventa y seis"), que llevan artículo masculino implícito ("el año..."),
# y como red de seguridad genérica para cualquier otro dígito suelto que se
# cuele en el guion. 500/700/900 son irregulares (quinientos/setecientos/
# novecientos, no "cincocientos/sietecientos/nuevecientos").
_CENTENAS = ["", "ciento", "doscientos", "trescientos", "cuatrocientos", "quinientos",
             "seiscientos", "setecientos", "ochocientos", "novecientos"]


def _dos_digitos_a_palabras(n: int) -> str:
    if n < 10:
        return _UNIDADES[n]
    if n < 20:
        return _DIEZ_A_DIECINUEVE[n - 10]
    if n < 30:
        return "veinte" if n == 20 else _VEINTIUNO_A_VEINTINUEVE[n - 21]
    decena, unidad = divmod(n, 10)
    if unidad == 0:
        return _DECENAS[decena]
    return f"{_DECENAS[decena]} y {_UNIDADES[unidad]}"


def _tres_digitos_a_palabras(n: int) -> str:
    if n == 100:
        return "cien"
    centena, resto = divmod(n, 100)
    partes = []
    if centena:
        partes.append(_CENTENAS[centena])
    if resto:
        partes.append(_dos_digitos_a_palabras(resto))
    return " ".join(partes)


def numero_a_palabras(n: int) -> str:
    """Convierte un entero a su forma en palabras en español. Pensado sobre
    todo para años (de ahí las centenas siempre en masculino), pero sirve de
    red de seguridad genérica para cualquier dígito suelto que llegue hasta
    aquí: según las reglas del locutor, todo número que no sea un año se
    escribe ya en letras, así que lo que sobrevive como dígitos sueltos es
    casi siempre un año. Por encima de un millón, o en negativos, se deja el
    número tal cual en vez de arriesgar una conversión incorrecta."""
    if n == 0:
        return "cero"
    if n < 0 or n >= 1_000_000:
        return str(n)
    if n < 1000:
        return _tres_digitos_a_palabras(n)
    miles, resto = divmod(n, 1000)
    prefijo = "mil" if miles == 1 else f"{_tres_digitos_a_palabras(miles)} mil"
    if resto == 0:
        return prefijo
    return f"{prefijo} {_tres_digitos_a_palabras(resto)}"


def normalize_for_tts(text: str) -> str:
    """Red de seguridad: convierte símbolos y dígitos sueltos a palabras,
    por si el modelo deja alguno sin transcribir a texto natural.

    Los años se piden en cifras a propósito (ver SYSTEM_LOCUTOR): así se
    evita que el modelo los deletree mal él mismo (p.ej. "mil cuatrovecientos"
    en vez de "mil cuatrocientos"). Pero el sintetizador de voz (Fish Audio)
    tiene su propio conversor de cifras a palabras, y ese conversor se ha
    visto en producción leyendo centenas en femenino ("mil ochocientas
    noventa y seis" para 1896, en vez de "ochocientos"). Para no depender de
    lo que haga el TTS con un número, aquí se convierten explícitamente
    todos los dígitos sueltos a palabras ANTES de enviar el texto a Fish
    Audio, con numero_a_palabras() (que sí tiene el género correcto)."""
    text = aplicar_pronunciaciones(text)
    text = text.replace("%", " por ciento")
    text = re.sub(r"(\d)\s*€", r"\1 euros", text)
    text = text.replace("€", "euros")
    text = re.sub(r"\$\s*(\d)", r"\1 dólares", text)
    text = text.replace("&", " y ")
    text = text.replace("#", " almohadilla ")
    text = re.sub(r"\b\d+\b", lambda m: numero_a_palabras(int(m.group(0))), text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------- 5. Guiones con Gemini ----------

# La persona y las reglas de estilo que NO cambian de una llamada a otra van
# en la system instruction, no repetidas dentro de cada prompt: Gemini las
# respeta de forma bastante más consistente ahí que embebidas en el turno de
# usuario, y además quedan fuera del texto variable.
SYSTEM_LOCUTOR = """Eres el guionista y locutor de un podcast diario de noticias en español de España.

Todo lo que escribes se va a leer EN VOZ ALTA con un sintetizador de voz, así que:

FORMATO
- Escribes prosa continua en texto plano: frases seguidas, tal y como se pronuncian.
- Un único bloque de texto corrido, sin títulos, sin viñetas, sin guiones de lista, sin asteriscos, sin markdown y sin emojis.
- Escribes SIEMPRE en español. No mezclas palabras sueltas en inglés (ni siquiera para números o cantidades: "doscientos", nunca "two hundred" ni "dos hundred"), salvo nombres propios o títulos de obras que no tengan traducción habitual.

TRATO AL OYENTE
- Hablas siempre con UNA sola persona, en segunda persona del singular: "tú", "tienes", "te cuento", "esto te interesa".

CIFRAS Y SÍMBOLOS
- Los números, porcentajes, precios y símbolos van escritos con letras, tal y como se leen: "quince por ciento", "treinta euros", "cien millones", "y" en lugar del ampersand.
- Los años SIEMPRE van en cifras, nunca escritos con letras: "1992", no "mil novecientos noventa y dos". Deletrear un año a mano es la fuente más habitual de números mal formados ("mil cuatrovecientos" en vez de "mil cuatrocientos"); las cifras se convierten a palabras de forma fiable más adelante, antes de generar el audio.
- Las centenas (doscientos/doscientas, trescientos/trescientas, cuatrocientos/cuatrocientas... novecientos/novecientas) concuerdan en género con lo que cuentan. Por defecto usa la forma masculina ("trescientos euros", "cuatrocientos empleados", "novecientos mil espectadores"), y solo la femenina cuando lo contado es explícitamente femenino ("trescientas personas", "novecientas páginas"). Ante la duda, masculino.

VERACIDAD
- Solo afirmas aquello que aparezca de forma explícita en los datos que te llegan en cada encargo.
- Cuando un dato no está en esos materiales, lo omites y continúas con el resto: prefieres contar menos cosas que rellenar un hueco."""

SYSTEM_META = """Eres el editor de un podcast diario de noticias en español de España.

Redactas el titular y la descripción del episodio a partir del guion que se te entrega.
Escribes en español natural, sin markdown, y ciñéndote exclusivamente a lo que el guion dice."""


def _tope_tokens(word_budget: int) -> int:
    """Tope real de tokens de salida a partir del presupuesto de palabras.

    El presupuesto por sí solo es una sugerencia que ningún modelo sabe
    contar; esto es el freno de emergencia que impide que un segmento se
    desmadre. Se deja margen holgado (en español un token va por debajo de
    una palabra) para que el corte, si llega, caiga después del contenido
    útil y lo recorte `recortar_a_frase_completa`.
    """
    return int(word_budget * 2.6) + 300


def build_intro_script(dt: datetime, efemerides: list, client, claude_client=None) -> str:
    # El día de la semana se calcula aquí y se le da hecho al modelo: si no
    # se le da, el modelo lo "adivina" por su cuenta y puede acertar el día
    # y mes pero fallar el día de la semana (visto en producción: dijo
    # "jueves" en una fecha que era sábado).
    fecha_natural = f"{DIAS_ES[dt.weekday()]} {dt.day} de {MESES_ES[dt.month - 1]}"
    if efemerides:
        opciones = "\n".join(f"- Año {e['year']}: {e['text']}" for e in efemerides)
        efemeride_txt = f"""Efemérides reales de hoy — elige UNA sola, la que te parezca más
interesante o llamativa PARA UNA AUDIENCIA DE ESPAÑA, y cuéntala con tus
propias palabras de forma natural:

{opciones}

La efeméride que menciones tiene que ser una de esta lista, y mencionas
exactamente una."""
    else:
        efemeride_txt = "Hoy no hay efeméride: da los buenos días, di la fecha y termina ahí."

    prompt = f"""{efemeride_txt}

ENCARGO
Escribe la introducción hablada del episodio de hoy.

- Empieza saludando con "Buenos días" y di que hoy es {fecha_natural}, exactamente así (día de la
  semana, día del mes y mes): no calcules ni cambies el día de la semana, usa el que se te da aquí.
- Continúa con la efeméride elegida, si la hay.
- Extensión: de dos a cuatro frases en total.
- Los únicos datos históricos que puedes dar son los de la lista de arriba.
- Termina justo después del saludo/efeméride, sin ninguna despedida ni
  cierre ("que tengas un buen día", "nos vemos", "hasta luego"...): esto es
  solo la apertura del episodio, no el final, así que no se dice adiós
  todavía.

Responde solo con el texto de la introducción."""
    text, _ = generate_script(client, claude_client, prompt, system_instruction=SYSTEM_LOCUTOR,
                              max_output_tokens=400)
    text = recortar_a_frase_completa(text)
    return normalize_for_tts(text)


def recortar_a_frase_completa(texto: str) -> str:
    """Si el texto termina a media frase, recorta hasta el último punto,
    exclamación o interrogación completo, para no dejar pasar nunca una
    idea a medias al audio.

    Se llama SIEMPRE (no solo cuando la API marca truncado), así que un
    punto final seguido de espacio (`\\s`) no basta: la respuesta de la API
    normalmente termina justo en el punto, sin espacio de sobra detrás. Por
    eso el final de frase también cuenta como válido si el punto está al
    final de la cadena (`$`), no solo si le sigue un espacio."""
    texto = texto.rstrip()
    finales = [m.end() for m in re.finditer(r"[\.\!\?](?=\s|$)", texto)]
    if not finales:
        return texto
    ultimo = finales[-1]
    # Si el texto ya termina justo en una frase completa (con poco margen
    # de diferencia), no tocamos nada.
    if len(texto) - ultimo <= 2:
        return texto
    return texto[:ultimo].strip()


# Muestra de tono y formato. Gemini aprende más de un ejemplo que de otra
# viñeta de reglas: aquí ve de un vistazo la prosa corrida, la fuente citada
# dentro de la frase, las cifras con letras y el enlace entre dos temas
# relacionados. El contenido es inventado a propósito para que no pueda
# colarse en el guion real.
EJEMPLO_SEGMENTO = """El Confidencial cuenta que la compañía cerrará su planta de Vitoria a final de año, con unos cuatrocientos empleos afectados; la dirección lo achaca a la caída de pedidos, aunque el comité de empresa lo discute y ya ha convocado paros para la semana que viene. Con esto enlaza otra historia que publica Expansión: el sector encadena tres trimestres de descensos y las previsiones para el año que viene tampoco invitan al optimismo."""


def build_folder_segment(folder_name: str, entries: list, client, word_budget: int,
                          dt: datetime = None, claude_client=None) -> str:
    if not entries:
        return ""

    def _fecha_articulo(e):
        pub = e.get("published")
        if not pub:
            return "fecha desconocida"
        return f"{DIAS_ES[pub.weekday()]} {pub.day} de {MESES_ES[pub.month - 1]}"

    # La fecha de cada artículo va en los materiales para que el modelo
    # pueda razonar sobre si un dato sigue vigente o es un resto de días
    # atrás (ver instrucción específica de TV más abajo).
    articles_text = "\n\n".join(
        f"- {e['title']} ({e['feed']}, publicado el {_fecha_articulo(e)}): {e['summary']}"
        for e in entries
    )

    # Aproximadamente cuántas noticias caben en el presupuesto. Es un ancla
    # mucho más fiable que la cifra de palabras: el modelo no sabe contar
    # palabras, pero sí sabe contar noticias y frases.
    n_noticias = max(3, min(8, round(word_budget / 80)))

    instruccion_especifica = ""
    # OJO: la carpeta se llama "TELEVISIÓN" en SECTION_MUSIC. Comparar contra
    # "TV" a secas hacía que este bloque no se aplicara nunca.
    if folder_name.strip().upper() in {"TV", "TELEVISIÓN", "TELEVISION"}:
        ref_dt = dt or datetime.now(timezone.utc)
        hoy_txt = f"{DIAS_ES[ref_dt.weekday()]} {ref_dt.day} de {MESES_ES[ref_dt.month - 1]}"
        ayer_dt = ref_dt - timedelta(days=1)
        ayer_txt = f"{DIAS_ES[ayer_dt.weekday()]} {ayer_dt.day} de {MESES_ES[ayer_dt.month - 1]}"
        instruccion_especifica = f"""
PRIORIDADES DE ESTA SECCIÓN
- Las audiencias de televisión se publican SIEMPRE al día siguiente del día
  que describen: una publicación fechada hoy ({hoy_txt}) cuenta las
  audiencias de ayer ({ayer_txt}).
- Busca entre los artículos uno sobre audiencias cuya fecha de publicación
  sea literalmente hoy ({hoy_txt}). Si existe, abre el segmento con esos
  datos, antes que con cualquier otra noticia de televisión, y puedes
  llamarlos "de ayer".
- Si NINGÚN artículo de audiencias está publicado hoy ({hoy_txt}) —por
  ejemplo porque el más reciente es de hace varios días—, OMITE POR COMPLETO
  el ángulo de audiencias: no lo menciones ni como apertura ni como noticia
  secundaria, y monta la sección solo con las demás noticias de televisión
  disponibles.
- Cuando sí haya publicación de hoy, cita explícitamente la cifra (share o
  espectadores) de estos programas siempre que la cifra aparezca escrita en
  los resúmenes de arriba: "Y ahora Sonsoles", "YAS Verano", "Directo al
  grano", "Malas lenguas".
- De esa lista, hablas solo de los programas cuya cifra puedas leer
  literalmente en los materiales. Los que no tengan dato disponible se
  quedan fuera del guion.
"""
    elif folder_name.strip().upper() == "CULTURA POP":
        instruccion_especifica = """
PRIORIDADES DE ESTA SECCIÓN
- Variety mezcla en su feed noticias de música con las de cine y
  televisión: quédate solo con las musicales de Variety y descarta las de
  cine o series, aunque parezcan relevantes — esas ya tienen su sitio en la
  sección de POPCORN.
"""

    prompt = f"""Sección del episodio de hoy: "{folder_name}"

MATERIALES DISPONIBLES (título, fuente y resumen). Hay bastantes más de los
que caben en el tiempo asignado, y esa es la idea: sobran a propósito.

{articles_text}

ENCARGO
Escribe el segmento hablado de esta sección.

SELECCIÓN
- Quédate con las {n_noticias} noticias (arriba o abajo, según den de sí) que
  más le importen al oyente por relevancia, no por ser las más recientes.
- Descarta sin miramientos lo menor, lo repetido y lo de relleno: el resto de
  artículos se quedan fuera y no pasa nada.
- Agrupa los temas relacionados y enlázalos entre sí, de forma que el segmento
  se escuche como un relato y no como una lista leída una por una.
- Los enlaces entre noticias ("siguiendo en España", "en la misma línea",
  "por otro lado"...) tienen que ser ciertos: usa "siguiendo en España" solo
  si la noticia que sigue ocurre de verdad en España, y así con cualquier
  otro enlace de lugar o tema. Si dos noticias seguidas no comparten ese
  hilo, engánchalas con una transición neutra o cambia de frase, pero nunca
  con un enlace que contradiga el contenido real.
- Varía los conectores entre noticias: no repitas la misma fórmula
  ("cambiando de tercio" u otra) más de una vez dentro de este segmento.
- Cuando varias noticias hablen de personas (artistas, famosos, deportistas),
  coloca primero a las más conocidas por el público general y deja para el
  final a los perfiles emergentes o menos reconocibles, que además son los
  primeros candidatos a caerse si falta espacio.

EXTENSIÓN
- Alrededor de {word_budget} palabras: entre tres y cinco frases por noticia.
- Si ves que se te va de las manos, quita una noticia entera antes que alargar
  las frases.
- Cierra siempre con una frase terminada. Terminar antes de lo previsto con
  una idea completa es mejor resultado que llegar al límite a mitad de una.

ARRANQUE Y CIERRE
- La transición que suena justo antes ya ha anunciado la sección y ya han
  sonado la portada y, según el día, otras secciones: tu primera palabra
  forma parte ya de la primera noticia, sin saludos, sin presentaciones y sin
  nombrar la sección.
- Termina también en la última noticia, sin despedida.

FUENTES
- Cada noticia lleva su fuente dicha dentro de la frase, de forma natural:
  "según publica El País...", "The Verge cuenta que...". No queda ninguna
  noticia sin decir de dónde sale.
{instruccion_especifica}
EJEMPLO DE TONO Y FORMATO
El contenido de este ejemplo es inventado y no tiene ninguna relación con las
noticias de hoy; fíjate únicamente en la forma:

{EJEMPLO_SEGMENTO}

Responde solo con el texto del segmento."""

    text, truncated = generate_script(
        client, claude_client, prompt,
        system_instruction=SYSTEM_LOCUTOR,
        max_output_tokens=_tope_tokens(word_budget),
    )
    # Se recorta SIEMPRE a la última frase completa, no solo cuando la API
    # marca `truncated`: esa marca puede no llegar aunque el texto quede
    # incompleto, y recortar_a_frase_completa() es un no-op sobre un texto
    # que ya termina bien, así que no hay coste en aplicarlo siempre.
    text_recortado = recortar_a_frase_completa(text)
    if text_recortado != text:
        motivo = "se quedó sin espacio de tokens" if truncated else "terminó a media frase"
        print(f"  [aviso] el segmento '{folder_name}' {motivo}, se recorta a la última frase completa")
    text = text_recortado
    return normalize_for_tts(text)


# Contrato de la respuesta: con la salida estructurada nativa el JSON llega
# válido por construcción, así que ni hace falta pedirlo con palabras ni
# rescatarlo con una expresión regular.
if types is not None:
    ESQUEMA_META = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "titular": types.Schema(
                type=types.Type.STRING,
                description="Titular llamativo y breve, de seis a diez palabras, sobre lo más interesante del episodio.",
            ),
            "descripcion": types.Schema(
                type=types.Type.STRING,
                description="Resumen de lo más llamativo del episodio, máximo tres frases, sin markdown.",
            ),
        },
        required=["titular", "descripcion"],
    )
else:
    ESQUEMA_META = None


def build_title_and_description(date_str: str, guion_final: str, client, claude_client=None) -> dict:
    prompt = f"""Este es el guion COMPLETO y DEFINITIVO del episodio de hoy ({date_str})
de un podcast de resumen de noticias — es exactamente lo que se va a leer en
voz alta, palabra por palabra:

---
{guion_final}
---

ENCARGO
Escribe el titular y la descripción de este episodio:
- "titular": llamativo y breve, de seis a diez palabras, sobre lo más
  interesante de lo que se cuenta hoy.
- "descripcion": lo más llamativo del episodio en un máximo de tres frases.

REGLA MÁS IMPORTANTE: el titular y la descripción se construyen únicamente con
cosas que aparezcan literalmente en el guion de arriba. Cualquier noticia,
dato o nombre que no esté en ese texto se queda fuera, por relevante que te
parezca o por bien que lo conozcas por otra vía. Ante la duda de si algo se
cuenta o no en el guion, se queda fuera.

Devuelve ÚNICAMENTE un JSON válido (sin texto adicional antes ni después, sin
markdown, sin bloques de código) con esta forma exacta:
{{"titular": "...", "descripcion": "..."}}"""

    try:
        raw, _ = generate_script(client, claude_client, prompt, system_instruction=SYSTEM_META,
                                 max_output_tokens=600, response_schema=ESQUEMA_META)
        data = json.loads(raw)
        if not isinstance(data, dict) or not data.get("titular"):
            raise ValueError("respuesta sin titular")
        return data
    except Exception as e:
        # Antes este fallo degradaba en silencio; ahora al menos queda en el log.
        print(f"  [aviso] no se pudo generar título/descripción ({e}); se usa el texto por defecto")
        return {"titular": "Resumen del día", "descripcion": "Resumen de noticias del día."}


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


def _loudnorm_pass(in_path: Path, out_path: Path, target_lufs: float, true_peak: float):
    """Una única pasada de ffmpeg `loudnorm` (medición y corrección en el
    mismo paso, sin dos pasadas): de sobra para un pipeline desatendido que
    ya se ajusta a oído, número a número, tras cada episodio real."""
    cmd = ["ffmpeg", "-y", "-i", str(in_path), "-af",
           f"loudnorm=I={target_lufs}:TP={true_peak}:LRA=7",
           "-c:a", "libmp3lame", "-b:a", "96k", str(out_path)]
    subprocess.run(cmd, check=True, capture_output=True)


def normalize_voice_audio(in_path: Path, out_path: Path):
    """Normaliza una locución (intro o sección) a VOICE_REF_LUFS antes de
    que llegue a mix_section_audio()/concatenate_segments(): da a todos los
    tramos hablados del episodio el mismo suelo de volumen, aunque Fish
    Audio los devuelva con niveles naturales distintos."""
    _loudnorm_pass(in_path, out_path, VOICE_REF_LUFS, true_peak=-1.5)


def normalize_jingle_audio(in_path: Path, out_path: Path):
    """Normaliza cabecera/cierre a MUSIC_REF_LUFS, el mismo nivel al que se
    normaliza el golpe de entrada de la música de cada sección: cabecera y
    cierre dejan de ser una excepción más alta que el resto del episodio."""
    _loudnorm_pass(in_path, out_path, MUSIC_REF_LUFS, true_peak=-2.0)


def concatenate_segments(segment_paths: list, out_path: Path):
    inputs = []
    for p in segment_paths:
        inputs += ["-i", str(p)]
    n = len(segment_paths)
    # Cada segmento pasa por aformat antes del concat: los segmentos hablados
    # (Fish Audio) y los mp3 de música/jingles suministrados aparte pueden
    # venir con sample rate o número de canales distintos, y el filtro
    # `concat` exige que todas las entradas compartan formato.
    normalize = "".join(
        f"[{i}:a:0]aformat=sample_rates=44100:channel_layouts=stereo[a{i}];" for i in range(n)
    )
    concat_inputs = "".join(f"[a{i}]" for i in range(n))
    filter_complex = normalize + f"{concat_inputs}concat=n={n}:v=0:a=1[out]"
    cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", filter_complex,
           "-map", "[out]", "-c:a", "libmp3lame", "-b:a", "96k", str(out_path)]
    subprocess.run(cmd, check=True, capture_output=True)


def db_to_amplitude(db: float) -> float:
    return 10 ** (db / 20)


def build_music_envelope_expr(voice_duration: float) -> str:
    """Expresión ffmpeg (filtro `volume`, eval=frame) que dibuja la
    envolvente de la música de fondo de una sección, en amplitud RELATIVA al
    nivel ya normalizado por `loudnorm` (1.0 = MUSIC_REF_LUFS, el volumen del
    golpe de entrada): entra YA sonando al nivel de fondo (nunca en silencio
    absoluto, para no dejar un hueco muerto justo tras la sección anterior),
    sube rápido al golpe de entrada, y baja al nivel de fondo de forma
    progresiva empezando un poco antes de que arranque la voz y terminando
    un poco después (se solapan, en vez de cortar en seco antes de la
    locución), se mantiene baja mientras se habla, y se desvanece a silencio
    en una cola corta al terminar la sección."""
    fade_in_end = MUSIC_FADE_IN_SEC
    duck_end = MUSIC_PRE_ROLL_SEC + MUSIC_DUCK_OVERLAP_SEC
    duck_start = duck_end - MUSIC_DUCK_FADE_SEC
    voice_end = MUSIC_PRE_ROLL_SEC + voice_duration
    tail_end = voice_end + MUSIC_TAIL_SEC
    bg = db_to_amplitude(-MUSIC_DUCK_DB)
    return (
        f"if(lt(t,{fade_in_end}),{bg}+(t/{fade_in_end})*(1.0-{bg}),"
        f"if(lt(t,{duck_start}),1.0,"
        f"if(lt(t,{duck_end}),1.0-(t-{duck_start})/{MUSIC_DUCK_FADE_SEC}*(1.0-{bg}),"
        f"if(lt(t,{voice_end}),{bg},"
        f"if(lt(t,{tail_end}),{bg}*(1-(t-{voice_end})/{MUSIC_TAIL_SEC}),0)))))"
    )


def mix_section_audio(voice_path: Path, music_path: Path, out_path: Path):
    """Mezcla la voz de una sección con su música de fondo: pre-roll musical
    antes de la voz, música baja bajo la locución, fade out al terminar."""
    voice_duration = get_duration_seconds(voice_path)
    total_len = MUSIC_PRE_ROLL_SEC + voice_duration + MUSIC_TAIL_SEC
    envelope = build_music_envelope_expr(voice_duration)
    pre_roll_ms = int(round(MUSIC_PRE_ROLL_SEC * 1000))

    filter_complex = (
        # `aloop` opera sobre las muestras ya decodificadas (bucle continuo,
        # sin reabrir el contenedor mp3 en cada vuelta) y `acompressor` nivela
        # los pasajes internos más flojos de la propia música ANTES de
        # normalizar/atenuar, para que no se noten como microsilencios al
        # repetirse en cada vuelta del loop.
        f"[1:a]aloop=loop=-1:size=2147483647,atrim=0:{total_len},asetpts=PTS-STARTPTS,"
        f"aformat=sample_rates=44100:channel_layouts=stereo,"
        f"{MUSIC_COMPRESSOR},"
        f"loudnorm=I={MUSIC_REF_LUFS}:TP=-2:LRA=7,"
        f"volume=eval=frame:volume='{envelope}'[music];"
        f"[0:a]aformat=sample_rates=44100:channel_layouts=stereo,"
        f"adelay={pre_roll_ms}|{pre_roll_ms}[voice];"
        f"[music][voice]amix=inputs=2:duration=longest:normalize=0[out]"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", str(voice_path),
        "-i", str(music_path),
        "-filter_complex", filter_complex,
        "-map", "[out]", "-t", str(total_len),
        "-c:a", "libmp3lame", "-b:a", "96k", str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def crossfade_chain(paths: list, out_path: Path, crossfade_sec: float = MUSIC_CROSSFADE_SEC):
    """Encadena varios mp3 ya mezclados (voz + música) solapando ligeramente
    la música de sus bordes (pre-roll/tail, sin voz) con `acrossfade`, en vez
    de pegarlos secos con `concat`."""
    inputs = []
    for p in paths:
        inputs += ["-i", str(p)]

    fmt_parts = [
        f"[{i}:a]aformat=sample_rates=44100:channel_layouts=stereo[f{i}]" for i in range(len(paths))
    ]
    if len(paths) == 1:
        filter_complex = "[0:a]aformat=sample_rates=44100:channel_layouts=stereo[out]"
    else:
        xfade_parts = []
        prev_label = "f0"
        for i in range(1, len(paths)):
            out_label = f"xf{i}" if i < len(paths) - 1 else "out"
            xfade_parts.append(
                f"[{prev_label}][f{i}]acrossfade=d={crossfade_sec}:c1=tri:c2=tri[{out_label}]"
            )
            prev_label = out_label
        filter_complex = ";".join(fmt_parts + xfade_parts)

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
    parser.add_argument("--music-dir", default="assets/music",
                         help="Carpeta con la cabecera, el cierre y la música de fondo de cada sección")
    parser.add_argument("--target-minutes", type=float, default=15.0,
                         help="Duración objetivo del episodio en minutos")
    parser.add_argument("--folders", nargs="*", default=None)
    args = parser.parse_args()

    if genai is None or types is None:
        sys.exit("Falta instalar el SDK: pip install google-genai")
    for var in ("GEMINI_API_KEY", "FISH_API_KEY"):
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

    music_dir = Path(args.music_dir)
    required_music_files = [CABECERA_FILENAME, CIERRE_FILENAME, *SECTION_MUSIC.values()]
    missing_music = [f for f in required_music_files if not (music_dir / f).exists()]
    if missing_music:
        sys.exit(f"Faltan ficheros de música en {music_dir}: {', '.join(missing_music)}")

    episodes_path = docs_dir / "episodes.json"
    episodes = json.loads(episodes_path.read_text()) if episodes_path.exists() else []

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    claude_client = None
    if anthropic is not None and os.environ.get("ANTHROPIC_API_KEY"):
        claude_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    else:
        print("  [aviso] ANTHROPIC_API_KEY no configurada: sin respaldo si Gemini se satura")
    print(f"Modelo de guion: {GEMINI_MODEL} (temperatura {GEMINI_TEMPERATURE})")
    folders = parse_opml_by_folder(args.opml_path)
    if args.folders:
        folders = {k: v for k, v in folders.items() if k in args.folders}
    # Solo se locutan las carpetas con música de fondo asignada (ver
    # SECTION_MUSIC); el resto (p.ej. "MEDIA TECH") queda fuera del episodio.
    folders = {k: v for k, v in folders.items() if k in SECTION_MUSIC}

    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")

    # --- Recopilar entradas por carpeta ---
    folder_entries = {}

    for folder_name, feeds in folders.items():
        print(f"== {folder_name} ({len(feeds)} feeds) ==")
        entries = fetch_recent_entries(feeds, args.hours)
        print(f"  {len(entries)} artículos disponibles en las últimas {args.hours}h")
        if entries:
            folder_entries[folder_name] = entries

    if not folder_entries:
        print("Sin novedades en ninguna carpeta hoy. No se genera episodio.")
        return

    # Orden fijo del episodio (ver SECTION_ORDER), no el orden de inserción
    # (que depende de en qué orden aparecen las carpetas en el OPML).
    folder_entries = {k: folder_entries[k] for k in SECTION_ORDER if k in folder_entries}

    # --- Presupuesto de palabras por carpeta (proporcional, no solo recencia) ---
    WPM = 160
    total_budget = int(args.target_minutes * WPM)
    intro_budget = 60
    remaining = max(total_budget - intro_budget, 300)

    weights = {f: math.sqrt(len(e)) for f, e in folder_entries.items()}
    total_weight = sum(weights.values())
    word_budgets = {}
    for f in folder_entries:
        budget = int(remaining * weights[f] / total_weight)
        word_budgets[f] = max(120, min(budget, 600))

    # --- Efeméride + intro ---
    efemeride = get_efemeride(now)
    print("Generando introducción...")
    intro_text = build_intro_script(now, efemeride, client, claude_client)

    # --- Segmentos por carpeta ---
    segment_texts = {}
    transicion_idx = 0
    primera_seccion = True
    for folder_name, entries in folder_entries.items():
        print(f"Generando segmento: {folder_name} (~{word_budgets[folder_name]} palabras)")
        cuerpo = build_folder_segment(folder_name, entries, client, word_budgets[folder_name],
                                      dt=now, claude_client=claude_client)
        if not cuerpo.strip():
            # Sin cuerpo, la sección sería solo la transición seguida de
            # música: mejor dejarla fuera del episodio que emitir el hueco.
            print(f"  [aviso] segmento vacío para '{folder_name}', se omite la sección")
            continue
        # La transición se decide DESPUÉS de confirmar que la sección tiene
        # contenido: así "Empezamos con..." siempre cae en la sección que de
        # verdad abre el episodio, no en una que luego resulte vacía.
        if primera_seccion:
            transicion = f"Empezamos con las novedades sobre {folder_name}."
            primera_seccion = False
        else:
            transicion = TRANSICIONES[transicion_idx % len(TRANSICIONES)].format(folder=folder_name)
            transicion_idx += 1
        segment_texts[folder_name] = f"{transicion} {cuerpo}"

    if not segment_texts:
        print("No se pudo generar ningún segmento hoy. No se genera episodio.")
        return

    # --- Título y descripción (basados en el guion REAL ya escrito, no en
    # las noticias en bruto antes de seleccionar, para que nunca mencionen
    # algo que luego no se cuenta) ---
    print("Generando título y descripción del episodio...")
    guion_noticias = "\n\n".join(segment_texts.values())
    meta = build_title_and_description(date_str, guion_noticias, client, claude_client)
    episode_title = f"{now.day} de {MESES_ES[now.month - 1]} — {meta.get('titular', 'Resumen del día')}"
    episode_description = meta.get("descripcion", "Resumen de noticias del día.")

    # --- Guardar transcripción interna completa ---
    full_transcript = intro_text + "\n\n" + "\n\n".join(segment_texts.values())
    (transcripts_dir / f"{date_str}.txt").write_text(full_transcript, encoding="utf-8")

    # --- Voz: intro + una locución por sección ---
    print("Generando audio con Fish Audio...")
    intro_path_raw = tmp_dir / "intro_raw.mp3"
    intro_path = tmp_dir / "intro.mp3"
    text_to_speech_fish(intro_text, intro_path_raw, os.environ["FISH_API_KEY"],
                         model=args.fish_model, reference_id=args.fish_reference_id)
    normalize_voice_audio(intro_path_raw, intro_path)

    ordered_sections = list(segment_texts.keys())
    section_mixed_paths = {}
    section_durations_ms = {}
    for folder_name in ordered_sections:
        safe = re.sub(r"[^\w\-]+", "_", folder_name.strip())
        voice_path_raw = tmp_dir / f"voz_{safe}_raw.mp3"
        voice_path = tmp_dir / f"voz_{safe}.mp3"
        text_to_speech_fish(segment_texts[folder_name], voice_path_raw, os.environ["FISH_API_KEY"],
                             model=args.fish_model, reference_id=args.fish_reference_id)
        # Cada sección es una llamada TTS independiente y puede volver con un
        # volumen natural distinto: normalizar aquí, antes de mezclar, es lo
        # que evita que la voz salte de nivel entre secciones.
        normalize_voice_audio(voice_path_raw, voice_path)
        print(f"  Mezclando música de fondo: {folder_name}")
        mixed_path = tmp_dir / f"mix_{safe}.mp3"
        mix_section_audio(voice_path, music_dir / SECTION_MUSIC[folder_name], mixed_path)
        section_mixed_paths[folder_name] = mixed_path
        section_durations_ms[folder_name] = get_duration_seconds(mixed_path) * 1000

    # La cabecera y el cierre se encadenan con crossfade igual que las
    # secciones entre sí: sus bordes son música pura (cabecera no lleva voz,
    # y el borde de la última sección es su cola de fade-out reservada), así
    # que solapar un poco evita el hueco de silencio que deja el corte seco
    # justo cuando la cabecera/el cierre ya están casi en silencio de por sí.
    print("  Encadenando cabecera, secciones y cierre con crossfade musical...")
    cabecera_path = tmp_dir / "cabecera_norm.mp3"
    cierre_path = tmp_dir / "cierre_norm.mp3"
    normalize_jingle_audio(music_dir / CABECERA_FILENAME, cabecera_path)
    normalize_jingle_audio(music_dir / CIERRE_FILENAME, cierre_path)
    chain_paths = [cabecera_path] + [section_mixed_paths[f] for f in ordered_sections] + [cierre_path]
    body_path = tmp_dir / "cuerpo_completo.mp3"
    crossfade_chain(chain_paths, body_path)

    final_mp3 = audio_dir / f"episodio_{date_str}.mp3"
    concatenate_segments([intro_path, body_path], final_mp3)

    # --- Capítulos ---
    # Dentro del tramo con crossfade, cada unión se solapa MUSIC_CROSSFADE_SEC
    # segundos, así que ese solape se resta al offset acumulado para que los
    # capítulos sigan cuadrando con el audio real.
    chapters = []
    crossfade_ms = MUSIC_CROSSFADE_SEC * 1000

    # La cabecera suena pegada al saludo/efeméride y forma parte del mismo
    # bloque de introducción: no es un capítulo aparte en la metadata.
    intro_dur = get_duration_seconds(intro_path) * 1000
    cabecera_dur = get_duration_seconds(cabecera_path) * 1000
    chapters.append({"title": "Introducción", "start_ms": 0.0, "end_ms": intro_dur + cabecera_dur})
    cursor_ms = intro_dur + cabecera_dur

    for folder_name in ordered_sections:
        cursor_ms -= crossfade_ms
        dur = section_durations_ms[folder_name]
        chapters.append({"title": folder_name, "start_ms": cursor_ms, "end_ms": cursor_ms + dur})
        cursor_ms += dur

    cursor_ms -= crossfade_ms
    cierre_dur = get_duration_seconds(cierre_path) * 1000
    chapters.append({"title": "Cierre", "start_ms": cursor_ms, "end_ms": cursor_ms + cierre_dur})

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
