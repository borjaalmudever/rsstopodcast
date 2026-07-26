# Podcast automático de tus carpetas de Reeder

Cada día, sin que tengas que hacer nada, este sistema:
1. Lee tus feeds agrupados por carpeta.
2. Escribe un resumen hablado con Claude.
3. Lo convierte a voz con Fish Audio.
4. Publica un episodio nuevo por carpeta en un podcast privado que escuchas en Apple Podcasts.

Todo corre en GitHub (gratis), no necesitas tener el ordenador encendido.

## Paso 1 — Crear el repositorio

1. Ve a https://github.com/new y crea un repositorio (puede ser privado o público — si es privado, Pages sigue funcionando en cuentas Pro; si tienes cuenta gratis, tiene que ser **público** para usar GitHub Pages gratis).
2. Sube estos archivos a la raíz del repo, manteniendo la carpeta `.github/workflows/`:
   - `resumen_feeds.py`
   - `generar_feed_podcast.py`
   - `iCloud_RSS.opml` (ya tiene tus carpetas actuales)
   - `docs/cover.jpg`
   - `.github/workflows/daily.yml`

   La forma más simple: en la página del repo, "Add file" → "Upload files", arrastra todo manteniendo la estructura de carpetas.

## Paso 2 — Activar GitHub Pages

1. En el repo: **Settings → Pages**.
2. En "Source", elige **Deploy from a branch**.
3. Branch: `main`, carpeta: `/docs`. Guarda.
4. GitHub te dará una URL tipo `https://tu-usuario.github.io/tu-repo/`. Apúntala, la necesitas ahora.

## Paso 3 — Configurar claves y variables

En **Settings → Secrets and variables → Actions**:

**Secrets** (pestaña "Secrets"), botón "New repository secret":
- `ANTHROPIC_API_KEY` → tu clave de https://console.anthropic.com
- `FISH_API_KEY` → tu clave de Fish Audio (Panel → API Keys en fish.audio)

**Variables** (pestaña "Variables"), botón "New repository variable":
- `PODCAST_BASE_URL` → la URL de Pages del paso 2, ej. `https://tu-usuario.github.io/tu-repo`
- `FISH_REFERENCE_ID` → (opcional) el ID de una voz concreta de tu cuenta de Fish Audio.
  Si no tienes una voz clonada y quieres usar una de la biblioteca pública, puedes dejarlo
  vacío (usará la voz por defecto del modelo) o copiar el ID desde fish.audio → Voice Library
  → (voz que te guste) → "Copy ID".

## Paso 4 — Probar manualmente

En el repo: **Actions → "Generar resumen diario de feeds" → Run workflow**. Esto ejecuta todo el
proceso una vez sin esperar al cron. Revisa los logs; al terminar deberían aparecer archivos
nuevos en `docs/audio/` y `docs/podcast.xml` actualizado (verás el commit automático).

## Paso 5 — Suscribirte desde el iPhone

1. Abre la app **Podcasts** de Apple.
2. Pestaña **Biblioteca** → botón "···" (arriba a la derecha) → **Seguir un show por URL**.
3. Pega: `https://tu-usuario.github.io/tu-repo/podcast.xml`
4. Listo — cada mañana te aparecerán episodios nuevos, uno por carpeta con novedades.

## Horario

Por defecto corre a las 06:30 UTC (≈ 8:30 hora de España). Para cambiarlo, edita la línea
`cron:` en `.github/workflows/daily.yml` (formato cron estándar, en UTC).

## Actualizar tus feeds/carpetas

Como Reeder no tiene exportación automática, cuando añadas o quites feeds/carpetas en Reeder:
1. Exporta de nuevo el OPML (Ajustes → Your Data → Export OPML).
2. Sustituye `iCloud_RSS.opml` en el repo (súbelo de nuevo con el mismo nombre).

## Notas

- Solo se genera episodio para carpetas con artículos nuevos ese día — si una carpeta está
  tranquila, simplemente no hay episodio, no te llega uno vacío.
- El resumen usa el título y extracto de cada entrada RSS, no el artículo completo.
- La carátula (`docs/cover.jpg`) es un placeholder simple — puedes sustituirla por tu propia
  imagen cuadrada (mínimo 1400×1400 px) con el mismo nombre de archivo.
- Coste: Claude y Fish Audio se cobran por uso según tu plan en cada plataforma; GitHub
  Actions/Pages son gratis para este volumen de uso.
