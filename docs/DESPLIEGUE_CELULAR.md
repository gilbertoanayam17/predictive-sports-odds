# Usar la app desde el celular (Streamlit Community Cloud, gratis)

Streamlit Community Cloud ejecuta la app en sus servidores, a partir de tu repositorio de GitHub. Así puedes abrirla desde el navegador del celular sin tener tu computadora encendida.

> Los nombres de los menús de Streamlit pueden cambiar con el tiempo; si alguno no coincide, busca la opción equivalente.

## 1. Antes de publicar
- El código debe estar en la rama que vas a publicar. Usa `claude/ecstatic-wright-6nlr98`, o `main` después de fusionar el pull request.
- **Nunca subas tu `.env`** (ya está en `.gitignore`). En la nube las claves van como *secrets*.

## 2. Crear la app
1. Entra en [share.streamlit.io](https://share.streamlit.io) e inicia sesión con tu cuenta de GitHub. Autoriza el acceso al repositorio `predictive-sports-odds`.
2. **Create app** → *Deploy a public app from GitHub* (o la opción equivalente):
   - Repositorio: `gilbertoanayam17/predictive-sports-odds`
   - Rama: `claude/ecstatic-wright-6nlr98` (o `main`)
   - Archivo principal: `app.py`
3. En **Advanced settings**:
   - **Python version:** 3.12
   - **Secrets:** pega esto con tus valores (formato TOML, con comillas):
     ```toml
     ODDS_API_KEY = "tu_clave_de_the_odds_api"
     # Opcionales:
     # TELEGRAM_BOT_TOKEN = "..."
     # TELEGRAM_CHAT_ID = "..."
     ```
4. **Deploy**. La primera vez tarda unos minutos en instalar las dependencias de `requirements.txt`.

## 3. ⚠ Hazla privada
Si la app es pública, **cualquiera con el enlace gastaría tus créditos de The Odds API**. Tu clave no se envía nunca al navegador (la app la usa solo en el servidor), pero sí puede usarla quien abra la app.

En el panel de la app: **Settings → Sharing** → limita quién puede verla (por ejemplo, solo tu correo). Comprueba el acceso abriendo el enlace en una ventana de incógnito.

## 4. En el celular
- Abre el enlace de la app (`https://<nombre>.streamlit.app`) en el navegador.
- **Añádela a la pantalla de inicio** para abrirla como una app: Chrome → menú ⋮ → *Añadir a pantalla de inicio*; Safari → Compartir → *Añadir a inicio*.
- En pantallas pequeñas, la barra lateral se abre con la flecha **>** de arriba a la izquierda.

## 5. Lo que hay que saber
| Tema | Detalle |
|---|---|
| **Registro de apuestas** | En la nube el disco **se borra** cuando la app se reinicia o se actualiza. En *Registro → Copia de seguridad del registro*, descarga una copia después de registrar o liquidar y restáurala cuando haga falta. Otra opción: usa el registro solo en tu computadora y la nube solo para consultar. |
| **Perfiles del estudio por liga** | `league_profiles.json` tampoco persiste en la nube. Ejecuta el estudio en tu computadora y súbelo al repositorio (no contiene datos privados), o repítelo en la nube cuando haga falta. |
| **Suspensión** | Las apps gratuitas se "duermen" tras un tiempo sin uso; al abrirlas tardan unos segundos en despertar. |
| **Actualizaciones** | Cada `git push` a la rama publicada actualiza la app automáticamente. |
| **Créditos** | Cada análisis gasta los mismos créditos que en local (~2 por liga). La barra lateral muestra los restantes. |
| **Alertas de Telegram** | Siguen funcionando desde tu computadora con la tarea programada; la nube no ejecuta tareas programadas. |
