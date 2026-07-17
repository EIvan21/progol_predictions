# ⚽ Progol Prediction Engine

Pipeline autónomo de Machine Learning que pronostica los resultados del **Progol**
(quiniela deportiva de México: 21 partidos por concurso — 14 principales + 7 de
revancha). Produce probabilidades **L/E/V** por partido, arma **quinielas
optimizadas por presupuesto**, genera **mapas de calor de marcador exacto**, y
publica todo en un **bot de Telegram**. Corre solo cada semana en Google Cloud.

---

## 📑 Tabla de contenido
1. [Arquitectura](#1-arquitectura)
2. [Los modelos](#2-los-modelos)
3. [Ingeniería de variables](#3-ingeniería-de-variables)
4. [Cálculo del Elo (a detalle)](#4-cálculo-del-elo-a-detalle)
5. [Datos: fuentes y separación](#5-datos-fuentes-y-separación)
6. [Entrenamiento](#6-entrenamiento)
7. [Reentrenamiento y seguimiento](#7-reentrenamiento-y-seguimiento)
8. [Modelo de marcador y mapas de calor](#8-modelo-de-marcador-y-mapas-de-calor)
9. [Optimizador de quiniela](#9-optimizador-de-quiniela)
10. [Bot de Telegram](#10-bot-de-telegram)
11. [Infraestructura y automatización](#11-infraestructura-y-automatización)
12. [Estructura del proyecto](#12-estructura-del-proyecto)
13. [Setup y ejecución](#13-setup-y-ejecución)

---

## 1. Arquitectura

```
fetch_data → preprocess → tune → train → walk_forward → backtest
                                            ↓
get_progol_ids → predict → predictions/latest.json + filas en la BD
                                            ↓
              bot/send_predictions → Telegram      reporting/score_report → mapas de marcador
              bot/app (long-polling: /predecir, /presupuesto, /marcadores, /historial…)
```

Todo el pipeline corre en una **VM de GCE (`progol-trainer`)** al arrancar: jala
el repo, corre cada paso, sube artefactos a GCS y se auto-apaga. La **VM del bot
(`progol-bot`)** corre long-polling y sincroniza desde GCS bajo demanda.

---

## 2. Los modelos

Son **tres sistemas distintos**:

### 🅰️ Modelo 1X2 — el principal (Machine Learning)
Clasificación de 3 clases (**L**ocal / **E**mpate / **V**isita).

- **Ensemble apilado** (`StackingClassifier`):
  - **Modelos base:** **XGBoost + CatBoost + Random Forest** (300 estimadores c/u)
  - **Meta-learner:** **Regresión Logística** que combina las 3 salidas
- **Calibración Dirichlet** (matriz, Kull et al. 2019) sobre un holdout → corrige
  el sesgo por clase que una temperatura escalar no puede.
- **Blend con el mercado:** mezcla la probabilidad del modelo con la implícita de
  los momios, con **peso por liga** (`config.MODEL_MARKET_BLEND_BY_LEAGUE`):
  ligas con mercado afilado (top-5 europeas) → más peso al mercado; ligas opacas
  (Liga MX Expansión, Rusia) → más peso al modelo.
- **Pesos de muestra:** decaimiento temporal (partidos recientes pesan más) +
  pesos por clase (balancea L/E/V).

> Código: `src/progol/modeling/train.py`, `tune.py`, `predict.py`.

### 🅱️ Modelo de marcador exacto (mixto)
Modelo generativo de goles → **matriz de marcador 7×7**. Dos backends:

1. **Dixon-Coles** — estadístico (Poisson por máxima verosimilitud). *No es ML.*
2. **XGBoost** — ML real: dos regresores `count:poisson` predicen los goles
   esperados de cada equipo desde las fuerzas Dixon-Coles + forma reciente.
3. **Meta-ensemble** — promedia ambas matrices (gana en el holdout temporal).

> Código: `src/progol/modeling/score_model.py`, `reporting/score_report.py`.

### 🅲️ Optimizador de quiniela (combinatorio, no ML)
**Monte Carlo** + **optimizador por presupuesto**: elige qué partidos jugar como
doble/triple para maximizar la probabilidad de cobertura dentro del presupuesto.
Costo = `base × 2^dobles × 3^triples`.

> Código: `src/progol/modeling/quiniela.py`.

---

## 3. Ingeniería de variables

El modelo 1X2 **no usa datos crudos** — usa **features diferenciales**
(Local − Visita) para eliminar el sesgo de escala. Definidas en
`config.FEATURE_COLS`:

| Feature | Qué captura |
|---|---|
| `xg_diff` | goles esperados (xG real, EWMA span=5) |
| `elo_diff` | rating Elo con ventaja de local (ver §4) |
| `gf_ewma_diff` / `ga_ewma_diff` | goles a favor / en contra recientes |
| `sf_ewma_diff` | tiros a favor |
| `sos_gf_diff` | fuerza del calendario (strength of schedule) |
| `momentum_diff` | forma reciente (puntos) |
| `rank_gap` | diferencia de posición en la tabla |
| `h2h_diff` | historial directo (últimos enfrentamientos) |
| `rest_diff` | días de descanso |
| `injuries_diff` | lesionados (Local − Visita), del endpoint /injuries |
| `total_goals_avg`, `draw_rate_avg` | señales "propenso al empate" (promedios, no diffs) |
| `is_cup` | 1 si es copa/knock-out (`config.CUP_LEAGUE_IDS`) |
| `is_artificial` | cancha artificial |

**Categóricas** (`config.CAT_COLS`): `home_id`, `referee`, `league_id`,
codificadas con **TargetEncoder**.

> Los momios **no** se entrenan como feature (solo ~22 de 74k filas históricas
> tienen momios); se aplican como blend en inferencia.
>
> Código: `src/progol/modeling/preprocess.py::calculate_alpha_features`.

---

## 4. Cálculo del Elo (a detalle)

El Elo es un rating dinámico de fuerza por equipo. Vive en
`src/progol/features/elo.py`.

**Constantes:**
```python
K_FACTOR   = 20     # cuánto se ajusta el rating por partido
BASE_RATING = 1500  # rating inicial
HOME_ADV   = 75     # puntos Elo que se suman al local
```

**Paso 1 — Resultado esperado del local** (curva logística estándar de Elo, con
la ventaja de local sumada al rating del local):
```
E_home = 1 / (1 + 10^((R_away − (R_home + HOME_ADV)) / 400))
```
- Si local y visita empatan en rating, `HOME_ADV=75` da al local ~60% de
  expectativa (la ventaja de jugar en casa).
- La escala `/400` es la convención Elo: 400 puntos de diferencia ≈ 10× más
  probable ganar.

**Paso 2 — Resultado real** (con empates a mitad de camino):
```
result = 1.0  si gana el local
       = 0.5  si empatan
       = 0.0  si gana el visitante
```

**Paso 3 — Actualización** (después del partido):
```
R_home_nuevo = R_home + K · (result − E_home)
R_away_nuevo = R_away + K · ((1 − result) − (1 − E_home))
```
- Si el local gana **contra pronóstico** (E_home bajo), sube mucho; si gana como
  favorito (E_home alto), sube poco. El sistema aprende de las sorpresas.
- Es de **suma cero**: lo que gana un equipo lo pierde el otro.

**Detalles clave del diseño:**

1. **Sin fuga de información (no leakage):** se recorre la BD en **orden
   cronológico**; la feature de cada partido es el rating **ANTES** de jugarlo, y
   la actualización ocurre **después**. Nunca ve su propio resultado.

2. **Cold-start inteligente por liga:** un equipo nuevo (recién ascendido, o la
   primera vez que aparece) **no** arranca en 1500 global, sino en la **media Elo
   de su liga**. Un ascendido a Liga MX empieza en ~1450 (la media de Liga MX),
   un prior mucho mejor que 1500. Ver `_league_mean()`.

3. **El rating se arrastra entre ligas:** si un equipo cambia de liga (ascenso/
   descenso, o juega copa continental), **conserva su Elo ganado** en vez de
   resetearse. Esto permite que el `is_cup` combine equipos de distinto nivel de
   forma coherente.

4. **Salida:** `elo_home` y `elo_away` por partido (→ `elo_diff` = feature), más
   un `history_df` con el rating de cada equipo en el tiempo.

**En inferencia:** `team_state.compute_elo_table(conn, before_date)` reconstruye
la tabla de Elo con todos los partidos **anteriores** a la fecha del fixture a
predecir — mismo cálculo, garantizando consistencia entrenamiento↔inferencia.

---

## 5. Datos: fuentes y separación

| Uso | Fuente | Volumen |
|---|---|---|
| **1X2 (entrenamiento)** | API-Football → `data/progol.db` (SQLite) | ~74,000 partidos, ~37 ligas de clubes, temporadas 2019-2026 |
| **Marcador (selecciones)** | `martj42/international_results` (CSV público) | ~49,000 partidos (1872-2026, se usa desde 2014) |
| **Marcador (clubes)** | `data/progol.db` (vía tabla `teams`) | mismo que 1X2 |
| **Resolver la quiniela** | scrape de `quinielaposible.com` + fuzzy-match a IDs de API | 21 partidos/concurso |

**Separación / validación (respeta el tiempo, nunca aleatorio):**
- **Split temporal** para el holdout de calibración.
- **Walk-forward validation** (`walk_forward.py`, 6 folds): entrena en pasado,
  valida en el futuro inmediato, y avanza la ventana.
- **Backtest económico** (`backtest.py`): simula apuestas con Kelly + ROI.

**Agregar una liga nueva al Progol requiere 3 cosas** (ver gotcha #2 en CLAUDE.md):
(a) apodo en `NICKNAME_MAP`, (b) league ID en `get_progol_ids.py`, y
(c) league ID en `fetch_data.py::LEAGUES` para tener datos de entrenamiento.

---

## 6. Entrenamiento

Pipeline, en orden (`run_pipeline.py` local, o `infra/startup.sh` en la VM):

```
fetch_data    → baja partidos de la API a progol.db (fetcher paralelo)
preprocess    → construye las features diferenciales → final_train_data.csv
tune          → Optuna busca hiperparámetros (30 trials, timeout 1200s)
train         → entrena el ensemble apilado + calibración Dirichlet
walk_forward  → valida robustez (6 folds temporales)
backtest      → prueba económica (Kelly 0.25, min-edge 0.04)
```

Guarda un **bundle versionado**: `models/v_YYYYMMDD_HHMMSS/calibrated_ensemble.pkl`
+ `metrics.json`, `feature_stats.json`, etc. `models/latest.json` apunta al actual.

---

## 7. Reentrenamiento y seguimiento

- **Cada miércoles** la VM **reentrena desde cero** con datos frescos (fetch →
  preprocess → train). El modelo incorpora los resultados de la semana.
- El **modelo de marcador** se reajusta en cada corrida (Dixon-Coles es
  instantáneo; el de clubes se **cachea semanal** en `models/score_club.pkl` por
  ser lento — ~600 equipos).
- **Seguimiento de precisión:** `reporting/score_eval.py` registra el acierto 1X2
  + log-loss de cada backend (DC / XGBoost / blend) en
  `reports/score_model_history.csv`, semana a semana → para ver si mejora.
- **Settlement:** `predict.py` llama a `database.settle_concurso_actuals()` para
  rellenar los resultados reales; `/historial` muestra el hit-rate por concurso.

---

## 8. Modelo de marcador y mapas de calor

Para cada partido se genera una figura de **3 paneles**:

1. **Heatmap** — probabilidad de cada marcador exacto (0-0, 1-0, 2-1…) como
   matriz de Poisson 7×7.
2. **Barra 1X2** — gana Local / Empate / gana Visita (derivado del heatmap).
3. **Top-10 marcadores** más probables.

**Cómo funciona (usa el modelo de marcador, NO el ensemble 1X2):**
- **Dixon-Coles** estima fuerza de **ataque** y **defensa** de cada equipo (+
  ventaja de local + corrección `rho` para marcadores bajos) por máxima
  verosimilitud, con decaimiento temporal (vida media 730 días).
- **XGBoost** predice los goles esperados λ de cada equipo desde esas fuerzas +
  forma reciente.
- Con λ se arma una **distribución de Poisson** de goles → la matriz.
- Se renderiza **Dixon-Coles vs XGBoost lado a lado** (`--backend both`) o el
  **blend** (`--backend meta`).
- Maneja **cancha neutral** (Mundial) salvo anfitriones (México/USA/Canadá).
- Salida: un PNG por partido + un **PDF combinado**, y opcionalmente a Telegram.

> Selecciones → datos de martj42. Clubes → `progol.db`.
> `python -m src.progol.reporting.score_report [--telegram] [--backend dc|xgb|both|meta|all]`

---

## 9. Optimizador de quiniela

`quiniela.optimize_budget(probs, budget)` recibe las probabilidades L/E/V de los
14 partidos principales y decide **qué jugar** para maximizar la cobertura:
- **Sencillo:** el pick más probable.
- **Doble/Triple:** abre los partidos más inciertos (donde subir a 2 o 3 signos
  da más ganancia de cobertura por peso gastado).
- Costo = `base × 2^dobles × 3^triples`; respeta el presupuesto.

Los partidos sin resolver usan un prior neutral (0.45/0.25/0.30) y tienden a salir
como triples. En el bot: `/presupuesto`.

---

## 10. Bot de Telegram

Bot `@gol_pro_bot` (`src/progol/bot/app.py`, long-polling).

**Predicciones**
| Comando | Qué hace |
|---|---|
| `/predecir_progol` | predicción del concurso actual (14 + revancha) |
| `/predecir_partido A vs B` | predice un partido específico o cualquier fixture en 7 días |
| `/ultima_prediccion_progol` | la última predicción guardada |

**Análisis**
| Comando | Qué hace |
|---|---|
| `/presupuesto` | plan óptimo de dobles/triples para tu presupuesto (conversación) |
| `/marcadores` | los mapas de marcador exacto (heatmaps) del concurso |
| `/historial` | aciertos/14 + revancha de los últimos 8 concursos |

**Cuenta / admin**
| Comando | Qué hace |
|---|---|
| `/whoami`, `/help`, `/cancelar` | utilidades |
| `/usuarios`, `/aprobar CHAT_ID`, `/bloquear CHAT_ID` | solo admin |

> El bot es **slim** (sin sklearn/xgboost): no genera nada pesado, solo sincroniza
> de GCS y responde. Los mapas los genera la VM y el bot los reenvía.

---

## 11. Infraestructura y automatización

```
Cloud Scheduler (America/Mexico_City)
  ├─ progol-weekly-stop  (mié 6:55)  → detiene la VM (anti-cuelgue)
  └─ progol-weekly       (mié 7:07)  → arranca la VM
        ↓
VM progol-trainer (e2-standard-4, us-central1-a)
   startup.sh: git reset --hard → fetch→train→predict→score_maps→Telegram
            → sube a GCS → se auto-apaga
        ↓
GCS gs://progol-data-storage   (db, models, predictions, reports, secrets, logs)
        ↓
VM progol-bot (e2-micro, siempre encendida)
   systemd long-polling → sincroniza de GCS → responde comandos
```

**Notas operativas:**
- El **startup-script vive en los metadatos de la VM**. Cambios a
  `infra/startup.sh` requieren `gcloud compute instances add-metadata … --metadata-from-file startup-script=…` para desplegar.
- El `.env` se baja de `gs://…/secrets/.env`. Se le quita CRLF antes de sourcear
  (venía de Windows y contaminaba las llaves con `\r`).
- **Dependencia externa:** la API-Football debe estar en **plan de pago** (Free
  no da fixtures de la temporada en curso).

---

## 12. Estructura del proyecto

```
progol_predictions/
├── run_pipeline.py                 # orquestador local (interactivo)
├── src/progol/
│   ├── config.py                   # rutas, ligas, features, blend por liga
│   ├── database.py                 # esquema SQLite + helpers
│   ├── ingest/
│   │   ├── fetch_data.py           # fetcher API-Football (paralelo)
│   │   ├── get_progol_ids.py       # scrape + resolución de la quiniela
│   │   └── international_results.py # dataset martj42 (selecciones)
│   ├── features/
│   │   ├── elo.py                  # rating Elo (ver §4)
│   │   └── team_state.py           # features en inferencia (espeja el train)
│   ├── modeling/
│   │   ├── preprocess.py           # ingeniería de variables
│   │   ├── tune.py                 # Optuna
│   │   ├── train.py                # ensemble + Dirichlet
│   │   ├── walk_forward.py         # validación temporal
│   │   ├── backtest.py             # ROI / Kelly
│   │   ├── predict.py              # inferencia + settlement
│   │   ├── score_model.py          # Dixon-Coles + XGBoost (marcador)
│   │   └── quiniela.py             # Monte Carlo + presupuesto
│   ├── reporting/
│   │   ├── eda.py, league_dashboard.py, progol_history.py
│   │   ├── score_report.py         # mapas de marcador (3 paneles)
│   │   └── score_eval.py           # seguimiento de precisión del marcador
│   └── bot/
│       ├── app.py                  # long-polling + comandos
│       ├── send_predictions.py     # push semanal
│       └── formatting.py           # render HTML
├── infra/
│   ├── startup.sh                  # bootstrap del trainer (pipeline + shutdown)
│   └── bot_startup.sh              # bootstrap del bot
├── tests/                          # pytest (114 tests)
├── data/    models/    reports/    predictions/   # gitignored (viven en GCS)
```

---

## 13. Setup y ejecución

**Requisitos:** Python **3.11**, una API key **de pago** de
[api-football.com](https://www.api-football.com/), y (para la nube) `gcloud`.

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

**`.env`** en la raíz:
```
FOOTBALL_API_KEY=tu_key
TELEGRAM_BOT_TOKEN=tu_token
TELEGRAM_CHAT_ID=tu_chat_id
GCS_BUCKET=progol-data-storage
```

**Correr:**
```bash
python run_pipeline.py                          # pipeline local (interactivo)
python -m src.progol.ingest.get_progol_ids      # solo resolver la quiniela
python -m src.progol.modeling.predict           # solo inferencia
python -m src.progol.reporting.score_report     # mapas de marcador
pytest                                          # tests
```

> Sincronizar estado desde GCS:
> `gcloud storage rsync -r gs://progol-data-storage/db data/`

---

## ⚖️ Aviso
Proyecto de uso académico y analítico privado. Apostar implica riesgo; usa las
predicciones con responsabilidad.
