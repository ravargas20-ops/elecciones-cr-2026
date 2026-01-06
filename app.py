import os
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from openai import OpenAI, RateLimitError, AuthenticationError, OpenAIError
from rag import load_documents, build_index, retrieve, format_sources

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Comparador Ciudadano CR 2026",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# UI Styles (polish)
# -----------------------------
st.markdown("""
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1200px; }
h1 { font-size: 1.85rem; margin-bottom: 0.25rem; }
h2 { font-size: 1.25rem; margin-top: 0.9rem; }
h3 { font-size: 1.1rem; margin-top: 0.6rem; }
p, li { font-size: 1rem; line-height: 1.45; }

.small-note { opacity: 0.78; font-size: 0.92rem; }
.hr { margin: 1rem 0; border-bottom: 1px solid rgba(255,255,255,0.08); }

.card {
  padding: 1rem 1rem;
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 14px;
  background: rgba(255,255,255,0.03);
}

.card-title { font-weight: 800; margin-bottom: 0.2rem; }

.badge {
  display: inline-block;
  padding: 0.22rem 0.6rem;
  border-radius: 999px;
  background: rgba(99,102,241,0.18);
  border: 1px solid rgba(99,102,241,0.35);
  font-size: 0.85rem;
  margin-right: 0.4rem;
}

.source {
  opacity: 0.85;
  font-size: 0.92rem;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.title("🇨🇷 Comparador Ciudadano — Elecciones Costa Rica 2026")
st.caption("Herramienta neutral para explorar planes de gobierno e Informe Estado de la Nación. Sin recomendaciones de voto.")

with st.expander("⚠️ Aviso importante (léelo en 10 segundos)", expanded=True):
    st.markdown("""
- Esta herramienta **NO recomienda por quién votar**.
- Solo muestra información basada en **documentos oficiales** (planes e Informe Estado de la Nación).
- Cuando responde, siempre incluye **Fuentes** (partido, documento y páginas).
    """.strip())

with st.expander("✅ Cómo usar (30 segundos)", expanded=False):
    st.markdown("""
1) Ve a **Dashboard** para ver prioridades por tema dentro de un partido.  
2) Ve a **Comparar por tema** para comparar 2–3 partidos en un tema (ej. empleo).  
3) Ve a **Preguntar** para hacer una pregunta directa y ver evidencia (con páginas).  
    """.strip())

DATA_PATH = "data/documentos_rag.csv"

# -----------------------------
# Load data + index
# -----------------------------
@st.cache_data
def load_data():
    df = load_documents(DATA_PATH)

    for col in ["partido", "doc_id", "fuente", "texto_chunk"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    if "fuente" not in df.columns:
        df["fuente"] = "Plan de Gobierno"

    return df

@st.cache_resource
def load_index(df):
    return build_index(df)

df = load_data()
# build_index must return: (model, index, embeddings)
model, index, embeddings = load_index(df)

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("⚙️ Filtros (opcional)")
st.sidebar.caption("Úsalos si quieres limitar resultados.")

partidos = sorted([p for p in df["partido"].unique().tolist() if p])
fuentes = sorted([f for f in df["fuente"].unique().tolist() if f])

sidebar_partido = st.sidebar.selectbox("Partido", ["Todos"] + partidos, index=0, key="sidebar_partido")
sidebar_fuente = st.sidebar.selectbox("Fuente", ["Todas"] + fuentes, index=0, key="sidebar_fuente")

k_sel = st.sidebar.slider("Extractos a mostrar", 3, 10, 6, 1, key="sidebar_k")
threshold_default = st.sidebar.slider(
    "Precisión de búsqueda (más alto = más estricto)",
    0.20, 0.60, 0.35, 0.01,
    key="sidebar_threshold"
)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: dentro de la app hay botones de temas para preguntas rápidas.")

# -----------------------------
# Helpers
# -----------------------------
def filter_rows(rows, partido=None, fuente=None, k=6):
    out = []
    for r in rows:
        if partido and r.get("partido") != partido:
            continue
        if fuente and r.get("fuente") != fuente:
            continue
        out.append(r)
        if len(out) >= k:
            break
    return out

def tema_relevancia(query: str):
    q = model.encode([query], normalize_embeddings=True).astype("float32")
    return embeddings @ q[0]

def show_evidence_cards(rows, max_items=3):
    shown = 0
    for r in rows:
        st.markdown(
            f'<div class="card">'
            f'<span class="badge">{r.get("partido","")}</span>'
            f'<span class="source">{r.get("doc_id","")} · pp. {r.get("pagina_inicio","")}-{r.get("pagina_fin","")}</span>'
            f'<div class="hr"></div>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.write(r.get("texto_chunk", ""))
        shown += 1
        if shown >= max_items:
            break

# -----------------------------
# Topics
# -----------------------------
TEMAS = {
    "Empleo": "empleo, trabajo, desempleo, salario, informalidad",
    "Seguridad": "seguridad, crimen, violencia, narcotráfico, policía",
    "Educación": "educación, escuela, colegio, universidad, becas",
    "Salud": "salud, CCSS, hospitales, listas de espera, medicina",
    "Economía / costo de vida": "inflación, costo de vida, impuestos, precios",
    "Vivienda": "vivienda, alquiler, bono, construcción",
    "Ambiente / energía": "ambiente, agua, energía, cambio climático",
    "Transparencia": "corrupción, transparencia, ética, rendición de cuentas",
}

PREGUNTAS_SUGERIDAS = {
    "Empleo": "¿Qué proponen sobre empleo y generación de trabajo?",
    "Seguridad": "¿Qué proponen para mejorar la seguridad ciudadana?",
    "Educación": "¿Qué proponen para mejorar la educación pública?",
    "Salud": "¿Qué proponen para fortalecer la CCSS y el sistema de salud?",
    "Economía / costo de vida": "¿Qué proponen para bajar el costo de vida y mejorar la economía?",
    "Vivienda": "¿Qué proponen sobre acceso a vivienda?",
    "Ambiente / energía": "¿Qué proponen sobre ambiente, agua y energía?",
    "Transparencia": "¿Qué proponen contra la corrupción y para más transparencia?",
}

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🆚 Comparar por tema", "💬 Preguntar"])

# =========================================================
# TAB 1: Dashboard + 2 options
# =========================================================
with tab1:
    st.markdown(
        '<div class="card"><div class="card-title">📊 Dashboard ciudadano</div>'
        '<div class="small-note">Ves prioridades por tema, comparas partidos en un tema, y validas con evidencia y páginas.</div></div>',
        unsafe_allow_html=True
    )
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    threshold = st.slider(
        "Precisión para calcular prioridades (más alto = más estricto)",
        0.20, 0.60, threshold_default, 0.01,
        key="threshold_dashboard"
    )

    # Build dist_df
    rows = []
    partidos_unicos = sorted([p for p in df["partido"].unique().tolist() if p])

    for partido in partidos_unicos:
        total = 0
        conteo = {}
        for tema, query in TEMAS.items():
            sims = tema_relevancia(query)
            mask = (df["partido"] == partido) & (sims >= threshold)
            n = int(mask.sum())
            conteo[tema] = n
            total += n

        if total == 0:
            continue

        for tema, n in conteo.items():
            rows.append({
                "partido": partido,
                "tema": tema,
                "porcentaje": (100.0 * n / total)
            })

    dist_df = pd.DataFrame(rows)
    if dist_df.empty:
        st.warning("No se pudo calcular prioridades con este nivel de precisión. Baja la precisión.")
        st.stop()

    dist_df["porcentaje"] = dist_df["porcentaje"].round(1)

    # 1) Prioridades por partido
    st.markdown("## 1) Prioridades dentro del plan (por partido)")
    partido_dashboard = st.selectbox(
        "Selecciona un partido",
        sorted(dist_df["partido"].unique().tolist()),
        key="party_dashboard_select"
    )

    df_plot = dist_df[dist_df["partido"] == partido_dashboard].copy()

    chart = (
        alt.Chart(df_plot)
        .mark_bar()
        .encode(
            x=alt.X("tema:N", sort="-y", title="Tema"),
            y=alt.Y("porcentaje:Q", title="% del plan"),
            tooltip=[alt.Tooltip("tema:N", title="Tema"),
                     alt.Tooltip("porcentaje:Q", title="Porcentaje", format=".1f")]
        )
        .properties(height=420)
    )
    labels = (
        alt.Chart(df_plot)
        .mark_text(dy=-8)
        .encode(
            x=alt.X("tema:N", sort="-y"),
            y="porcentaje:Q",
            text=alt.Text("porcentaje:Q", format=".1f")
        )
    )
    st.altair_chart(chart + labels, use_container_width=True)
    st.caption("Ordenado de mayor a menor. Abajo puedes ver evidencia con páginas.")

    temas_ordenados = df_plot.sort_values("porcentaje", ascending=False)["tema"].tolist()
    tema_sel = st.selectbox("Ver evidencia por tema", temas_ordenados, key="tema_evidencia_dashboard")

    sims = tema_relevancia(TEMAS[tema_sel])
    idx = np.argsort(-sims)

    evidence_rows = []
    for i in idx:
        if df.iloc[i]["partido"] == partido_dashboard and sims[i] >= threshold:
            evidence_rows.append(df.iloc[i].to_dict())
        if len(evidence_rows) >= max(3, k_sel):
            break

    if not evidence_rows:
        st.info("No se encontró evidencia con este nivel de precisión. Baja la precisión.")
    else:
        show_evidence_cards(evidence_rows, max_items=min(3, len(evidence_rows)))
        with st.expander("Ver más evidencia"):
            show_evidence_cards(evidence_rows, max_items=len(evidence_rows))

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # 2) Comparar hasta 3 partidos en un tema
    st.markdown("## 2) Comparar partidos en un tema (hasta 3)")
    st.caption("Elige un tema y hasta 3 partidos. Verás el % de énfasis y evidencia oficial.")

    tema_comp = st.selectbox("Tema a comparar", list(TEMAS.keys()), key="tema_comp_dashboard")
    partidos_lista = sorted(dist_df["partido"].unique().tolist())

    partidos_comp = st.multiselect(
        "Selecciona hasta 3 partidos",
        partidos_lista,
        default=partidos_lista[:3] if len(partidos_lista) >= 3 else partidos_lista,
        max_selections=3,
        key="partidos_comp_dashboard"
    )

    df_comp = dist_df[(dist_df["tema"] == tema_comp) & (dist_df["partido"].isin(partidos_comp))].copy()

    if df_comp.empty:
        st.info("Selecciona un tema y 1–3 partidos.")
    else:
        df_comp["porcentaje"] = df_comp["porcentaje"].round(1)

        chart_comp = (
            alt.Chart(df_comp)
            .mark_bar()
            .encode(
                x=alt.X("partido:N", sort="-y", title="Partido"),
                y=alt.Y("porcentaje:Q", title=f"% del plan dedicado a: {tema_comp}"),
                tooltip=[alt.Tooltip("partido:N", title="Partido"),
                         alt.Tooltip("porcentaje:Q", title="Porcentaje", format=".1f")]
            )
            .properties(height=340)
        )

        labels_comp = (
            alt.Chart(df_comp)
            .mark_text(dy=-8)
            .encode(
                x=alt.X("partido:N", sort="-y"),
                y="porcentaje:Q",
                text=alt.Text("porcentaje:Q", format=".1f")
            )
        )

        st.altair_chart(chart_comp + labels_comp, use_container_width=True)

        st.markdown("#### Evidencia por partido (extractos oficiales)")
        sims_tema = tema_relevancia(TEMAS[tema_comp])
        idx_tema = np.argsort(-sims_tema)

        for p in partidos_comp:
            st.markdown(f"**{p}**")
            ev = []
            for i in idx_tema:
                if df.iloc[i]["partido"] == p and sims_tema[i] >= threshold:
                    ev.append(df.iloc[i].to_dict())
                if len(ev) >= 2:
                    break
            if not ev:
                st.caption("Sin evidencia encontrada con la precisión actual.")
            else:
                show_evidence_cards(ev, max_items=len(ev))
            st.markdown("---")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # 3) Vista rápida top temas
    st.markdown("## 3) Vista rápida de prioridades (Top temas)")
    st.caption("Una vista corta para entender rápidamente qué enfatiza más el plan.")

    partido_prioridades = st.selectbox("Partido (vista rápida)", partidos_lista, key="partido_prioridades_dashboard")
    df_p = dist_df[dist_df["partido"] == partido_prioridades].copy()
    df_p["porcentaje"] = df_p["porcentaje"].round(1)

    top_n = st.slider("Cantidad de temas", 4, min(10, len(df_p)), min(6, len(df_p)), 1, key="top_n_prioridades")
    df_top = df_p.sort_values("porcentaje", ascending=False).head(top_n)

    chart_top = (
        alt.Chart(df_top)
        .mark_bar()
        .encode(
            x=alt.X("tema:N", sort="-y", title="Tema"),
            y=alt.Y("porcentaje:Q", title="% del plan"),
            tooltip=[alt.Tooltip("tema:N", title="Tema"),
                     alt.Tooltip("porcentaje:Q", title="Porcentaje", format=".1f")]
        )
        .properties(height=320)
    )
    labels_top = (
        alt.Chart(df_top)
        .mark_text(dy=-8)
        .encode(
            x=alt.X("tema:N", sort="-y"),
            y="porcentaje:Q",
            text=alt.Text("porcentaje:Q", format=".1f")
        )
    )
    st.altair_chart(chart_top + labels_top, use_container_width=True)

# =========================================================
# TAB 2: Compare by topic (keys added)
# =========================================================
with tab2:
    st.subheader("🆚 Comparar por tema")
    st.caption("Elige un tema, selecciona partidos y revisa evidencia directa (con páginas).")

    temas_lista = list(TEMAS.keys())
    cols = st.columns(4)
    selected_tema = st.session_state.get("tema_compare", temas_lista[0])

    for i, tema in enumerate(temas_lista):
        if cols[i % 4].button(tema, use_container_width=True, key=f"tab2_tema_btn_{tema}"):
            selected_tema = tema
            st.session_state["tema_compare"] = tema

    st.markdown(f"**Tema seleccionado:** {selected_tema}")
    default_q = PREGUNTAS_SUGERIDAS.get(selected_tema, f"¿Qué proponen sobre {selected_tema.lower()}?")
    st.session_state.setdefault("compare_q", default_q)

    compare_q = st.text_input("Pregunta para comparar", value=st.session_state.get("compare_q", default_q), key="tab2_compare_q")
    st.session_state["compare_q"] = compare_q

    default_parties = partidos[:3] if len(partidos) >= 3 else partidos
    selected_parties = st.multiselect("Partidos a comparar", partidos, default=default_parties, max_selections=5, key="tab2_selected_parties")

    if st.button("Comparar", type="primary", key="tab2_compare_btn"):
        if not compare_q.strip():
            st.warning("Escribe una pregunta.")
        elif not selected_parties:
            st.warning("Selecciona al menos un partido.")
        else:
            base_rows = retrieve(df, model, index, compare_q, k=max(k_sel * 5, 20))

            partido_filter = None if sidebar_partido == "Todos" else sidebar_partido
            fuente_filter = None if sidebar_fuente == "Todas" else sidebar_fuente

            for p in selected_parties:
                st.markdown("---")
                st.markdown(f"### {p}")

                use_partido = p if partido_filter is None else partido_filter
                rows_p = filter_rows(base_rows, partido=use_partido, fuente=fuente_filter, k=3)

                if not rows_p:
                    st.caption("Sin evidencia encontrada con estos filtros.")
                    continue

                show_evidence_cards(rows_p, max_items=len(rows_p))
                st.markdown("**Fuentes**")
                st.markdown(format_sources(rows_p))

# =========================================================
# TAB 3: Ask (keys added to buttons)
# =========================================================
with tab3:
    st.subheader("💬 Preguntar")
    st.caption("Escribe tu pregunta. Si no hay cuota de AI, igual verás evidencia con páginas.")

    sug_cols = st.columns(3)
    if sug_cols[0].button("Empleo", use_container_width=True, key="tab3_sug_empleo"):
        st.session_state["question"] = PREGUNTAS_SUGERIDAS["Empleo"]
    if sug_cols[1].button("Seguridad", use_container_width=True, key="tab3_sug_seguridad"):
        st.session_state["question"] = PREGUNTAS_SUGERIDAS["Seguridad"]
    if sug_cols[2].button("Educación", use_container_width=True, key="tab3_sug_educacion"):
        st.session_state["question"] = PREGUNTAS_SUGERIDAS["Educación"]

    question = st.text_input("Tu pregunta", value=st.session_state.get("question", ""), key="tab3_question_input")
    st.session_state["question"] = question

    if st.button("Buscar", type="primary", key="tab3_search_btn") and question.strip():
        base_rows = retrieve(df, model, index, question, k=max(k_sel, 6))

        partido_filter = None if sidebar_partido == "Todos" else sidebar_partido
        fuente_filter = None if sidebar_fuente == "Todas" else sidebar_fuente
        rows = filter_rows(base_rows, partido=partido_filter, fuente=fuente_filter, k=k_sel)

        if not rows:
            st.warning("No se encontró evidencia con esos filtros. Prueba quitando filtros o cambiando la pregunta.")
        else:
            context = "\n\n".join(r["texto_chunk"] for r in rows)
            sources_text = format_sources(rows)

            prompt = f"""
Eres un asistente neutral. NO recomiendas por quién votar. NO inventes.
Responde SOLO con el contexto.

CONTEXTO:
{context}

PREGUNTA:
{question}

Al final incluye:
Fuentes:
{sources_text}
""".strip()

            api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
            client = OpenAI(api_key=api_key) if api_key else None

            try:
                if client is None:
                    raise RateLimitError("No API key configured")

                response = client.chat.completions.create(
                    model=st.secrets.get("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2
                )

                st.markdown("### Respuesta")
                st.write(response.choices[0].message.content)

                st.markdown("### Fuentes")
                st.markdown(sources_text)

            except RateLimitError:
                st.warning("⚠️ Modo evidencia: mostrando extractos exactos de los documentos con páginas.")
                show_evidence_cards(rows, max_items=len(rows))
                st.markdown("### Fuentes")
                st.markdown(sources_text)

            except AuthenticationError:
                st.error("❌ API key inválida. Revisa OPENAI_API_KEY.")

            except OpenAIError as e:
                st.error(f"❌ Error de OpenAI: {e}")

    st.markdown("---")
    with st.expander("Guía rápida"):
        st.markdown("""
- En **Dashboard**, explora prioridades por tema dentro de un partido.  
- En **Comparar por tema**, compara partidos para un tema que te importa.  
- En **Preguntar**, escribe tu duda y revisa las **Fuentes** para verificar.  
        """.strip())
