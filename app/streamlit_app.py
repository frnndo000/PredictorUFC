"""App Streamlit: simulador dinámico de peleas de UFC.

Elige dos peleadores -> probabilidad de victoria de cada uno + método probable.
Ejecutar:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

# Permite ejecutar con `streamlit run` (añade la raíz del proyecto al path).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from src.models.predict import list_fighters, simulate_fight

METHOD_ES = {"KO/TKO": "KO/TKO", "SUB": "Sumisión", "DEC": "Decisión"}

st.set_page_config(page_title="Predictor UFC", page_icon="🥊", layout="centered")


@st.cache_data
def _fighters():
    df = list_fighters().drop_duplicates("name")
    return df, dict(zip(df["name"], df["fighter_id"]))


st.title("🥊 Predictor UFC")
st.caption("Simula una pelea hipotética entre dos peleadores y estima la probabilidad "
           "de victoria y el método. Modelo entrenado solo con datos previos a cada "
           "pelea (sin fuga de datos).")

fighters_df, name_to_id = _fighters()
names = fighters_df["name"].tolist()

col_a, col_b = st.columns(2)
with col_a:
    name_a = st.selectbox("Peleador A (esquina roja)", names,
                          index=names.index("Sean Strickland") if "Sean Strickland" in names else 0)
with col_b:
    name_b = st.selectbox("Peleador B (esquina azul)", names,
                          index=names.index("Israel Adesanya") if "Israel Adesanya" in names else 1)

with st.expander("Contexto de la pelea (opcional)"):
    title_bout = st.checkbox("Pelea por título", value=False)

if st.button("Simular pelea", type="primary", use_container_width=True):
    if name_a == name_b:
        st.warning("Elige dos peleadores distintos.")
        st.stop()

    r = simulate_fight(name_to_id[name_a], name_to_id[name_b], title_bout=title_bout)
    p_a, p_b = r["p_a"], r["p_b"]
    winner = r["name_a"] if p_a >= p_b else r["name_b"]

    st.subheader(f"🏆 Favorito: {winner}")
    c1, c2 = st.columns(2)
    c1.metric(r["name_a"], f"{p_a*100:.0f}%")
    c2.metric(r["name_b"], f"{p_b*100:.0f}%")
    st.progress(p_a, text=f"{r['name_a']}  {p_a*100:.0f}%  ·  {p_b*100:.0f}%  {r['name_b']}")

    st.markdown("#### Método de victoria más probable")
    method = {METHOD_ES.get(k, k): v for k, v in r["method"].items()}
    method_df = pd.DataFrame({"Probabilidad": method}).sort_values("Probabilidad", ascending=False)
    top_method = method_df.index[0]
    st.write(f"Más probable: **{top_method}** ({method_df.iloc[0, 0]*100:.0f}%)")
    st.bar_chart(method_df, horizontal=True)

st.divider()
st.caption("⚠️ Proyecto educativo / de portafolio. No usar para apuestas. "
           "Datos: ufcstats.com.")
