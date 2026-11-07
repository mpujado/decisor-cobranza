"""
Decisor de Cobranza Temprana — Etapa 1 (versión mínima)

Marco en dos etapas:
  1) Etapa estadística: QDA estima P(ω_k | x) para las 3 clases.
  2) Etapa decisoria:   a* = argmin_i  R(a_i | x) = Σ_k C(a_i, ω_k) · P(ω_k | x)

Datos 100 % simulados. La app recomienda; la decisión final es del analista.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuración fija de la etapa 1
# ---------------------------------------------------------------------------
RUTA_CSV = Path(__file__).parent / "clientes_simulados_cobranza.csv"
SEMILLA = 42
PROPORCION_TEST = 0.30

# n_productos se excluye: es discreta (1–8) y viola el supuesto de normalidad
# multivariada de QDA (decisión de criterio, documentada como limitación).
VARIABLES = ["dias_atraso", "cuota_ingreso", "antiguedad_meses", "saldo_promedio_miles"]

ETIQUETAS_VAR = {
    "dias_atraso": "Días de atraso",
    "cuota_ingreso": "Relación cuota / ingreso",
    "antiguedad_meses": "Antigüedad (meses)",
    "saldo_promedio_miles": "Saldo promedio (miles de $)",
}

CLASES = ["0 · se_regulariza_solo", "1 · regulariza_con_gestion", "2 · mora_grave"]
ACCIONES = ["No hacer nada", "Recordatorio", "Llamado", "Derivar a recupero"]

# Matriz de costos C(acción, clase) en pesos por cliente (definida en la simulación)
COSTOS_BASE = pd.DataFrame(
    [
        [0, 236_500, 765_000],
        [150, 142_050, 765_150],
        [4_000, 4_000, 769_000],
        [80_000, 80_000, 595_000],
    ],
    index=ACCIONES,
    columns=CLASES,
)


# ---------------------------------------------------------------------------
# Etapa estadística: datos y modelo (se calculan una sola vez)
# ---------------------------------------------------------------------------
@st.cache_data
def cargar_datos(ruta: Path) -> pd.DataFrame:
    return pd.read_csv(ruta)


@st.cache_resource
def ajustar_modelo(df: pd.DataFrame):
    X = df[VARIABLES].to_numpy()
    y = df["clase"].to_numpy()
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, df.index.to_numpy(),
        test_size=PROPORCION_TEST, stratify=y, random_state=SEMILLA,
    )
    # Priors = proporciones observadas en train. Sin SMOTE ni rebalanceo:
    # rebalancear descalibra las posteriores y degrada la decisión por costos.
    # QDA es invariante a transformaciones afines, así que no se escalan variables.
    qda = QuadraticDiscriminantAnalysis(store_covariance=True)
    qda.fit(X_tr, y_tr)
    return qda, (X_tr, y_tr, idx_tr), (X_te, y_te, idx_te)


def pesos(v: float) -> str:
    """Formato argentino: $ 1.234.567"""
    return "$ " + f"{v:,.0f}".replace(",", ".")


def riesgo_esperado(posteriores: np.ndarray, costos: pd.DataFrame) -> np.ndarray:
    """R(a_i | x) = Σ_k C(a_i, ω_k) · P(ω_k | x).  posteriores: (n, 3) → (n, 4)."""
    return posteriores @ costos.to_numpy(dtype=float).T


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Decisor de Cobranza Temprana", layout="wide")
st.title("Decisor de Cobranza Temprana")
st.caption(
    "Datos simulados. La app **recomienda**; la decisión final es del analista. "
    "Etapa 1 — versión mínima."
)

if not RUTA_CSV.exists():
    st.error(f"No se encontró el archivo de datos: {RUTA_CSV.name}")
    st.stop()

df = cargar_datos(RUTA_CSV)
qda, (X_tr, y_tr, idx_tr), (X_te, y_te, idx_te) = ajustar_modelo(df)

# --- Barra lateral: datos del cliente ---------------------------------------
st.sidebar.header("Cliente")
modo = st.sidebar.radio(
    "Origen de los datos",
    ["Cliente del dataset (control)", "Carga manual"],
    help="Usá 'Cliente del dataset' para contrastar contra el notebook.",
)

if modo == "Cliente del dataset (control)":
    fila = st.sidebar.number_input(
        "Índice de fila en el CSV", min_value=0, max_value=len(df) - 1, value=0, step=1
    )
    cliente = df.loc[fila, VARIABLES].to_dict()
    conjunto = "train" if fila in set(idx_tr) else "test"
    st.sidebar.write(f"Pertenece al conjunto: **{conjunto}**")
    st.sidebar.write(f"Clase real: **{df.loc[fila, 'clase_nombre']}**")
    for v in VARIABLES:
        st.sidebar.write(f"{ETIQUETAS_VAR[v]}: {cliente[v]}")
else:
    cliente = {
        "dias_atraso": st.sidebar.slider(ETIQUETAS_VAR["dias_atraso"], 0, 30, 12),
        "cuota_ingreso": st.sidebar.slider(ETIQUETAS_VAR["cuota_ingreso"], 0.0, 0.85, 0.28, 0.01),
        "antiguedad_meses": st.sidebar.slider(ETIQUETAS_VAR["antiguedad_meses"], 0, 150, 48),
        "saldo_promedio_miles": st.sidebar.slider(
            ETIQUETAS_VAR["saldo_promedio_miles"], 0.0, 2600.0, 670.0, 10.0
        ),
    }

x = np.array([[cliente[v] for v in VARIABLES]], dtype=float)

# --- Matriz de costos editable ----------------------------------------------
st.subheader("1. Matriz de costos C(acción, clase) — pesos por cliente")
st.write("Editá cualquier celda: la recomendación se recalcula al instante.")
costos = st.data_editor(COSTOS_BASE, width="stretch", key="costos")
if costos.isna().any().any() or (costos < 0).any().any():
    st.error("La matriz de costos debe tener todos los valores completos y no negativos.")
    st.stop()

# --- Etapa estadística + decisoria para el cliente ---------------------------
post = qda.predict_proba(x)  # (1, 3)
riesgos = riesgo_esperado(post, costos)[0]  # (4,)
i_opt = int(np.argmin(riesgos))
clase_mas_probable = CLASES[int(np.argmax(post[0]))]

col1, col2 = st.columns(2)

with col1:
    st.subheader("2. Etapa estadística: P(clase | x)")
    tabla_post = pd.DataFrame({"Probabilidad": post[0]}, index=CLASES)
    st.bar_chart(tabla_post, horizontal=True)
    st.dataframe(tabla_post.style.format("{:.3f}"), width="stretch")
    st.write(f"Clase más probable: **{clase_mas_probable}**")

with col2:
    st.subheader("3. Etapa decisoria: riesgo esperado R(a | x)")
    tabla_riesgo = pd.DataFrame({"Costo esperado ($)": riesgos}, index=ACCIONES)
    st.dataframe(
        tabla_riesgo.style.format(pesos).highlight_min(color="#c8e6c9"),
        width="stretch",
    )
    st.success(f"Acción recomendada: **{ACCIONES[i_opt]}** "
               f"(costo esperado {pesos(riesgos[i_opt])})")

with st.expander("¿Cómo se calcula?"):
    st.latex(r"R(a_i \mid \mathbf{x}) = \sum_{k=0}^{2} C(a_i,\omega_k)\,P(\omega_k \mid \mathbf{x})")
    st.latex(r"a^* = \arg\min_{a_i} R(a_i \mid \mathbf{x})")
    st.write(
        "Las probabilidades las estima el modelo QDA (supone normalidad multivariada "
        "con una matriz de covarianzas propia por clase). Los costos los define quien "
        "conoce el negocio. La acción recomendada no tiene por qué coincidir con la "
        "clase más probable."
    )

# --- Referencia rápida sobre test --------------------------------------------
with st.expander("Referencia del modelo sobre el conjunto de test"):
    post_te = qda.predict_proba(X_te)
    pred_te = post_te.argmax(axis=1)
    acc_opt_te = riesgo_esperado(post_te, costos).argmin(axis=1)
    C = costos.to_numpy(dtype=float)
    costo_politica = C[acc_opt_te, y_te].mean()
    st.write(
        f"Train: {len(y_tr)} clientes · Test: {len(y_te)} clientes · Semilla: {SEMILLA}"
    )
    st.write(f"Accuracy (clase más probable) en test: **{(pred_te == y_te).mean():.3f}**")
    st.write(f"Costo medio por cliente de la política de costo mínimo en test: "
             f"**{pesos(costo_politica)}**")
    st.write("Distribución de acciones recomendadas en test:")
    st.dataframe(
        pd.Series(acc_opt_te).map(dict(enumerate(ACCIONES))).value_counts()
        .reindex(ACCIONES, fill_value=0).rename("Clientes"),
        width="stretch",
    )
