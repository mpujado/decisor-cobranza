"""
Decisor de Cobranza Temprana — Etapa 2

Marco en dos etapas:
  1) Etapa estadística: QDA estima P(ω_k | x) para las 3 clases.
  2) Etapa decisoria:   a* = argmin_i  R(a_i | x) = Σ_k C(a_i, ω_k) · P(ω_k | x)

El modelo replica el pipeline de la investigación (notebooks 0b, 3, 5 y 07):
  - Datos: clientes_simulados_normal.csv (normal multivariada por clase en escala logit).
  - 5 variables, partición 70/30 estratificada, semilla 42, QDA sin rebalanceo.
El usuario carga valores en unidades reales y la app aplica la misma transformación logit.

Datos 100 % simulados. La app recomienda; la decisión final es del analista.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.special import expit, logit
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuración (idéntica a la investigación)
# ---------------------------------------------------------------------------
RUTA_CSV = Path(__file__).parent / "clientes_simulados_normal.csv"
SEMILLA = 42
PROPORCION_TEST = 0.30

VARIABLES = ["dias_atraso", "cuota_ingreso", "antiguedad_meses",
             "saldo_promedio_miles", "n_productos"]

# Límites de negocio usados en 0b para llevar cada variable a (0, 1) y luego a logit
LIMITES = {
    "dias_atraso": (0.5, 30.5),
    "cuota_ingreso": (0.0, 1.0),
    "antiguedad_meses": (0.0, 180.0),
    "saldo_promedio_miles": (0.0, 3000.0),
    "n_productos": (0.5, 8.5),
}

ETIQUETAS_VAR = {
    "dias_atraso": "Días de atraso",
    "cuota_ingreso": "Relación cuota / ingreso",
    "antiguedad_meses": "Antigüedad (meses)",
    "saldo_promedio_miles": "Saldo promedio (miles de $)",
    "n_productos": "Cantidad de productos",
}

CLASES = ["0 · se_regulariza_solo", "1 · regulariza_con_gestion", "2 · mora_grave"]
ACCIONES = ["No hacer nada", "Recordatorio", "Llamado", "Derivar a recupero"]

# Matriz de costos C(acción, clase) en pesos por cliente (notebook 6a / 07)
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

# Regla ingenua: clase más probable → acción asociada (notebook 07)
MAPEO_INGENUA = {0: 0, 1: 2, 2: 3}  # 0 → nada, 1 → llamado, 2 → derivar


# ---------------------------------------------------------------------------
# Transformaciones entre escala real y escala logit
# ---------------------------------------------------------------------------
def a_logit(valores: dict) -> np.ndarray:
    """Valores en unidades reales → vector en escala logit (la del modelo)."""
    return np.array([[logit((valores[v] - a) / (b - a)) for v, (a, b) in LIMITES.items()]])


def a_real(fila_logit: pd.Series) -> dict:
    """Fila en escala logit → valores en unidades reales (para mostrar)."""
    return {v: a + (b - a) * expit(fila_logit[v]) for v, (a, b) in LIMITES.items()}


def pesos(v: float) -> str:
    """Formato argentino: $ 1.234.567"""
    return "$ " + f"{v:,.0f}".replace(",", ".")


# ---------------------------------------------------------------------------
# Etapa estadística: datos y modelo (se calculan una sola vez)
# ---------------------------------------------------------------------------
@st.cache_data
def cargar_datos(ruta: Path) -> pd.DataFrame:
    return pd.read_csv(ruta)


@st.cache_resource
def ajustar_modelo(df: pd.DataFrame):
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[VARIABLES], df["clase"],
        test_size=PROPORCION_TEST, stratify=df["clase"], random_state=SEMILLA,
    )
    # Priors = proporciones de entrenamiento. Sin SMOTE ni rebalanceo.
    # QDA es invariante a transformaciones afines: no se escalan variables.
    qda = QuadraticDiscriminantAnalysis().fit(X_tr, y_tr)
    return qda, X_tr, X_te, y_tr.to_numpy(), y_te.to_numpy()


def riesgo_esperado(posteriores: np.ndarray, costos: pd.DataFrame) -> np.ndarray:
    """R(a_i | x) = Σ_k C(a_i, ω_k) · P(ω_k | x).  posteriores: (n, 3) → (n, 4)."""
    return posteriores @ costos.to_numpy(dtype=float).T


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Decisor de Cobranza Temprana", layout="wide")
st.title("Decisor de Cobranza Temprana")
st.caption("Datos simulados. La app **recomienda**; la decisión final es del analista.")

if not RUTA_CSV.exists():
    st.error(f"No se encontró el archivo de datos: {RUTA_CSV.name}")
    st.stop()

df = cargar_datos(RUTA_CSV)
qda, X_tr, X_te, y_tr, y_te = ajustar_modelo(df)

# --- Matriz de costos editable (la usan todas las pestañas) ------------------
st.subheader("Matriz de costos C(acción, clase) — pesos por cliente")
st.write("Editá cualquier celda: todas las pestañas se recalculan al instante.")
costos = st.data_editor(COSTOS_BASE, width="stretch", key="costos")
if costos.isna().any().any() or (costos < 0).any().any():
    st.error("La matriz de costos debe tener todos los valores completos y no negativos.")
    st.stop()
C = costos.to_numpy(dtype=float)

tab_cliente, tab_politicas, tab_metodo = st.tabs(
    ["Cliente individual", "Comparación de políticas", "Cómo funciona"]
)

# =============================================================================
# Pestaña 1: cliente individual
# =============================================================================
with tab_cliente:
    col_in, col_out = st.columns([1, 2])

    with col_in:
        st.markdown("**Datos del cliente**")
        modo = st.radio(
            "Origen de los datos",
            ["Carga manual", "Cliente del conjunto de test (control)"],
            help="Usá 'Cliente del conjunto de test' para contrastar contra el notebook.",
        )
        if modo == "Carga manual":
            cliente = {
                "dias_atraso": st.slider(ETIQUETAS_VAR["dias_atraso"], 1, 30, 12),
                "cuota_ingreso": st.slider(ETIQUETAS_VAR["cuota_ingreso"], 0.01, 0.99, 0.28, 0.01),
                "antiguedad_meses": st.slider(ETIQUETAS_VAR["antiguedad_meses"], 1, 179, 48),
                "saldo_promedio_miles": st.slider(
                    ETIQUETAS_VAR["saldo_promedio_miles"], 10, 2990, 670, 10),
                "n_productos": st.slider(ETIQUETAS_VAR["n_productos"], 1, 8, 3),
            }
            x = a_logit(cliente)
            clase_real = None
        else:
            idx_test = X_te.index.to_list()
            pos = st.number_input("Posición en el conjunto de test (0 a 1.499)",
                                  min_value=0, max_value=len(idx_test) - 1, value=0, step=1)
            fila = X_te.iloc[int(pos)]
            x = fila.to_numpy().reshape(1, -1)
            cliente = a_real(fila)
            clase_real = CLASES[int(y_te[int(pos)])]
            st.write(f"Fila del CSV: **{idx_test[int(pos)]}**")
            st.write(f"Clase real: **{clase_real}**")
            for v in VARIABLES:
                st.write(f"{ETIQUETAS_VAR[v]}: {cliente[v]:,.2f}")

    post = qda.predict_proba(x)
    riesgos = riesgo_esperado(post, costos)[0]
    i_opt = int(np.argmin(riesgos))
    i_mas_probable = int(np.argmax(post[0]))

    with col_out:
        st.markdown("**Etapa estadística: P(clase | x)**")
        tabla_post = pd.DataFrame({"Probabilidad": post[0]}, index=CLASES)
        st.bar_chart(tabla_post, horizontal=True)
        st.dataframe(tabla_post.style.format("{:.3f}"), width="stretch")

        st.markdown("**Etapa decisoria: riesgo esperado R(a | x)**")
        tabla_riesgo = pd.DataFrame({"Costo esperado": riesgos}, index=ACCIONES)
        st.dataframe(tabla_riesgo.style.format(pesos).highlight_min(color="#c8e6c9"),
                     width="stretch")

        st.success(f"Acción recomendada: **{ACCIONES[i_opt]}** "
                   f"(costo esperado {pesos(riesgos[i_opt])})")
        accion_ingenua = ACCIONES[MAPEO_INGENUA[i_mas_probable]]
        if accion_ingenua != ACCIONES[i_opt]:
            st.info(f"La clase más probable es **{CLASES[i_mas_probable]}**, que con la regla "
                    f"ingenua llevaría a **{accion_ingenua}**. Los costos cambian la decisión.")

# =============================================================================
# Pestaña 2: comparación de políticas en test (notebook 07)
# =============================================================================
with tab_politicas:
    st.markdown(
        f"Evaluación sobre los **{len(y_te):,} clientes de test**".replace(",", ".")
        + ", que el modelo no vio durante el ajuste. Cada política asigna una acción por "
        "cliente y se suma el costo de esa acción según la clase real."
    )
    prob_te = qda.predict_proba(X_te)
    n = len(y_te)
    politicas = {
        "Regla de Bayes (mínimo riesgo)": riesgo_esperado(prob_te, costos).argmin(axis=1),
        "Regla ingenua (clase más probable)": np.vectorize(MAPEO_INGENUA.get)(prob_te.argmax(1)),
        "Trivial: no hacer nada a nadie": np.full(n, 0),
        "Trivial: recordatorio a todos": np.full(n, 1),
        "Trivial: llamar a todos": np.full(n, 2),
        "Referencia ideal: conociendo la clase real": C[:, y_te].argmin(axis=0),
    }
    filas = []
    costo_medio = {}
    for nombre, acc in politicas.items():
        c = C[acc, y_te]
        costo_medio[nombre] = c.mean()
        filas.append({"Política": nombre, "Costo total": pesos(c.sum()),
                      "Costo medio por cliente": pesos(c.mean()),
                      "Llamados": int((acc == 2).sum()),
                      "Derivaciones": int((acc == 3).sum())})
    st.dataframe(pd.DataFrame(filas).set_index("Política"), width="stretch")

    bayes = costo_medio["Regla de Bayes (mínimo riesgo)"]
    ingenua = costo_medio["Regla ingenua (clase más probable)"]
    ideal = costo_medio["Referencia ideal: conociendo la clase real"]
    m1, m2 = st.columns(2)
    m1.metric("Ahorro de Bayes frente a la ingenua (por cliente)", pesos(ingenua - bayes),
              f"{(ingenua - bayes) / ingenua * 100:.1f} %".replace(".", ","))
    m2.metric("Brecha de Bayes frente al ideal (por cliente)", pesos(bayes - ideal))

    st.markdown("**Acción asignada según la clase real**")
    c1, c2 = st.columns(2)
    for col, nombre in [(c1, "Regla ingenua (clase más probable)"),
                        (c2, "Regla de Bayes (mínimo riesgo)")]:
        tab = pd.crosstab(pd.Series([CLASES[k] for k in y_te], name="Clase real"),
                          pd.Series([ACCIONES[a] for a in politicas[nombre]], name="Acción"))
        col.markdown(f"_{nombre}_")
        col.dataframe(tab.reindex(index=CLASES, columns=ACCIONES, fill_value=0),
                      width="stretch")

# =============================================================================
# Pestaña 3: cómo funciona
# =============================================================================
with tab_metodo:
    st.markdown("**1. Etapa estadística (QDA)**")
    st.write("Cada clase se modela como una normal multivariada con su propia media y su "
             "propia matriz de covarianzas. Con las probabilidades a priori π_k (proporciones "
             "de entrenamiento) se obtiene la probabilidad posterior de cada clase:")
    st.latex(r"P(\omega_k \mid \mathbf{x}) = \frac{\pi_k\, f_k(\mathbf{x})}"
             r"{\sum_{j} \pi_j\, f_j(\mathbf{x})}")
    st.markdown("**2. Etapa decisoria (regla de Bayes con costos)**")
    st.latex(r"R(a_i \mid \mathbf{x}) = \sum_{k=0}^{2} C(a_i,\omega_k)\,P(\omega_k \mid \mathbf{x})"
             r"\qquad a^* = \arg\min_{a_i} R(a_i \mid \mathbf{x})")
    st.write("La acción recomendada no tiene por qué coincidir con la clase más probable: "
             "la matriz de costos puede empujar la decisión hacia la acción que evita el "
             "error más caro.")
    st.markdown("**3. Escala de las variables**")
    st.write("Los datos simulados están en escala logit: cada variable se lleva a (0, 1) con "
             "límites de negocio y luego se aplica logit, para que el supuesto de normalidad "
             "multivariada de QDA sea razonable. La app recibe los valores en unidades reales "
             "y aplica la misma transformación:")
    st.latex(r"z = \operatorname{logit}\!\left(\frac{x - a}{b - a}\right)")
    st.dataframe(pd.DataFrame(LIMITES, index=["a (mínimo)", "b (máximo)"]).T
                 .rename(index=ETIQUETAS_VAR), width="stretch")
    st.caption(f"Entrenamiento: {len(y_tr):,} clientes · Test: {len(y_te):,} clientes · "
               f"Semilla: {SEMILLA}".replace(",", "."))
