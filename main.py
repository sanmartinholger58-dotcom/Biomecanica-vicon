"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         PLATAFORMA DE ANÁLISIS BIOMECÁNICO DE SALTOS — Sistema Vicon         ║
║         Desarrollado para análisis de datos Nexus / Force Plates             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import signal
from scipy.stats import ttest_ind, mannwhitneyu, shapiro
import warnings
warnings.filterwarnings("ignore")

# ─── Configuración de página ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Análisis Biomecánico de Saltos — Vicon",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS personalizado ────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Fondo y tipografía general */
    .main { background-color: #0e1117; }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }

    /* Header principal */
    .hero-header {
        background: linear-gradient(135deg, #1a1f2e 0%, #0d1b2a 50%, #1a2744 100%);
        border: 1px solid #2d4a7a;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    .hero-header h1 {
        color: #4fc3f7;
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
        letter-spacing: 0.5px;
    }
    .hero-header p {
        color: #78909c;
        font-size: 0.95rem;
        margin: 0;
    }

    /* Tarjetas de métricas KPI */
    .kpi-card {
        background: linear-gradient(135deg, #1a2744, #0d1b2a);
        border: 1px solid #2d4a7a;
        border-radius: 12px;
        padding: 1.2rem 1rem;
        text-align: center;
        height: 100%;
    }
    .kpi-label {
        color: #78909c;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.4rem;
    }
    .kpi-value {
        color: #4fc3f7;
        font-size: 1.9rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .kpi-unit {
        color: #546e7a;
        font-size: 0.85rem;
        margin-top: 0.2rem;
    }
    .kpi-delta-pos { color: #66bb6a; font-size: 0.8rem; }
    .kpi-delta-neg { color: #ef5350; font-size: 0.8rem; }

    /* Sección de fase */
    .phase-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .phase-flight  { background: #1a3a2a; color: #66bb6a; border: 1px solid #2e7d32; }
    .phase-landing { background: #3a1a1a; color: #ef5350; border: 1px solid #c62828; }
    .phase-contact { background: #1a2a3a; color: #42a5f5; border: 1px solid #1565c0; }

    /* Tabla de resultados */
    .results-table { border-collapse: collapse; width: 100%; }
    .results-table th {
        background: #1a2744;
        color: #4fc3f7;
        padding: 0.6rem 1rem;
        text-align: left;
        font-size: 0.82rem;
        letter-spacing: 0.5px;
    }
    .results-table td {
        padding: 0.55rem 1rem;
        font-size: 0.88rem;
        border-bottom: 1px solid #1a2744;
        color: #cfd8dc;
    }
    .results-table tr:nth-child(even) td { background: #111827; }

    /* Separador de sección */
    .section-title {
        color: #4fc3f7;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.5rem 0;
        border-bottom: 2px solid #2d4a7a;
        margin: 1.5rem 0 1rem 0;
    }

    /* Info box */
    .info-box {
        background: #0d1b2a;
        border-left: 4px solid #4fc3f7;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0 1rem 0;
        color: #90a4ae;
        font-size: 0.85rem;
    }

    /* Sidebar mejorado */
    [data-testid="stSidebar"] {
        background: #0d1b2a;
        border-right: 1px solid #1a2744;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label {
        color: #78909c !important;
        font-size: 0.82rem !important;
    }
    .sidebar-section {
        background: #1a2744;
        border-radius: 8px;
        padding: 0.8rem 0.6rem;
        margin-bottom: 0.8rem;
    }
    .sidebar-title {
        color: #4fc3f7;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO DE PARSEO Y PROCESAMIENTO DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

def fix_decimal(val):
    """Convierte formato Vicon multi-punto: '926.250.488' → 926.250488"""
    if pd.isna(val) or str(val).strip() in ["", "nan", "NaN"]:
        return np.nan
    s = str(val).strip()
    try:
        return float(s)
    except ValueError:
        neg = s.startswith("-")
        abs_s = s.lstrip("-")
        parts = abs_s.split(".")
        if len(parts) >= 2:
            result = float(parts[0] + "." + "".join(parts[1:]))
            return -result if neg else result
        return np.nan


@st.cache_data(show_spinner=False)
def parse_traj(uploaded_file):
    """Parsea archivo de trayectorias Vicon Nexus (.csv)"""
    content = uploaded_file.read()
    from io import BytesIO
    buf = BytesIO(content)
    header_raw = pd.read_csv(buf, header=None, nrows=3, encoding="utf-8-sig")
    r0, r1 = header_raw.iloc[0].tolist(), header_raw.iloc[1].tolist()

    cols, current_marker = [], None
    for i, (g, label) in enumerate(zip(r0, r1)):
        g_str = str(g)
        if ":" in g_str and g_str != "nan":
            current_marker = g_str.split(":")[-1].strip()
        elif g_str not in ["nan", "Frame", "Sub Frame", "NaN"]:
            current_marker = g_str.strip()
        if i == 0:
            cols.append("Frame")
        elif i == 1:
            cols.append("SubFrame")
        elif str(label) not in ["nan", "NaN"] and current_marker:
            cols.append(f"{current_marker}_{label}")
        else:
            cols.append(f"_drop_{i}")

    buf.seek(0)
    df = pd.read_csv(buf, header=None, skiprows=3, encoding="utf-8-sig", dtype=str)
    if len(cols) > df.shape[1]:
        cols = cols[: df.shape[1]]
    elif len(cols) < df.shape[1]:
        cols += [f"_extra_{i}" for i in range(df.shape[1] - len(cols))]
    df.columns = cols
    for col in df.columns:
        df[col] = df[col].apply(fix_decimal)
    drop_cols = [c for c in df.columns if c.startswith("_")]
    return df.drop(columns=drop_cols)


@st.cache_data(show_spinner=False)
def parse_force(uploaded_file):
    """Parsea archivo de fuerzas Vicon Nexus (.csv)"""
    content = uploaded_file.read()
    from io import BytesIO
    buf = BytesIO(content)
    df = pd.read_csv(buf, header=None, skiprows=3, encoding="utf-8-sig")
    if df.shape[1] == 20:
        df.columns = [
            "Frame", "SubFrame",
            "FT_Fx", "FT_Fy", "FT_Fz",
            "FT_Mx", "FT_My", "FT_Mz",
            "FT_Cx", "FT_Cy", "FT_Cz",
            "FP_Fx", "FP_Fy", "FP_Fz",
            "FP_Mx", "FP_My", "FP_Mz",
            "FP_Cx", "FP_Cy", "FP_Cz",
        ]
    elif df.shape[1] == 11:  # una sola plataforma
        df.columns = [
            "Frame", "SubFrame",
            "FT_Fx", "FT_Fy", "FT_Fz",
            "FT_Mx", "FT_My", "FT_Mz",
            "FT_Cx", "FT_Cy", "FT_Cz",
        ]
        for col in ["FP_Fx","FP_Fy","FP_Fz","FP_Mx","FP_My","FP_Mz","FP_Cx","FP_Cy","FP_Cz"]:
            df[col] = 0.0
    return df.apply(pd.to_numeric, errors="coerce")


def detect_phases(force_df, threshold=20.0):
    """Detecta fases de vuelo y contacto desde GRF vertical"""
    grf = force_df["GRF_z"].values
    contact = grf > threshold
    diff = np.diff(contact.astype(int), prepend=0)
    takeoff_idx = np.where(diff == -1)[0].tolist()
    landing_idx = np.where(diff == 1)[0].tolist()
    return takeoff_idx, landing_idx


def butter_lowpass(data, cutoff, fs, order=4):
    """Filtro Butterworth paso bajo"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
    return signal.filtfilt(b, a, data)


def vec_angle(A, B, C):
    """Ángulo en B (grados) entre segmentos BA y BC"""
    vec1, vec2 = A - B, C - B
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return np.nan
    cos_a = np.dot(vec1, vec2) / (norm1 * norm2)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


# ══════════════════════════════════════════════════════════════════════════════
#  MOTOR DE CÁLCULO BIOMECÁNICO
# ══════════════════════════════════════════════════════════════════════════════

def compute_biomechanics(force_df, traj_df, fs_force=1000, fs_traj=100,
                         mass_kg_override=None, grf_threshold=20.0,
                         filter_cutoff=50.0, apply_filter=True):
    """
    Calcula variables biomecánicas completas a partir de datos de plataforma
    de fuerza y trayectorias de marcadores Vicon.

    Retorna: dict con todas las métricas calculadas
    """
    res = {}
    f = force_df.copy()
    f["GRF_z"] = -f["FT_Fz"]
    f["GRF_x"] = -f["FT_Fx"]
    f["GRF_y"] = -f["FT_Fy"]
    f["GRF_mag"] = np.sqrt(f["GRF_x"]**2 + f["GRF_y"]**2 + f["GRF_z"]**2)
    f["time"] = np.arange(len(f)) / fs_force

    # Filtrado GRF
    if apply_filter and filter_cutoff > 0:
        try:
            f["GRF_z_filt"] = butter_lowpass(f["GRF_z"].values, filter_cutoff, fs_force)
            f["GRF_x_filt"] = butter_lowpass(f["GRF_x"].values, filter_cutoff, fs_force)
            f["GRF_y_filt"] = butter_lowpass(f["GRF_y"].values, filter_cutoff, fs_force)
        except Exception:
            f["GRF_z_filt"] = f["GRF_z"]
    else:
        f["GRF_z_filt"] = f["GRF_z"]

    grf_threshold = grf_threshold
    takeoffs, landings = detect_phases(f, threshold=grf_threshold)

    # ── Peso corporal ─────────────────────────────────────────────────────────
    if mass_kg_override and mass_kg_override > 0:
        mass_kg = mass_kg_override
        BW = mass_kg * 9.81
    else:
        # Usar los últimos 500 frames estables después del aterrizaje
        tail_grf = f["GRF_z"].tail(500)
        stable = tail_grf[tail_grf > 50]
        BW = stable.median() if len(stable) > 10 else f["GRF_z"][f["GRF_z"] > 50].median()
        mass_kg = BW / 9.81

    res["BW_N"] = round(float(BW), 2)
    res["mass_kg"] = round(float(mass_kg), 2)
    res["fs_force"] = fs_force
    res["fs_traj"] = fs_traj
    res["n_force_samples"] = len(f)
    res["duration_s"] = round(len(f) / fs_force, 3)

    # ── Identificación del salto principal ───────────────────────────────────
    flight_phases = []
    for to in takeoffs:
        for la in landings:
            if la > to:
                flight_phases.append((to, la, la - to))
                break

    res["n_flight_phases"] = len(flight_phases)
    res["force_df"] = f  # dataframe con columnas calculadas

    if not flight_phases:
        res["error"] = "No se detectó fase de vuelo en los datos."
        return res

    # Salto principal = mayor tiempo de vuelo
    best = max(flight_phases, key=lambda x: x[2])
    to_idx, la_idx, dur = best
    res["takeoff_idx"] = to_idx
    res["landing_idx"] = la_idx

    # ── TIEMPO DE VUELO ───────────────────────────────────────────────────────
    flight_time = dur / fs_force
    res["flight_time_s"] = round(flight_time, 4)

    # ── ALTURA DE SALTO (método tiempo de vuelo) ──────────────────────────────
    # h = g * t² / 8  (t = tiempo de vuelo completo)
    h_flight = 9.81 * flight_time**2 / 8
    res["jump_height_flight_m"] = round(h_flight, 4)
    res["jump_height_flight_cm"] = round(h_flight * 100, 2)

    # ── FASE DE PROPULSIÓN (countermovement) ──────────────────────────────────
    # Buscar inicio del movimiento: GRF cae por debajo de BW
    prop_search_start = max(0, to_idx - int(3.0 * fs_force))
    grf_pre = f.iloc[prop_search_start:to_idx]["GRF_z_filt"].values

    # Inicio countermovement: primer punto donde GRF < 0.95*BW
    cm_candidates = np.where(grf_pre < 0.95 * BW)[0]
    if len(cm_candidates) > 0:
        cm_start_rel = cm_candidates[0]
        cm_start_abs = prop_search_start + cm_start_rel
    else:
        cm_start_abs = prop_search_start

    res["cm_start_idx"] = cm_start_abs
    cm_duration = (to_idx - cm_start_abs) / fs_force
    res["cm_duration_s"] = round(cm_duration, 4)

    # ── MÉTODO DEL IMPULSO ────────────────────────────────────────────────────
    # v_takeoff = impulse / mass  → h = v²/(2g)
    prop_grf = f.iloc[cm_start_abs:to_idx]["GRF_z_filt"].values
    dt = 1.0 / fs_force
    net_force = prop_grf - BW
    impulse = float(np.trapezoid(net_force, dx=dt))
    v_takeoff = impulse / mass_kg
    h_impulse = (v_takeoff**2) / (2 * 9.81) if v_takeoff > 0 else np.nan
    res["impulse_Ns"] = round(impulse, 3)
    res["v_takeoff_ms"] = round(v_takeoff, 4)
    res["jump_height_impulse_m"] = round(h_impulse, 4) if not np.isnan(h_impulse) else None
    res["jump_height_impulse_cm"] = round(h_impulse * 100, 2) if not np.isnan(h_impulse) else None

    # ── FUERZAS PICO ──────────────────────────────────────────────────────────
    res["peak_GRF_N"] = round(float(f["GRF_z"].max()), 2)
    res["peak_GRF_BW"] = round(float(f["GRF_z"].max() / BW), 3)

    # Pico durante propulsión (push-off)
    push_grf = f.iloc[cm_start_abs:to_idx]["GRF_z"].values
    res["peak_push_GRF_N"] = round(float(push_grf.max()), 2) if len(push_grf) > 0 else None
    res["peak_push_GRF_BW"] = round(float(push_grf.max() / BW), 3) if len(push_grf) > 0 else None

    # Pico en aterrizaje
    land_end = min(la_idx + int(0.5 * fs_force), len(f))
    landing_grf = f.iloc[la_idx:land_end]["GRF_z"].values
    res["peak_landing_GRF_N"] = round(float(landing_grf.max()), 2) if len(landing_grf) > 0 else None
    res["peak_landing_GRF_BW"] = round(float(landing_grf.max() / BW), 3) if len(landing_grf) > 0 else None

    # Fuerza mínima (descarga countermovement)
    res["min_GRF_N"] = round(float(f["GRF_z"].min()), 2)
    res["min_GRF_BW"] = round(float(f["GRF_z"].min() / BW), 3)

    # ── TASA DE DESARROLLO DE FUERZA (RFD) ────────────────────────────────────
    # RFD = ΔF/Δt en ms ventana de 0-200ms desde el inicio de la carga
    grf_vals = f["GRF_z_filt"].values
    # Encontrar punto de inicio de carga (GRF cruza el umbral)
    contact_mask = grf_vals > grf_threshold
    if np.any(contact_mask):
        first_contact_idx_grf = np.argmax(contact_mask)
        # RFD 0-50ms
        rfd_50ms = int(0.05 * fs_force)
        rfd_100ms = int(0.10 * fs_force)
        rfd_200ms = int(0.20 * fs_force)
        end_50 = min(first_contact_idx_grf + rfd_50ms, len(grf_vals))
        end_100 = min(first_contact_idx_grf + rfd_100ms, len(grf_vals))
        end_200 = min(first_contact_idx_grf + rfd_200ms, len(grf_vals))
        res["RFD_0_50ms"] = round(float(grf_vals[end_50] - grf_vals[first_contact_idx_grf]) / 0.05, 1)
        res["RFD_0_100ms"] = round(float(grf_vals[end_100] - grf_vals[first_contact_idx_grf]) / 0.10, 1)
        res["RFD_0_200ms"] = round(float(grf_vals[end_200] - grf_vals[first_contact_idx_grf]) / 0.20, 1)
    else:
        res["RFD_0_50ms"] = None
        res["RFD_0_100ms"] = None
        res["RFD_0_200ms"] = None

    # ── POTENCIA ──────────────────────────────────────────────────────────────
    # Potencia = F × v_CoM ; v_CoM integrada de la fuerza neta
    if len(push_grf) > 5:
        net_f = push_grf - BW
        vel_arr = np.cumsum(net_f) * dt / mass_kg
        power_arr = push_grf * vel_arr
        res["peak_power_W"] = round(float(np.max(np.abs(power_arr))), 1)
        pos_power = power_arr[power_arr > 0]
        res["mean_power_W"] = round(float(np.mean(pos_power)), 1) if len(pos_power) > 0 else None
        # Potencia relativa al peso corporal
        res["peak_power_W_kg"] = round(res["peak_power_W"] / mass_kg, 2)
        res["power_arr"] = power_arr
        res["vel_arr"] = vel_arr
    else:
        res["peak_power_W"] = None
        res["mean_power_W"] = None
        res["peak_power_W_kg"] = None
        res["power_arr"] = np.array([])
        res["vel_arr"] = np.array([])

    # ── ÍNDICE DE ELASTICIDAD (Reactive Strength Index modificado) ─────────
    # RSImod = jump_height / time_to_takeoff_from_landing
    res["RSI_mod"] = None  # Requiere CMJA o Drop Jump con datos de contact time

    # ── SIMETRÍA ──────────────────────────────────────────────────────────────
    # Basada en el CoP (Center of Pressure) durante propulsión
    if len(push_grf) > 5 and "FT_Cx" in f.columns:
        cop_x = f.iloc[cm_start_abs:to_idx]["FT_Cx"].values
        cop_y = f.iloc[cm_start_abs:to_idx]["FT_Cy"].values
        res["cop_range_x_mm"] = round(float(np.nanmax(cop_x) - np.nanmin(cop_x)), 2)
        res["cop_range_y_mm"] = round(float(np.nanmax(cop_y) - np.nanmin(cop_y)), 2)
    else:
        res["cop_range_x_mm"] = None
        res["cop_range_y_mm"] = None

    # ── ÍNDICE DE DESCARGA (Unloading) ────────────────────────────────────────
    # Ratio de descarga durante countermovement
    res["unloading_ratio"] = round(float(BW / res["peak_push_GRF_N"]), 3) if res["peak_push_GRF_N"] else None

    # ── KINEMATICS (trayectorias) ─────────────────────────────────────────────
    if traj_df is not None and len(traj_df) > 0:
        traj = traj_df.copy()
        traj["time"] = (traj["Frame"] - 1) / fs_traj

        res["traj_frames"] = len(traj)

        # Crear Centro de Masa (CoM) - Promedio de LASI y RASI
        traj["CoM_X"] = (traj["LASI_X"] + traj.get("RASI_X", traj["LASI_X"])) / 2
        traj["CoM_Y"] = (traj["LASI_Y"] + traj.get("RASI_Y", traj["LASI_Y"])) / 2
        traj["CoM_Z"] = (traj["LASI_Z"] + traj.get("RASI_Z", traj["LASI_Z"])) / 2

        # Interpolación lineal para mantener continuidad durante el vuelo
        traj["CoM_Z"] = traj["CoM_Z"].interpolate(method='linear', limit_direction='both')

        # Guardar para graficar
        res["traj_valid_df"] = traj

        # Estadísticas del CoM
        if len(traj) > 2 and traj["CoM_Z"].notna().any():
            res["CoM_Z_max_mm"] = round(float(traj["CoM_Z"].max()), 2)
            res["CoM_Z_min_mm"] = round(float(traj["CoM_Z"].min()), 2)
            res["CoM_Z_range_mm"] = round(float(traj["CoM_Z"].max() - traj["CoM_Z"].min()), 2)
        else:
            res["CoM_Z_max_mm"] = None
            res["CoM_Z_min_mm"] = None
            res["CoM_Z_range_mm"] = None

        # ── ÁNGULOS ARTICULARES ──────────────────────────────────────────────
        angle_data = []
        for _, row in traj.iterrows():
            frame_angles = {"Frame": row["Frame"], "time": row["time"]}

            # Rodilla Izquierda
            try:
                A = np.array([row["LTHI_X"], row["LTHI_Y"], row["LTHI_Z"]])
                B = np.array([row["LKNE_X"], row["LKNE_Y"], row["LKNE_Z"]])
                C = np.array([row["LANK_X"], row["LANK_Y"], row["LANK_Z"]])
                frame_angles["knee_L"] = vec_angle(A, B, C)
            except:
                frame_angles["knee_L"] = np.nan

            # Rodilla Derecha
            try:
                A = np.array([row["RTHI_X"], row["RTHI_Y"], row["RTHI_Z"]])
                B = np.array([row["RKNE_X"], row["RKNE_Y"], row["RKNE_Z"]])
                C = np.array([row["RANK_X"], row["RANK_Y"], row["RANK_Z"]])
                frame_angles["knee_R"] = vec_angle(A, B, C)
            except:
                frame_angles["knee_R"] = np.nan

            # Cadera Izquierda
            try:
                A = np.array([row["LASI_X"], row["LASI_Y"], row["LASI_Z"]])
                B = np.array([row["LTHI_X"], row["LTHI_Y"], row["LTHI_Z"]])
                C = np.array([row["LKNE_X"], row["LKNE_Y"], row["LKNE_Z"]])
                frame_angles["hip_L"] = vec_angle(A, B, C)
            except:
                frame_angles["hip_L"] = np.nan

            # Cadera Derecha
            try:
                A = np.array([row["RASI_X"], row["RASI_Y"], row["RASI_Z"]])
                B = np.array([row["RTHI_X"], row["RTHI_Y"], row["RTHI_Z"]])
                C = np.array([row["RKNE_X"], row["RKNE_Y"], row["RKNE_Z"]])
                frame_angles["hip_R"] = vec_angle(A, B, C)
            except:
                frame_angles["hip_R"] = np.nan

            # Tobillo Izquierdo
            try:
                A = np.array([row["LKNE_X"], row["LKNE_Y"], row["LKNE_Z"]])
                B = np.array([row["LANK_X"], row["LANK_Y"], row["LANK_Z"]])
                C = np.array([row["LTOE_X"], row["LTOE_Y"], row["LTOE_Z"]])
                frame_angles["ankle_L"] = vec_angle(A, B, C)
            except:
                frame_angles["ankle_L"] = np.nan

            # Tobillo Derecho
            try:
                A = np.array([row["RKNE_X"], row["RKNE_Y"], row["RKNE_Z"]])
                B = np.array([row["RANK_X"], row["RANK_Y"], row["RANK_Z"]])
                C = np.array([row["RTOE_X"], row["RTOE_Y"], row["RTOE_Z"]])
                frame_angles["ankle_R"] = vec_angle(A, B, C)
            except:
                frame_angles["ankle_R"] = np.nan

            angle_data.append(frame_angles)

        angles_df = pd.DataFrame(angle_data)
        res["angles_df"] = angles_df

        # Estadísticas de ángulos
        for joint in ["knee_L", "knee_R", "hip_L", "hip_R", "ankle_L", "ankle_R"]:
            vals = angles_df[joint].dropna()
            if len(vals) > 0:
                res[f"{joint}_min"] = round(float(vals.min()), 2)
                res[f"{joint}_max"] = round(float(vals.max()), 2)
                res[f"{joint}_mean"] = round(float(vals.mean()), 2)
                res[f"{joint}_range"] = round(float(vals.max() - vals.min()), 2)

        # Simetría bilateral (ASI)
        for pair in [("knee_L", "knee_R"), ("hip_L", "hip_R"), ("ankle_L", "ankle_R")]:
            jL, jR = pair
            vL = angles_df[jL].dropna()
            vR = angles_df[jR].dropna()
            if len(vL) > 0 and len(vR) > 0:
                mean_L = vL.mean()
                mean_R = vR.mean()
                if (mean_L + mean_R) > 0:
                    asi = abs(mean_L - mean_R) / ((mean_L + mean_R) / 2) * 100
                    res[f"ASI_{jL.split('_')[0]}"] = round(float(asi), 2)
    else:
        res["traj_valid_df"] = None
        res["angles_df"] = None

    return res


# ══════════════════════════════════════════════════════════════════════════════
#  COMPONENTES DE VISUALIZACIÓN
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {
    "primary": "#4fc3f7",
    "secondary": "#66bb6a",
    "accent": "#ef5350",
    "warning": "#ffa726",
    "bg": "#0d1b2a",
    "grid": "#1a2744",
    "text": "#cfd8dc",
}


def plotly_layout(title="", height=380):
    return dict(
        title=dict(text=title, font=dict(color=COLORS["text"], size=14)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"], family="Inter, sans-serif"),
        height=height,
        margin=dict(l=50, r=20, t=45, b=40),
        xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        hoverlabel=dict(bgcolor="#1a2744", bordercolor="#4fc3f7", font_size=12),
    )


def plot_grf_timeline(res, label="Datos", color="#4fc3f7", height=380):
    """GRF con mejor etiquetado de fases"""
    f = res["force_df"]
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=f["time"], y=f["GRF_z_filt"],
        name=f"GRF vertical ({label})",
        line=dict(color=color, width=3),
        fill="tozeroy", fillcolor=f"rgba(79,195,247,0.12)",
    ))

    # Línea de peso corporal (más visible)
    fig.add_hline(y=res["BW_N"], line_dash="dash", line_color="#ffa726", line_width=3,
                  annotation_text=f"PESO CORPORAL ({res['BW_N']:.0f} N)",
                  annotation_font_size=14, annotation_font_color="#ffa726", annotation_yshift=16)

    # Fases
    to_t = res["takeoff_idx"] / res["fs_force"]
    la_t = res["landing_idx"] / res["fs_force"]
    cm_t = res["cm_start_idx"] / res["fs_force"]

    fig.add_vrect(x0=cm_t, x1=to_t, fillcolor="#66bb6a", opacity=0.15,
                  annotation_text="Propulsión", annotation_position="top left",
                  annotation_font_size=13, annotation_font_color="#66bb6a")
    fig.add_vrect(x0=to_t, x1=la_t, fillcolor="#ef5350", opacity=0.15,
                  annotation_text="VUELO", annotation_position="top",
                  annotation_font_size=13, annotation_font_color="#ef5350")

    fig.add_vline(x=to_t, line_dash="dot", line_color="#ef5350", line_width=3,
                  annotation_text="", annotation_font_size=14)
    fig.add_vline(x=la_t, line_dash="dot", line_color="#26ff79", line_width=3,
                  annotation_text="DESCENSO", annotation_font_size=14, annotation_font_color="#26ff79")

    fig.update_layout(
        title=f"Fuerza de Reacción del Suelo — {label}",
        xaxis_title="Tiempo (s)",
        yaxis_title="GRF vertical (N)",
        height=height,
        plot_bgcolor="#0d1b2a",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cfd8dc"),
        xaxis=dict(gridcolor="#1a2744"),
        yaxis=dict(gridcolor="#1a2744")
    )
    return fig

def plot_power_velocity(res, label="Datos", height=360):
    """Potencia y velocidad del CoM durante propulsión"""
    f = res["force_df"]
    power = res.get("power_arr", np.array([]))
    vel = res.get("vel_arr", np.array([]))
    if len(power) == 0:
        return None

    cm_idx = res["cm_start_idx"]
    to_idx = res["takeoff_idx"]
    t_prop = f.iloc[cm_idx:to_idx]["time"].values

    min_len = min(len(t_prop), len(power), len(vel))
    t_prop = t_prop[:min_len]
    power = power[:min_len]
    vel = vel[:min_len]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=t_prop, y=power, name="Potencia (W)",
                             line=dict(color=COLORS["primary"], width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=t_prop, y=vel, name="Velocidad CoM (m/s)",
                             line=dict(color=COLORS["secondary"], width=2, dash="dot")), secondary_y=True)

    layout = plotly_layout(f"Potencia y Velocidad del CoM — {label}", height)
    layout["xaxis"]["title"] = "Tiempo (s)"
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="Potencia (W)", secondary_y=False,
                     gridcolor=COLORS["grid"], color=COLORS["primary"])
    fig.update_yaxes(title_text="Velocidad (m/s)", secondary_y=True, color=COLORS["secondary"])
    return fig


def plot_angles(res, side="L", label="Datos", height=360):
    """Gráfico de ángulos articulares en el tiempo"""
    adf = res.get("angles_df")
    if adf is None or len(adf) == 0:
        return None

    fig = go.Figure()
    joints = {"knee": "Rodilla", "hip": "Cadera", "ankle": "Tobillo"}
    col_map = {
        "knee": COLORS["primary"],
        "hip": COLORS["secondary"],
        "ankle": COLORS["warning"],
    }
    for jkey, jname in joints.items():
        col = f"{jkey}_{side}"
        if col in adf.columns:
            vals = adf[col].values
            fig.add_trace(go.Scatter(
                x=adf["time"], y=vals,
                name=f"{jname} ({'Izq.' if side == 'L' else 'Der.'})",
                line=dict(color=col_map[jkey], width=2),
                connectgaps=False,
            ))

    layout = plotly_layout(
        f"Ángulos Articulares ({'Izquierdo' if side == 'L' else 'Derecho'}) — {label}", height)
    layout["xaxis"]["title"] = "Tiempo (s)"
    layout["yaxis"]["title"] = "Ángulo (°)"
    fig.update_layout(**layout)
    return fig


def plot_cop(res, label="Datos", height=340):
    """Trayectoria del Centro de Presión"""
    f = res["force_df"]
    cm_idx = res["cm_start_idx"]
    to_idx = res["takeoff_idx"]

    cop_x = f.iloc[cm_idx:to_idx]["FT_Cx"].values
    cop_y = f.iloc[cm_idx:to_idx]["FT_Cy"].values

    valid = ~np.isnan(cop_x) & ~np.isnan(cop_y)
    if not np.any(valid):
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cop_x[valid], y=cop_y[valid],
        mode="lines+markers",
        name="Trayectoria CoP",
        line=dict(color=COLORS["primary"], width=1.5),
        marker=dict(size=3, color=np.arange(np.sum(valid)), colorscale="Blues"),
    ))
    fig.add_trace(go.Scatter(
        x=[cop_x[valid][0]], y=[cop_y[valid][0]],
        mode="markers", name="Inicio",
        marker=dict(color=COLORS["secondary"], size=10, symbol="circle"),
    ))
    fig.add_trace(go.Scatter(
        x=[cop_x[valid][-1]], y=[cop_y[valid][-1]],
        mode="markers", name="Final (despegue)",
        marker=dict(color=COLORS["accent"], size=10, symbol="x"),
    ))

    layout = plotly_layout(f"Centro de Presión (CoP) — {label}", height)
    layout["xaxis"]["title"] = "CoP X (mm)"
    layout["yaxis"]["title"] = "CoP Y (mm)"
    layout["yaxis"]["scaleanchor"] = "x"
    fig.update_layout(**layout)
    return fig


def plot_com_trajectory(res, label="Datos", height=460):
    """Trayectoria Vertical del Centro de Masa - Versión Corregida"""
    vdf = res.get("traj_valid_df")
    if vdf is None or len(vdf) == 0 or "CoM_Z" not in vdf.columns:
        st.warning("No hay datos suficientes del Centro de Masa.")
        return None

    df = vdf.copy()
    df["CoM_Z"] = df["CoM_Z"].interpolate(method='linear', limit_direction='both')
    df["CoM_Z"] = df["CoM_Z"].rolling(window=5, min_periods=1, center=True).mean()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["time"], 
        y=df["CoM_Z"],
        mode="lines",
        name="Centro de Masa",
        line=dict(color="#4fc3f7", width=4),
        fill="tozeroy",
        fillcolor="rgba(79,195,247,0.10)"
    ))

    # Fases
    # Se dibujan primero las líneas y luego las etiquetas como anotaciones
    # independientes para evitar que DESPEGUE y ATERRIZAJE se superpongan
    # cuando ambos eventos están cercanos en el eje temporal.
    if "takeoff_idx" in res and res["takeoff_idx"] is not None:
        to_t = res["takeoff_idx"] / res.get("fs_force", 1000)
        fig.add_vline(
            x=to_t,
            line_dash="dash",
            line_color="#ef5350",
            line_width=3,
        )
        fig.add_annotation(
            x=to_t,
            y=1.10,
            xref="x",
            yref="paper",
            text="DESPEGUE",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            xshift=-8,
            font=dict(color="#ef5350", size=13),
            bgcolor="rgba(10,17,31,0.85)",
            bordercolor="#ef5350",
            borderwidth=1,
            borderpad=3,
        )

    if "landing_idx" in res and res["landing_idx"] is not None:
        la_t = res["landing_idx"] / res.get("fs_force", 1000)
        fig.add_vline(
            x=la_t,
            line_dash="dash",
            line_color="#ffa726",
            line_width=3,
        )
        fig.add_annotation(
            x=la_t,
            y=1.02,
            xref="x",
            yref="paper",
            text="ATERRIZAJE",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            xshift=8,
            font=dict(color="#ffa726", size=13),
            bgcolor="rgba(10,17,31,0.85)",
            bordercolor="#ffa726",
            borderwidth=1,
            borderpad=3,
        )

    fig.update_layout(
        title=f"Trayectoria Vertical del Centro de Masa — {label}",
        xaxis_title="Tiempo (s)",
        yaxis_title="Altura del CoM (mm)",
        height=height,
        plot_bgcolor="#0a111f",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0", size=13),
        margin=dict(l=50, r=20, t=90, b=45),
        yaxis=dict(gridcolor="#1a2744"),
        xaxis=dict(gridcolor="#1a2744")
    )

    return fig


def plot_bilateral_symmetry(res, height=360):
    """Diagrama de barras para simetría bilateral"""
    joints = ["Rodilla", "Cadera", "Tobillo"]
    keys = [("knee_L_mean", "knee_R_mean"),
            ("hip_L_mean", "hip_R_mean"),
            ("ankle_L_mean", "ankle_R_mean")]

    fig = go.Figure()
    vals_L = [res.get(k[0], 0) or 0 for k in keys]
    vals_R = [res.get(k[1], 0) or 0 for k in keys]

    fig.add_trace(go.Bar(
        name="Izq.",
        x=joints, y=vals_L,
        marker_color=COLORS["primary"], opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        name="Der.",
        x=joints, y=vals_R,
        marker_color=COLORS["secondary"], opacity=0.85,
    ))

    layout = plotly_layout("Ángulos Medios — Comparación Bilateral", height)
    layout["barmode"] = "group"
    layout["xaxis"]["title"] = "Articulación"
    layout["yaxis"]["title"] = "Ángulo medio (°)"
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS DE UI
# ══════════════════════════════════════════════════════════════════════════════

def kpi(label, value, unit="", delta=None):
    delta_html = ""
    if delta is not None:
        cls = "kpi-delta-pos" if delta >= 0 else "kpi-delta-neg"
        sign = "+" if delta >= 0 else ""
        delta_html = f'<div class="{cls}">{sign}{delta:.2f}%</div>'
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-unit">{unit}</div>
        {delta_html}
    </div>
    """


def section(title):
    st.markdown(f'<div class="section-title">⬥ {title}</div>', unsafe_allow_html=True)


def info(text):
    st.markdown(f'<div class="info-box">ℹ {text}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  INTERFACE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-header">
        <h1>Plataforma de Análisis Biomecánico de Saltos</h1>
        <p>Sistema de análisis de fuerzas y cinemática — Datos Vicon Nexus | Investigación Biomecánica</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Configuración")

        st.markdown('<div class="sidebar-title">📂 Archivos de entrada</div>', unsafe_allow_html=True)
        f_data = st.file_uploader("(FUERZA.csv)", type="csv", key="f_data")
        t_data = st.file_uploader("(TRAYECTORIA.csv)", type="csv", key="t_data")

        st.markdown("---")
        st.markdown('<div class="sidebar-title">🔧 Parámetros</div>', unsafe_allow_html=True)

        mass_override = st.number_input(
            "Masa corporal (kg) — dejar 0 para auto-detectar",
            min_value=0.0, max_value=200.0, value=0.0, step=0.5,
            help="Si se ingresa un valor > 0, se usa este valor en lugar del estimado automáticamente."
        )
        fs_force = st.selectbox("Frecuencia muestreo fuerza (Hz)", [1000, 2000, 500], index=0)
        fs_traj = st.selectbox("Frecuencia muestreo trayectorias (Hz)", [100, 200, 120, 50], index=0)
        grf_thresh = st.slider("Umbral detección contacto (N)", 5, 100, 20,
                               help="Umbral de GRF para determinar contacto con la plataforma.")
        filter_cutoff = st.slider("Frecuencia de corte filtro GRF (Hz)", 10, 100, 50,
                                  help="Filtro Butterworth paso-bajo aplicado a la GRF.")
        apply_filter = st.checkbox("Aplicar filtro a GRF", value=True)

        st.markdown("---")
        run_btn = st.button("▶ CALCULAR ANÁLISIS", use_container_width=True, type="primary")

    # ── Validación de archivos ─────────────────────────────────────────────────
    if not run_btn:
        st.markdown("""
        <div style="text-align:center; padding: 4rem 2rem; color: #546e7a;">
            <div style="font-size:4rem; margin-bottom:1rem;">📁</div>
            <h3 style="color: #4fc3f7;">Suba los archivos CSV de Vicon en el panel lateral</h3>
            <p>Se requiere al menos el archivo de <b>Fuerza</b>.<br>
            Se cargará un único archivo de fuerza y un único archivo de trayectoria.</p>
            <br>
            <p style="font-size:0.85rem;">
            ✓ Archivos exportados desde Vicon Nexus<br>
            ✓ Formato CSV multi-cabecera (3 filas de header)<br>
            ✓ Plataformas: <i>Cerca_treadmill</i> / <i>Cerca_puerta</i><br>
            ✓ Marcadores LASI, RASI, LKNE, RKNE, LANK, RANK, etc.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    if f_data is None or t_data is None:
        st.error("❌ Se requiere cargar un archivo de Fuerza.csv y un archivo de TRAYECTORIA.csv para continuar.")
        return

    # ── Procesamiento ──────────────────────────────────────────────────────────
    with st.spinner("⏳ Procesando datos biomecánicos…"):
        try:
            force_df_data = parse_force(f_data)
            traj_df_data = parse_traj(t_data)
            res_data = compute_biomechanics(
                force_df_data, traj_df_data,
                fs_force=fs_force, fs_traj=fs_traj,
                mass_kg_override=mass_override if mass_override > 0 else None,
                grf_threshold=grf_thresh,
                filter_cutoff=filter_cutoff,
                apply_filter=apply_filter,
            )
        except Exception as e:
            st.error(f"❌ Error procesando los datos: {e}")
            st.exception(e)
            return

    if "error" in res_data:
        st.error(f"❌ {res_data['error']}")
        return

    # ══════════════════════════════════════════════════════════════════════════
    #  TABS PRINCIPALES
    # ══════════════════════════════════════════════════════════════════════════
    tabs = ["📊 Resumen", "⚡ Fuerzas", "🦵 Cinemática", "📋 Datos"]
    selected_tabs = st.tabs(tabs)

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 1 — RESUMEN KPI
    # ══════════════════════════════════════════════════════════════════════════
    with selected_tabs[0]:
        section("Métricas Principales del Salto")

        cols = st.columns(4)
        with cols[0]:
            st.markdown(kpi("Altura de Salto",
                            f"{res_data['jump_height_flight_cm']:.2f}", "cm"), unsafe_allow_html=True)
        with cols[1]:
            st.markdown(kpi("Tiempo de Vuelo",
                            f"{res_data['flight_time_s']*1000:.1f}", "ms"), unsafe_allow_html=True)
        with cols[2]:
            v_to = res_data.get("v_takeoff_ms")
            st.markdown(kpi("Velocidad Despegue",
                            f"{v_to:.3f}" if v_to else "—", "m/s"), unsafe_allow_html=True)
        with cols[3]:
            pp = res_data.get("peak_power_W")
            st.markdown(kpi("Potencia Pico",
                            f"{pp:.1f}" if pp else "—", "W"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        cols2 = st.columns(4)
        with cols2[0]:
            st.markdown(kpi("GRF Pico", f"{res_data['peak_GRF_N']:.1f}", "N"), unsafe_allow_html=True)
        with cols2[1]:
            st.markdown(kpi("GRF Pico / PC", f"{res_data['peak_GRF_BW']:.2f}", "× BW"), unsafe_allow_html=True)
        with cols2[2]:
            st.markdown(kpi("Peso Corporal", f"{res_data['BW_N']:.1f}", "N"), unsafe_allow_html=True)
        with cols2[3]:
            st.markdown(kpi("Masa Estimada", f"{res_data['mass_kg']:.1f}", "kg"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # GRF overview
        section("Fuerza de Reacción del Suelo — Vista General")
        fig_grf = plot_grf_timeline(res_data, "Datos", COLORS["primary"], height=380)
        st.plotly_chart(fig_grf, use_container_width=True)

        # Tabla resumen completa
        section("Tabla de Variables Biomecánicas Completa")
        summary_data = {
            "Variable": [
                "Tiempo de vuelo", "Altura salto (t. vuelo)", "Altura salto (impulso)",
                "Velocidad de despegue", "Impulso propulsión",
                "GRF pico total", "GRF pico propulsión", "GRF pico / PC",
                "GRF aterrizaje pico", "GRF aterrizaje / PC",
                "Potencia pico", "Potencia relativa", "Potencia media",
                "RFD 0-50ms", "RFD 0-100ms", "RFD 0-200ms",
                "Duración countermovement", "CoP rango X", "CoP rango Y",
                "Masa corporal", "Peso corporal",
            ],
            "Unidad": [
                "ms", "cm", "cm",
                "m/s", "N·s",
                "N", "N", "×BW",
                "N", "×BW",
                "W", "W/kg", "W",
                "N/s", "N/s", "N/s",
                "ms", "mm", "mm",
                "kg", "N",
            ],
            "Resultado": [
                f"{res_data.get('flight_time_s',0)*1000:.1f}",
                f"{res_data.get('jump_height_flight_cm',0):.2f}",
                f"{res_data.get('jump_height_impulse_cm') or '—'}",
                f"{res_data.get('v_takeoff_ms') or '—'}",
                f"{res_data.get('impulse_Ns') or '—'}",
                f"{res_data.get('peak_GRF_N',0):.2f}",
                f"{res_data.get('peak_push_GRF_N') or '—'}",
                f"{res_data.get('peak_GRF_BW',0):.3f}",
                f"{res_data.get('peak_landing_GRF_N') or '—'}",
                f"{res_data.get('peak_landing_GRF_BW') or '—'}",
                f"{res_data.get('peak_power_W') or '—'}",
                f"{res_data.get('peak_power_W_kg') or '—'}",
                f"{res_data.get('mean_power_W') or '—'}",
                f"{res_data.get('RFD_0_50ms') or '—'}",
                f"{res_data.get('RFD_0_100ms') or '—'}",
                f"{res_data.get('RFD_0_200ms') or '—'}",
                f"{res_data.get('cm_duration_s',0)*1000:.1f}",
                f"{res_data.get('cop_range_x_mm') or '—'}",
                f"{res_data.get('cop_range_y_mm') or '—'}",
                f"{res_data.get('mass_kg',0):.2f}",
                f"{res_data.get('BW_N',0):.2f}",
            ],
        }

        st.dataframe(
            pd.DataFrame(summary_data),
            use_container_width=True,
            hide_index=True,
        )

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 2 — FUERZAS
    # ══════════════════════════════════════════════════════════════════════════
    with selected_tabs[1]:
        section("Análisis Detallado de Fuerzas")

        col_sel = "Datos"
        res_sel = res_data

        # GRF timeline detallado
        fig_grf_full = plot_grf_timeline(res_sel, col_sel, COLORS["primary"], 400)
        st.plotly_chart(fig_grf_full, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            section("Potencia y Velocidad del CoM")
            fig_pv = plot_power_velocity(res_sel, col_sel, height=340)
            if fig_pv:
                st.plotly_chart(fig_pv, use_container_width=True)
            else:
                info("Datos de potencia no disponibles (requiere datos durante propulsión).")

        with col2:
            section("Centro de Presión (CoP)")
            fig_cop = plot_cop(res_sel, col_sel, height=340)
            if fig_cop:
                st.plotly_chart(fig_cop, use_container_width=True)
            else:
                info("Datos de CoP no disponibles.")

        # GRF tridimensional
        section("GRF Tridireccional")
        f = res_sel["force_df"]
        fig3d = go.Figure()
        fig3d.add_trace(go.Scatter(x=f["time"], y=f["GRF_z_filt"],
                                   name="GRF vertical (Z)", line=dict(color="#4fc3f7", width=2)))
        fig3d.add_trace(go.Scatter(x=f["time"], y=f["GRF_x"],
                                   name="GRF anterior-posterior (X)", line=dict(color="#66bb6a", width=1.5)))
        fig3d.add_trace(go.Scatter(x=f["time"], y=f["GRF_y"],
                                   name="GRF medio-lateral (Y)", line=dict(color="#ffa726", width=1.5)))
        layout3d = plotly_layout(f"Componentes GRF — {col_sel}", 350)
        layout3d["xaxis"]["title"] = "Tiempo (s)"
        layout3d["yaxis"]["title"] = "Fuerza (N)"
        fig3d.update_layout(**layout3d)
        st.plotly_chart(fig3d, use_container_width=True)

        # RFD
        section("Tasa de Desarrollo de Fuerza (RFD)")
        col_rfd1, col_rfd2, col_rfd3 = st.columns(3)
        with col_rfd1:
            v = res_sel.get("RFD_0_50ms")
            st.metric("RFD 0-50ms", f"{v:.0f} N/s" if v else "—")
        with col_rfd2:
            v = res_sel.get("RFD_0_100ms")
            st.metric("RFD 0-100ms", f"{v:.0f} N/s" if v else "—")
        with col_rfd3:
            v = res_sel.get("RFD_0_200ms")
            st.metric("RFD 0-200ms", f"{v:.0f} N/s" if v else "—")

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 3 — CINEMÁTICA
    # ══════════════════════════════════════════════════════════════════════════
    with selected_tabs[2]:
        section("Análisis Cinemático — Trayectorias de Marcadores")

        if res_data.get("angles_df") is None:
            st.warning("⚠️ No se cargó archivo de trayectorias. Suba el archivo TRAYECTORIA.csv para ver cinemática.")
        else:
            col_sel_k = "Datos"
            res_k = res_data

            col1, col2 = st.columns(2)
            with col1:
                section("Ángulos Articulares — Izquierdo")
                fig_ang_L = plot_angles(res_k, "L", col_sel_k, height=340)
                if fig_ang_L:
                    st.plotly_chart(fig_ang_L, use_container_width=True)

            with col2:
                section("Ángulos Articulares — Derecho")
                fig_ang_R = plot_angles(res_k, "R", col_sel_k, height=340)
                if fig_ang_R:
                    st.plotly_chart(fig_ang_R, use_container_width=True)

            # CoM
            section("Trayectoria Vertical del Centro de Masa")
            fig_com = plot_com_trajectory(res_k, col_sel_k)
            if fig_com:
                st.plotly_chart(fig_com, use_container_width=True)

            # Tabla de ángulos
            section("Estadísticas de Ángulos Articulares")
            angle_rows = []
            for side, side_label in [("L", "Izquierdo"), ("R", "Derecho")]:
                for joint, joint_label in [("knee", "Rodilla"), ("hip", "Cadera"), ("ankle", "Tobillo")]:
                    key_base = f"{joint}_{side}"
                    angle_rows.append({
                        "Articulación": joint_label,
                        "Lado": side_label,
                        "Mín (°)": res_k.get(f"{key_base}_min"),
                        "Máx (°)": res_k.get(f"{key_base}_max"),
                        "Medio (°)": res_k.get(f"{key_base}_mean"),
                        "Rango (°)": res_k.get(f"{key_base}_range"),
                    })
            angle_tbl = pd.DataFrame(angle_rows).dropna(subset=["Mín (°)"])
            st.dataframe(angle_tbl, use_container_width=True, hide_index=True)

            # Simetría bilateral
            section("Índice de Asimetría Bilateral (ASI)")
            info("ASI = |L − R| / ((L + R) / 2) × 100%   |   < 10%: simétrico, 10-15%: leve, > 15%: asimétrico")
            asi_data = {
                "Articulación": ["Rodilla", "Cadera", "Tobillo"],
                "ASI (%)": [
                    res_k.get("ASI_knee"),
                    res_k.get("ASI_hip"),
                    res_k.get("ASI_ankle"),
                ],
            }
            asi_df = pd.DataFrame(asi_data).dropna(subset=["ASI (%)"])
            if not asi_df.empty:
                fig_sym = plot_bilateral_symmetry(res_data, height=320)
                st.plotly_chart(fig_sym, use_container_width=True)
                st.dataframe(asi_df, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 4 — DATOS CRUDOS
    # ══════════════════════════════════════════════════════════════════════════
    with selected_tabs[3]:
        _raw_tab(res_data, f_data)


def _raw_tab(res_data, f_data_ref):
    """Pestaña de datos crudos y exportación"""
    section("Datos y Exportación")

    sel = st.selectbox("Seleccionar dataset:", ["Fuerza", "Ángulos"])

    if sel == "Fuerza":
        df_show = res_data["force_df"][["time", "GRF_x", "GRF_y", "GRF_z", "GRF_z_filt",
                                      "FT_Mx", "FT_My", "FT_Mz", "FT_Cx", "FT_Cy", "FT_Cz"]].round(4)
    elif sel == "Ángulos":
        df_show = res_data.get("angles_df")
        if df_show is None:
            st.warning("No hay datos de trayectoria cargados.")
            return
    else:
        st.warning("Dataset no disponible.")
        return

    st.dataframe(df_show.head(200), use_container_width=True)
    st.caption(f"Mostrando primeras 200 filas de {len(df_show)} totales.")

    # Descarga CSV
    csv_data = df_show.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"⬇ Descargar {sel} (.csv)",
        data=csv_data,
        file_name=f"{sel.replace(' ', '_').lower()}.csv",
        mime="text/csv",
    )

    section("Parámetros de Análisis Utilizados")
    params = {
        "Frecuencia fuerza (Hz)": res_data["fs_force"],
        "Frecuencia trayectoria (Hz)": res_data["fs_traj"],
        "Muestras de fuerza": res_data["n_force_samples"],
        "Duración grabación (s)": res_data["duration_s"],
        "Frames de trayectoria válidos": res_data.get("traj_frames", "—"),
        "Fases de vuelo detectadas": res_data.get("n_flight_phases", "—"),
        "Índice despegue (muestra)": res_data.get("takeoff_idx", "—"),
        "Índice aterrizaje (muestra)": res_data.get("landing_idx", "—"),
    }
    st.table(pd.DataFrame(params.items(), columns=["Parámetro", "Valor"]))


if __name__ == "__main__":
    main()