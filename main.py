"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         PLATAFORMA DE ANÁLISIS BIOMECÁNICO DE SALTOS — Sistema Vicon         ║
║         Desarrollado para análisis de datos Nexus / Force Plates             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import signal

warnings.filterwarnings("ignore")

G = 9.81

# =============================================================================
# Configuración visual
# =============================================================================

st.set_page_config(
    page_title="Análisis Biomecánico General - Vicon",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "primary": "#4fc3f7",
    "secondary": "#66bb6a",
    "accent": "#ef5350",
    "warning": "#ffa726",
    "ok": "#66bb6a",
    "review": "#ffa726",
    "bad": "#ef5350",
    "bg": "#0d1b2a",
    "grid": "#1a2744",
    "text": "#cfd8dc",
}

st.markdown(
    """
<style>
.main { background-color: #0e1117; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
.hero-header {
    background: linear-gradient(135deg, #1a1f2e 0%, #0d1b2a 50%, #1a2744 100%);
    border: 1px solid #2d4a7a; border-radius: 16px; padding: 1.5rem 2rem;
    margin-bottom: 1.2rem; text-align: center;
}
.hero-header h1 { color: #4fc3f7; font-size: 1.9rem; font-weight: 700; margin: 0 0 .3rem 0; }
.hero-header p { color: #90a4ae; font-size: .92rem; margin: 0; }
.section-title { color:#4fc3f7; font-size:1.05rem; font-weight:700; border-bottom:2px solid #2d4a7a; padding:.4rem 0; margin:1.1rem 0 .8rem 0; }
.kpi-card { background: linear-gradient(135deg, #1a2744, #0d1b2a); border:1px solid #2d4a7a; border-radius:12px; padding:1rem .8rem; text-align:center; height:100%; }
.kpi-label { color:#8aa6b3; font-size:.72rem; letter-spacing:1.2px; text-transform:uppercase; margin-bottom:.35rem; }
.kpi-value { color:#4fc3f7; font-size:1.75rem; font-weight:800; line-height:1.1; }
.kpi-unit { color:#6b8794; font-size:.8rem; margin-top:.25rem; }
.status-ok { color:#66bb6a; font-weight:700; }
.status-review { color:#ffa726; font-weight:700; }
.status-bad { color:#ef5350; font-weight:700; }
.info-box { background:#0d1b2a; border-left:4px solid #4fc3f7; border-radius:0 8px 8px 0; padding:.8rem 1rem; color:#a9bdc7; margin:.5rem 0 .8rem 0; }
.warn-box { background:#2a1e0d; border-left:4px solid #ffa726; border-radius:0 8px 8px 0; padding:.8rem 1rem; color:#ffd59b; margin:.5rem 0 .8rem 0; }
.bad-box { background:#2a0d0d; border-left:4px solid #ef5350; border-radius:0 8px 8px 0; padding:.8rem 1rem; color:#ffc0c0; margin:.5rem 0 .8rem 0; }
[data-testid="stSidebar"] { background:#0d1b2a; border-right:1px solid #1a2744; }
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# Estructuras auxiliares
# =============================================================================

@dataclass
class ForcePlate:
    key: str
    name: str
    fx: str
    fy: str
    fz: str
    mx: str
    my: str
    mz: str
    cx: str
    cy: str
    cz: str
    sign_z: float = -1.0


@dataclass
class ContactBlock:
    start: int
    end: int  # índice inclusivo
    duration_s: float
    peak_n: float
    active_plates: str


@dataclass
class JumpEvent:
    pre_contact: ContactBlock
    flight_start: int
    flight_end: int
    landing_contact: ContactBlock
    flight_time_s: float
    takeoff_idx: int
    landing_idx: int


# =============================================================================
# Utilidades generales
# =============================================================================

def section(title: str):
    st.markdown(f'<div class="section-title">◆ {title}</div>', unsafe_allow_html=True)


def info_box(text: str, kind: str = "info"):
    cls = {"info": "info-box", "warn": "warn-box", "bad": "bad-box"}.get(kind, "info-box")
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def kpi(label: str, value, unit: str = "") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        value = "—"
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-unit">{unit}</div>
    </div>
    """


def status_label(status: str) -> str:
    mapping = {
        "OK": '<span class="status-ok">OK</span>',
        "REVISAR": '<span class="status-review">REVISAR</span>',
        "NO VÁLIDO": '<span class="status-bad">NO VÁLIDO</span>',
    }
    return mapping.get(status, status)


def sanitize_name(name: str, fallback: str) -> str:
    clean = re.sub(r"[^0-9a-zA-ZáéíóúÁÉÍÓÚñÑ_]+", "_", str(name).strip())
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean or fallback


def fix_decimal(val):
    """Convierte números normales y algunos formatos Vicon con múltiples puntos."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s in ["", "nan", "NaN", "None"]:
        return np.nan
    s = s.replace(" ", "")
    # Decimal comma sin separador de miles
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        neg = s.startswith("-")
        abs_s = s.lstrip("-")
        parts = abs_s.split(".")
        if len(parts) >= 2 and all(p.isdigit() for p in parts if p != ""):
            try:
                result = float(parts[0] + "." + "".join(parts[1:]))
                return -result if neg else result
            except Exception:
                return np.nan
        return np.nan


def safe_float(v, default=np.nan):
    try:
        if v is None:
            return default
        f = float(v)
        return f if np.isfinite(f) else default
    except Exception:
        return default


def butter_lowpass_safe(data: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if len(arr) < 20 or cutoff <= 0 or cutoff >= fs / 2:
        return arr.copy()
    try:
        b, a = signal.butter(order, cutoff / (0.5 * fs), btype="low", analog=False)
        return signal.filtfilt(b, a, arr)
    except Exception:
        return arr.copy()


def contiguous_regions(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Devuelve bloques True como pares (inicio, fin inclusivo)."""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return []
    diff = np.diff(m.astype(int), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e >= s]


def remove_short_true_blocks(mask: np.ndarray, min_len: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    for s, e in contiguous_regions(out):
        if (e - s + 1) < min_len:
            out[s:e + 1] = False
    return out


def fill_short_false_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    inv_blocks = contiguous_regions(~out)
    for s, e in inv_blocks:
        if s == 0 or e == len(out) - 1:
            continue
        if (e - s + 1) <= max_gap:
            out[s:e + 1] = True
    return out


def nearest_idx_by_time(t: np.ndarray, target: float) -> int:
    if len(t) == 0 or not np.isfinite(target):
        return 0
    return int(np.nanargmin(np.abs(t - target)))


def vec_angle(A, B, C):
    A, B, C = np.asarray(A, dtype=float), np.asarray(B, dtype=float), np.asarray(C, dtype=float)
    if np.any(~np.isfinite(A)) or np.any(~np.isfinite(B)) or np.any(~np.isfinite(C)):
        return np.nan
    v1, v2 = A - B, C - B
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return np.nan
    cos_a = np.dot(v1, v2) / (n1 * n2)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

# =============================================================================
# Parser de fuerza general
# =============================================================================

@st.cache_data(show_spinner=False)
def parse_force_general(file_bytes: bytes) -> Tuple[pd.DataFrame, List[ForcePlate], List[str]]:
    warnings: List[str] = []
    buf = BytesIO(file_bytes)
    header = pd.read_csv(buf, header=None, nrows=3, encoding="utf-8-sig", dtype=str).fillna("")
    h0 = header.iloc[0].tolist() if len(header) > 0 else []

    buf.seek(0)
    raw = pd.read_csv(buf, header=None, skiprows=3, encoding="utf-8-sig", dtype=str)
    raw = raw.dropna(how="all")
    ncols = raw.shape[1]

    if ncols < 11:
        raise ValueError(f"El archivo de fuerza tiene {ncols} columnas. Se esperaban al menos 11 columnas.")

    nplates = max(1, (ncols - 2) // 9)
    expected_cols = 2 + nplates * 9
    if ncols != expected_cols:
        warnings.append(
            f"El número de columnas ({ncols}) no coincide exactamente con 2 + 9×plataformas. "
            f"Se procesarán las primeras {expected_cols} columnas."
        )
        raw = raw.iloc[:, :expected_cols]

    data = raw.apply(lambda col: col.map(fix_decimal))

    columns = ["Frame", "SubFrame"]
    plates: List[ForcePlate] = []
    vars9 = ["Fx", "Fy", "Fz", "Mx", "My", "Mz", "Cx", "Cy", "Cz"]

    for p in range(nplates):
        start_col = 2 + 9 * p
        label = ""
        # Buscar una celda de encabezado cercana con el nombre de plataforma
        for j in range(start_col, min(start_col + 9, len(h0))):
            candidate = str(h0[j]).strip()
            if candidate and candidate.lower() != "nan":
                label = candidate.split(" - ")[0].strip()
                break
        if not label:
            label = f"Plataforma_{p + 1}"
        key = f"P{p + 1}"
        for v in vars9:
            columns.append(f"{key}_{v}")
        plates.append(
            ForcePlate(
                key=key,
                name=label,
                fx=f"{key}_Fx", fy=f"{key}_Fy", fz=f"{key}_Fz",
                mx=f"{key}_Mx", my=f"{key}_My", mz=f"{key}_Mz",
                cx=f"{key}_Cx", cy=f"{key}_Cy", cz=f"{key}_Cz",
            )
        )

    data.columns = columns
    data = data.dropna(subset=["Frame"]).reset_index(drop=True)
    return data, plates, warnings


def prepare_force_signals(
    force_df: pd.DataFrame,
    plates: List[ForcePlate],
    fs_force: float,
    fs_traj: float,
    threshold: float,
    cutoff: float,
    apply_filter: bool,
) -> Tuple[pd.DataFrame, List[ForcePlate], List[str]]:
    f = force_df.copy()
    warnings: List[str] = []

    if "SubFrame" not in f.columns:
        f["SubFrame"] = np.arange(len(f)) % int(max(fs_force / fs_traj, 1))
    f["SubFrame"] = f["SubFrame"].fillna(0)
    first_frame = safe_float(f["Frame"].iloc[0], 0)
    f["time"] = (f["Frame"] - first_frame) / fs_traj + f["SubFrame"] / fs_force
    f["time"] = f["time"] - safe_float(f["time"].iloc[0], 0)

    total_z = np.zeros(len(f), dtype=float)
    total_x = np.zeros(len(f), dtype=float)
    total_y = np.zeros(len(f), dtype=float)

    for plate in plates:
        z_raw = pd.to_numeric(f[plate.fz], errors="coerce").fillna(0).to_numpy(dtype=float)
        # Detección de signo: la dirección dominante de la fuerza vertical durante contacto debe ser positiva.
        max_pos = np.nanpercentile(z_raw, 99) if len(z_raw) else 0
        max_neg = abs(np.nanpercentile(z_raw, 1)) if len(z_raw) else 0
        plate.sign_z = -1.0 if max_neg >= max_pos else 1.0

        grf_z = plate.sign_z * z_raw
        # Eliminar pequeñas oscilaciones negativas después de orientar el signo
        grf_z = np.where(grf_z > 0, grf_z, 0.0)
        f[f"{plate.key}_GRF_z"] = grf_z
        f[f"{plate.key}_contact"] = grf_z > threshold

        # X/Y solo se orientan de forma convencional; para magnitud no es crítico.
        f[f"{plate.key}_GRF_x"] = -pd.to_numeric(f[plate.fx], errors="coerce").fillna(0).to_numpy(dtype=float)
        f[f"{plate.key}_GRF_y"] = -pd.to_numeric(f[plate.fy], errors="coerce").fillna(0).to_numpy(dtype=float)

        total_z += f[f"{plate.key}_GRF_z"].to_numpy(dtype=float)
        total_x += f[f"{plate.key}_GRF_x"].to_numpy(dtype=float)
        total_y += f[f"{plate.key}_GRF_y"].to_numpy(dtype=float)

    f["GRF_z"] = total_z
    f["GRF_x"] = total_x
    f["GRF_y"] = total_y
    f["GRF_mag"] = np.sqrt(total_x ** 2 + total_y ** 2 + total_z ** 2)

    if apply_filter:
        f["GRF_z_filt"] = butter_lowpass_safe(f["GRF_z"].to_numpy(dtype=float), cutoff, fs_force)
        f["GRF_z_filt"] = np.where(f["GRF_z_filt"] > 0, f["GRF_z_filt"], 0.0)
    else:
        f["GRF_z_filt"] = f["GRF_z"]

    f["contact_raw"] = f["GRF_z"] > threshold
    return f, plates, warnings

# =============================================================================
# Detección de eventos
# =============================================================================

def build_contact_blocks(f: pd.DataFrame, plates: List[ForcePlate], contact_mask: np.ndarray, fs_force: float) -> List[ContactBlock]:
    blocks: List[ContactBlock] = []
    for s, e in contiguous_regions(contact_mask):
        active = []
        for plate in plates:
            col = f"{plate.key}_GRF_z"
            if col in f.columns and float(np.nanmax(f.iloc[s:e + 1][col])) > 20:
                active.append(f"{plate.key}:{plate.name}")
        blocks.append(
            ContactBlock(
                start=s,
                end=e,
                duration_s=(e - s + 1) / fs_force,
                peak_n=float(np.nanmax(f.iloc[s:e + 1]["GRF_z"])),
                active_plates=", ".join(active) if active else "—",
            )
        )
    return blocks


def detect_jump_events(
    f: pd.DataFrame,
    plates: List[ForcePlate],
    fs_force: float,
    threshold: float,
    min_contact_ms: float = 30,
    min_flight_ms: float = 80,
    max_gap_ms: float = 20,
) -> Tuple[List[JumpEvent], List[ContactBlock], np.ndarray, List[str]]:
    warnings: List[str] = []
    min_contact = max(1, int(min_contact_ms / 1000 * fs_force))
    min_flight = max(1, int(min_flight_ms / 1000 * fs_force))
    max_gap = max(0, int(max_gap_ms / 1000 * fs_force))

    # Para ubicar despegue/aterrizaje se usa la GRF sin filtrar.
    # El filtro puede adelantar o atrasar cruces de umbral, especialmente en aterrizajes bruscos.
    contact = f["GRF_z"].to_numpy(dtype=float) > threshold
    contact = fill_short_false_gaps(contact, max_gap)
    contact = remove_short_true_blocks(contact, min_contact)

    blocks = build_contact_blocks(f, plates, contact, fs_force)
    events: List[JumpEvent] = []

    if len(blocks) < 2:
        warnings.append("No se encontraron dos contactos separados por vuelo. Puede ser un ensayo incompleto o sin salto.")
        return events, blocks, contact, warnings

    for i in range(len(blocks) - 1):
        pre = blocks[i]
        post = blocks[i + 1]
        flight_start = pre.end + 1
        flight_end = post.start - 1
        flight_len = flight_end - flight_start + 1
        if flight_len >= min_flight:
            events.append(
                JumpEvent(
                    pre_contact=pre,
                    flight_start=flight_start,
                    flight_end=flight_end,
                    landing_contact=post,
                    flight_time_s=flight_len / fs_force,
                    takeoff_idx=pre.end + 1,
                    landing_idx=post.start,
                )
            )

    if not events:
        # Detectar casos incompletos para un mensaje claro
        if len(blocks) >= 1 and blocks[-1].end < len(f) - min_flight and not np.any(contact[blocks[-1].end + 1:]):
            warnings.append("El archivo podría terminar en vuelo o no contener aterrizaje posterior.")
        else:
            warnings.append("Hay contactos, pero no una fase de vuelo suficientemente larga entre ellos.")

    return events, blocks, contact, warnings

# =============================================================================
# Control de calidad y métricas de fuerza
# =============================================================================

def estimate_body_weight(
    f: pd.DataFrame,
    blocks: List[ContactBlock],
    fs_force: float,
    mass_override: Optional[float],
    threshold: float,
) -> Tuple[float, float, str, List[str]]:
    warnings: List[str] = []
    if mass_override is not None and mass_override > 0:
        bw = float(mass_override) * G
        return bw, float(mass_override), "OK", ["Masa corporal ingresada manualmente."]

    if not blocks:
        return np.nan, np.nan, "NO VÁLIDO", ["No se pudo estimar la masa porque no hay contacto con plataforma."]

    candidates = []
    # Buscar ventanas de 300 ms dentro de contactos con baja variabilidad.
    win = max(20, int(0.30 * fs_force))
    for b in blocks:
        segment = f.iloc[b.start:b.end + 1]["GRF_z"].to_numpy(dtype=float)
        if len(segment) < win:
            continue
        for start in range(0, len(segment) - win + 1, max(1, win // 4)):
            w = segment[start:start + win]
            mean = float(np.nanmean(w))
            sd = float(np.nanstd(w))
            if mean > threshold and sd / max(mean, 1) < 0.08:
                candidates.append((sd / max(mean, 1), mean))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        bw = candidates[0][1]
        mass = bw / G
        return bw, mass, "REVISAR", ["Masa estimada automáticamente desde una ventana estable; verificar con la masa real del sujeto."]

    contact_vals = f.loc[f["GRF_z"] > threshold, "GRF_z"].to_numpy(dtype=float)
    if len(contact_vals) > 20:
        bw = float(np.nanmedian(contact_vals))
        mass = bw / G
        warnings.append("Masa estimada con mediana de contactos, sin fase estable clara. Debe revisarse manualmente.")
        return bw, mass, "REVISAR", warnings

    return np.nan, np.nan, "NO VÁLIDO", ["No existe una fase estable para estimar peso corporal. Ingrese la masa manualmente."]


def weighted_cop(f: pd.DataFrame, plates: List[ForcePlate], threshold: float) -> pd.DataFrame:
    n = len(f)
    num_x = np.zeros(n, dtype=float)
    num_y = np.zeros(n, dtype=float)
    den = np.zeros(n, dtype=float)

    for plate in plates:
        z_col = f"{plate.key}_GRF_z"
        if z_col not in f.columns:
            continue
        w = f[z_col].to_numpy(dtype=float)
        cx = pd.to_numeric(f[plate.cx], errors="coerce").to_numpy(dtype=float)
        cy = pd.to_numeric(f[plate.cy], errors="coerce").to_numpy(dtype=float)
        valid = (w > threshold) & np.isfinite(cx) & np.isfinite(cy)
        num_x[valid] += cx[valid] * w[valid]
        num_y[valid] += cy[valid] * w[valid]
        den[valid] += w[valid]

    cop_x = np.full(n, np.nan, dtype=float)
    cop_y = np.full(n, np.nan, dtype=float)
    valid_den = den > 0
    cop_x[valid_den] = num_x[valid_den] / den[valid_den]
    cop_y[valid_den] = num_y[valid_den] / den[valid_den]
    return pd.DataFrame({"CoP_X": cop_x, "CoP_Y": cop_y})


def compute_force_metrics(
    f: pd.DataFrame,
    plates: List[ForcePlate],
    event: JumpEvent,
    blocks: List[ContactBlock],
    fs_force: float,
    threshold: float,
    mass_override: Optional[float],
) -> Dict:
    res: Dict = {"warnings": [], "qc_rows": []}
    bw, mass, mass_status, mass_notes = estimate_body_weight(f, blocks, fs_force, mass_override, threshold)
    res["warnings"].extend(mass_notes)

    to_idx = event.takeoff_idx
    la_idx = event.landing_idx
    contact_start_idx = event.pre_contact.start
    contact_end_idx = event.pre_contact.end
    landing_end_idx = event.landing_contact.end

    flight_time = event.flight_time_s
    h_flight = G * flight_time ** 2 / 8
    v_flight = G * flight_time / 2

    res.update({
        "force_df": f,
        "plates": plates,
        "contact_blocks": blocks,
        "event": event,
        "BW_N": bw,
        "mass_kg": mass,
        "mass_status": mass_status,
        "takeoff_idx": to_idx,
        "landing_idx": la_idx,
        "contact_start_idx": contact_start_idx,
        "contact_end_idx": contact_end_idx,
        "landing_end_idx": landing_end_idx,
        "flight_time_s": flight_time,
        "jump_height_flight_m": h_flight,
        "jump_height_flight_cm": h_flight * 100,
        "v_takeoff_flight_ms": v_flight,
        "peak_GRF_N": float(np.nanmax(f["GRF_z"])),
        "peak_GRF_BW": float(np.nanmax(f["GRF_z"]) / bw) if np.isfinite(bw) and bw > 0 else np.nan,
        "peak_push_GRF_N": float(np.nanmax(f.iloc[contact_start_idx:contact_end_idx + 1]["GRF_z"])),
        "peak_landing_GRF_N": float(np.nanmax(f.iloc[la_idx:landing_end_idx + 1]["GRF_z"])),
    })
    res["peak_landing_GRF_BW"] = res["peak_landing_GRF_N"] / bw if np.isfinite(bw) and bw > 0 else np.nan

    # Impulso y potencia: se calculan, pero se validan contra la velocidad por tiempo de vuelo.
    prop = f.iloc[contact_start_idx:contact_end_idx + 1].copy()
    dt = 1.0 / fs_force
    if np.isfinite(mass) and mass > 0:
        net_force = prop["GRF_z_filt"].to_numpy(dtype=float) - bw
        impulse = float(np.trapezoid(net_force, dx=dt)) if len(net_force) > 1 else 0.0
        v_impulse = impulse / mass
        h_impulse = v_impulse ** 2 / (2 * G) if v_impulse > 0 else np.nan
        res["impulse_Ns"] = impulse
        res["v_takeoff_impulse_ms"] = v_impulse
        res["jump_height_impulse_cm"] = h_impulse * 100 if np.isfinite(h_impulse) else np.nan

        rel_err = abs(v_impulse - v_flight) / max(abs(v_flight), 1e-6)
        impulse_status = "OK" if rel_err <= 0.25 and mass_status == "OK" else "REVISAR"
        if rel_err > 0.25:
            res["warnings"].append(
                f"La velocidad por impulso ({v_impulse:.2f} m/s) difiere de la velocidad por vuelo ({v_flight:.2f} m/s). "
                "Se reporta como principal la velocidad por tiempo de vuelo."
            )

        # Velocidad integrada con corrección lineal para que termine en v_flight.
        acc = net_force / mass
        vel = np.cumsum(acc) * dt
        if len(vel) > 2:
            drift = vel[-1] - v_flight
            vel_corr = vel - np.linspace(0, drift, len(vel))
            power = prop["GRF_z_filt"].to_numpy(dtype=float) * vel_corr
            pos_power = power[power > 0]
            res["vel_arr"] = vel_corr
            res["power_arr"] = power
            res["power_time"] = prop["time"].to_numpy(dtype=float)
            res["peak_power_W"] = float(np.nanmax(power)) if len(power) else np.nan
            res["mean_power_W"] = float(np.nanmean(pos_power)) if len(pos_power) else np.nan
            res["peak_power_W_kg"] = res["peak_power_W"] / mass if np.isfinite(res["peak_power_W"]) else np.nan
            power_status = impulse_status if mass_status == "OK" else "REVISAR"
        else:
            res["vel_arr"] = np.array([])
            res["power_arr"] = np.array([])
            res["power_time"] = np.array([])
            res["peak_power_W"] = np.nan
            res["mean_power_W"] = np.nan
            res["peak_power_W_kg"] = np.nan
            power_status = "NO VÁLIDO"
    else:
        res["impulse_Ns"] = np.nan
        res["v_takeoff_impulse_ms"] = np.nan
        res["jump_height_impulse_cm"] = np.nan
        res["vel_arr"] = np.array([])
        res["power_arr"] = np.array([])
        res["power_time"] = np.array([])
        res["peak_power_W"] = np.nan
        res["mean_power_W"] = np.nan
        res["peak_power_W_kg"] = np.nan
        impulse_status = "NO VÁLIDO"
        power_status = "NO VÁLIDO"

    # CoP ponderado de todas las plataformas
    cop_df = weighted_cop(f, plates, threshold)
    f["CoP_X"] = cop_df["CoP_X"]
    f["CoP_Y"] = cop_df["CoP_Y"]
    cop_prop = f.iloc[contact_start_idx:contact_end_idx + 1][["CoP_X", "CoP_Y"]].dropna()
    if len(cop_prop) > 3:
        res["cop_range_x_mm"] = float(cop_prop["CoP_X"].max() - cop_prop["CoP_X"].min())
        res["cop_range_y_mm"] = float(cop_prop["CoP_Y"].max() - cop_prop["CoP_Y"].min())
        cop_status = "OK"
    else:
        res["cop_range_x_mm"] = np.nan
        res["cop_range_y_mm"] = np.nan
        cop_status = "NO VÁLIDO"

    # RFD sobre contacto pre-despegue
    grf_contact = prop["GRF_z_filt"].to_numpy(dtype=float)
    if len(grf_contact) > int(0.2 * fs_force):
        f0 = grf_contact[0]
        for ms in [50, 100, 200]:
            n = min(int(ms / 1000 * fs_force), len(grf_contact) - 1)
            res[f"RFD_0_{ms}ms"] = float((grf_contact[n] - f0) / (ms / 1000))
    else:
        for ms in [50, 100, 200]:
            res[f"RFD_0_{ms}ms"] = np.nan

    res["qc_rows"] = [
        {"Elemento": "Archivo de fuerza", "Estado": "OK", "Detalle": f"{len(f)} muestras; {len(plates)} plataforma(s) detectada(s)."},
        {"Elemento": "Evento de salto", "Estado": "OK", "Detalle": f"Vuelo de {flight_time*1000:.1f} ms entre dos contactos."},
        {"Elemento": "Masa corporal", "Estado": mass_status, "Detalle": f"{mass:.2f} kg / {bw:.1f} N" if np.isfinite(mass) else "No disponible"},
        {"Elemento": "Altura por tiempo de vuelo", "Estado": "OK", "Detalle": "Métrica principal recomendada."},
        {"Elemento": "Velocidad por tiempo de vuelo", "Estado": "OK", "Detalle": "Métrica principal recomendada."},
        {"Elemento": "Impulso", "Estado": impulse_status, "Detalle": "Revisar si no hubo fase estable o si difiere del método de vuelo."},
        {"Elemento": "Potencia", "Estado": power_status, "Detalle": "Depende de masa y consistencia del impulso."},
        {"Elemento": "Centro de presión", "Estado": cop_status, "Detalle": "CoP ponderado por la fuerza vertical de las plataformas activas."},
    ]

    return res

# =============================================================================
# Parser y procesamiento de trayectoria
# =============================================================================

@st.cache_data(show_spinner=False)
def parse_traj_general(file_bytes: bytes) -> Tuple[pd.DataFrame, List[str]]:
    warnings: List[str] = []
    buf = BytesIO(file_bytes)
    header = pd.read_csv(buf, header=None, nrows=3, encoding="utf-8-sig", dtype=str).fillna("")
    r0 = header.iloc[0].tolist() if len(header) > 0 else []
    r1 = header.iloc[1].tolist() if len(header) > 1 else []

    cols: List[str] = []
    current_marker = None
    for i in range(max(len(r0), len(r1))):
        g = str(r0[i]).strip() if i < len(r0) else ""
        label = str(r1[i]).strip() if i < len(r1) else ""
        if i == 0:
            cols.append("Frame")
            continue
        if i == 1:
            cols.append("SubFrame")
            continue
        if g and g.lower() != "nan":
            current_marker = g.split(":")[-1].strip()
        if label and label.lower() not in ["nan", ""] and current_marker:
            axis = label.strip().upper()
            cols.append(f"{sanitize_name(current_marker, f'M{i}')}_{axis}")
        else:
            cols.append(f"_drop_{i}")

    buf.seek(0)
    raw = pd.read_csv(buf, header=None, skiprows=3, encoding="utf-8-sig", dtype=str)
    raw = raw.dropna(how="all")
    if len(cols) > raw.shape[1]:
        cols = cols[: raw.shape[1]]
    elif len(cols) < raw.shape[1]:
        cols += [f"_extra_{i}" for i in range(raw.shape[1] - len(cols))]

    raw.columns = cols
    df = raw.apply(lambda col: col.map(fix_decimal))
    drop_cols = [c for c in df.columns if c.startswith("_")]
    df = df.drop(columns=drop_cols, errors="ignore")
    df = df.dropna(subset=["Frame"]).reset_index(drop=True)
    return df, warnings


def marker_names(df: pd.DataFrame) -> List[str]:
    markers = set()
    for c in df.columns:
        if c.endswith("_X") or c.endswith("_Y") or c.endswith("_Z"):
            markers.add(c[:-2])
    return sorted(markers)


def clean_trajectory(
    traj_df: pd.DataFrame,
    fs_traj: float,
    first_force_frame: Optional[float],
    max_interp_gap: int,
    treat_zero_as_missing: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    warnings: List[str] = []
    traj = traj_df.copy()
    if "SubFrame" not in traj.columns:
        traj["SubFrame"] = 0

    if first_force_frame is None:
        first_force_frame = safe_float(traj["Frame"].iloc[0], 0)
        warnings.append("No se cargó fuerza para sincronizar; el tiempo de trayectoria empieza en su primer frame.")

    traj["time"] = (traj["Frame"] - first_force_frame) / fs_traj + traj["SubFrame"].fillna(0) / fs_traj

    markers = marker_names(traj)
    quality_rows = []

    for m in markers:
        cols = [f"{m}_X", f"{m}_Y", f"{m}_Z"]
        if not all(c in traj.columns for c in cols):
            continue
        xyz = traj[cols].astype(float)
        # Marcar triples 0,0,0 como pérdida de marcador.
        if treat_zero_as_missing:
            lost = (xyz.abs().sum(axis=1) < 1e-9)
            traj.loc[lost, cols] = np.nan
        valid_before = traj[cols].notna().all(axis=1)
        valid_pct = float(valid_before.mean() * 100) if len(valid_before) else 0
        missing_blocks = contiguous_regions(~valid_before.to_numpy(dtype=bool))
        longest_gap = max([(e - s + 1) for s, e in missing_blocks], default=0)

        for c in cols:
            traj[c] = traj[c].interpolate(method="linear", limit=max_interp_gap, limit_direction="both")

        valid_after = traj[cols].notna().all(axis=1)
        valid_after_pct = float(valid_after.mean() * 100) if len(valid_after) else 0
        status = "OK" if valid_after_pct >= 90 and longest_gap <= max_interp_gap else ("REVISAR" if valid_after_pct >= 70 else "NO VÁLIDO")
        quality_rows.append({
            "Marcador": m,
            "Validez inicial (%)": round(valid_pct, 1),
            "Validez post-limpieza (%)": round(valid_after_pct, 1),
            "Mayor hueco (frames)": int(longest_gap),
            "Estado": status,
        })

    qdf = pd.DataFrame(quality_rows)
    if not qdf.empty and (qdf["Estado"] == "NO VÁLIDO").any():
        warnings.append("Existen marcadores con baja validez; no todas las variables cinemáticas son confiables.")
    return traj, qdf, warnings


def compute_kinematics(traj: pd.DataFrame, qdf: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict], List[str]]:
    warnings: List[str] = []
    out = traj.copy()

    # Centro de masa simplificado: pelvis = promedio LASI/RASI si existen.
    if all(c in out.columns for c in ["LASI_X", "LASI_Y", "LASI_Z", "RASI_X", "RASI_Y", "RASI_Z"]):
        out["CoM_X"] = (out["LASI_X"] + out["RASI_X"]) / 2
        out["CoM_Y"] = (out["LASI_Y"] + out["RASI_Y"]) / 2
        out["CoM_Z"] = (out["LASI_Z"] + out["RASI_Z"]) / 2
        com_valid = out[["CoM_X", "CoM_Y", "CoM_Z"]].notna().all(axis=1).mean() * 100
        com_status = "OK" if com_valid >= 90 else ("REVISAR" if com_valid >= 70 else "NO VÁLIDO")
    else:
        out["CoM_X"] = np.nan
        out["CoM_Y"] = np.nan
        out["CoM_Z"] = np.nan
        com_valid = 0
        com_status = "NO VÁLIDO"
        warnings.append("No se encontraron LASI y RASI completos para estimar el CoM/pelvis.")

    # Ángulos por marcadores disponibles.
    angle_specs = {
        "knee_L": ("LTHI", "LKNE", "LANK"),
        "knee_R": ("RTHI", "RKNE", "RANK"),
        "hip_L": ("LASI", "LTHI", "LKNE"),
        "hip_R": ("RASI", "RTHI", "RKNE"),
        "ankle_L": ("LKNE", "LANK", "LTOE"),
        "ankle_R": ("RKNE", "RANK", "RTOE"),
    }
    angles = pd.DataFrame({"Frame": out["Frame"], "time": out["time"]})
    qc_rows: List[Dict] = [{"Elemento": "Centro de masa/pelvis", "Estado": com_status, "Detalle": f"Validez {com_valid:.1f}%"}]

    for name, (A_m, B_m, C_m) in angle_specs.items():
        needed = [f"{m}_{ax}" for m in [A_m, B_m, C_m] for ax in ["X", "Y", "Z"]]
        if not all(c in out.columns for c in needed):
            angles[name] = np.nan
            qc_rows.append({"Elemento": f"Ángulo {name}", "Estado": "NO VÁLIDO", "Detalle": "Marcadores requeridos no disponibles."})
            continue
        vals = []
        for _, row in out.iterrows():
            A = [row[f"{A_m}_X"], row[f"{A_m}_Y"], row[f"{A_m}_Z"]]
            B = [row[f"{B_m}_X"], row[f"{B_m}_Y"], row[f"{B_m}_Z"]]
            C = [row[f"{C_m}_X"], row[f"{C_m}_Y"], row[f"{C_m}_Z"]]
            vals.append(vec_angle(A, B, C))
        angles[name] = vals
        valid_pct = float(pd.Series(vals).notna().mean() * 100)
        # Filtro para cambios irreales: grandes saltos de un frame a otro.
        jumps = pd.Series(vals).diff().abs()
        big_jumps = int((jumps > 45).sum())
        status = "OK" if valid_pct >= 90 and big_jumps <= 3 else ("REVISAR" if valid_pct >= 70 else "NO VÁLIDO")
        qc_rows.append({"Elemento": f"Ángulo {name}", "Estado": status, "Detalle": f"Validez {valid_pct:.1f}%; saltos bruscos: {big_jumps}"})

    return out, angles, qc_rows, warnings

# =============================================================================
# Gráficas
# =============================================================================

def base_layout(title: str, height: int = 380) -> dict:
    return dict(
        title=dict(text=title, font=dict(color=COLORS["text"], size=14)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"]),
        height=height,
        margin=dict(l=55, r=25, t=50, b=45),
        xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    )


def plot_grf(res: Dict) -> go.Figure:
    f = res["force_df"]
    ev: JumpEvent = res["event"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f["time"], y=f["GRF_z_filt"], name="GRF vertical total", line=dict(color=COLORS["primary"], width=3)))

    # Por plataforma
    for plate in res["plates"]:
        col = f"{plate.key}_GRF_z"
        if col in f.columns and np.nanmax(f[col]) > 20:
            fig.add_trace(go.Scatter(x=f["time"], y=f[col], name=f"{plate.key} {plate.name}", line=dict(width=1.4), opacity=0.6))

    if np.isfinite(res.get("BW_N", np.nan)):
        fig.add_hline(y=res["BW_N"], line_dash="dash", line_color=COLORS["warning"], line_width=2,
                      annotation_text=f"PESO CORPORAL ({res['BW_N']:.0f} N)", annotation_font_color=COLORS["warning"])

    t_contact0 = f["time"].iloc[ev.pre_contact.start]
    t_to = f["time"].iloc[ev.takeoff_idx]
    t_la = f["time"].iloc[ev.landing_idx]
    fig.add_vrect(x0=t_contact0, x1=t_to, fillcolor=COLORS["secondary"], opacity=0.16,
                  annotation_text="Contacto/pre-despegue", annotation_position="top left")
    fig.add_vrect(x0=t_to, x1=t_la, fillcolor=COLORS["accent"], opacity=0.16,
                  annotation_text="VUELO", annotation_position="top")
    fig.add_vline(x=t_to, line_dash="dot", line_color=COLORS["accent"], line_width=3,
                  annotation_text="DESPEGUE", annotation_font_color=COLORS["accent"])
    fig.add_vline(x=t_la, line_dash="dot", line_color="#26ff79", line_width=3,
                  annotation_text="ATERRIZAJE", annotation_font_color="#26ff79")
    fig.update_layout(**base_layout("Fuerza de Reacción del Suelo - Vista General", 405))
    fig.update_xaxes(title_text="Tiempo (s)")
    fig.update_yaxes(title_text="GRF vertical (N)")
    return fig


def plot_power_velocity(res: Dict) -> Optional[go.Figure]:
    if len(res.get("power_arr", [])) == 0:
        return None
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=res["power_time"], y=res["power_arr"], name="Potencia (W)", line=dict(color=COLORS["primary"], width=2.5)), secondary_y=False)
    fig.add_trace(go.Scatter(x=res["power_time"], y=res["vel_arr"], name="Velocidad vertical corregida (m/s)", line=dict(color=COLORS["secondary"], width=2.5, dash="dot")), secondary_y=True)
    fig.update_layout(**base_layout("Potencia y Velocidad del CoM", 350))
    fig.update_xaxes(title_text="Tiempo (s)")
    fig.update_yaxes(title_text="Potencia (W)", secondary_y=False)
    fig.update_yaxes(title_text="Velocidad (m/s)", secondary_y=True)
    return fig


def plot_cop(res: Dict) -> Optional[go.Figure]:
    f = res["force_df"]
    ev: JumpEvent = res["event"]
    seg = f.iloc[ev.pre_contact.start:ev.pre_contact.end + 1].copy()
    seg = seg.dropna(subset=["CoP_X", "CoP_Y"])
    if len(seg) < 3:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=seg["CoP_X"], y=seg["CoP_Y"], mode="lines+markers", name="Trayectoria CoP",
                             line=dict(color=COLORS["primary"], width=1.8), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=[seg["CoP_X"].iloc[0]], y=[seg["CoP_Y"].iloc[0]], mode="markers", name="Inicio", marker=dict(color=COLORS["secondary"], size=10)))
    fig.add_trace(go.Scatter(x=[seg["CoP_X"].iloc[-1]], y=[seg["CoP_Y"].iloc[-1]], mode="markers", name="Final", marker=dict(color=COLORS["accent"], size=10, symbol="x")))
    fig.update_layout(**base_layout("Centro de Presión (CoP)", 350))
    fig.update_xaxes(title_text="CoP X (mm)")
    fig.update_yaxes(title_text="CoP Y (mm)", scaleanchor="x")
    return fig


def plot_angles(angles: pd.DataFrame, side: str) -> Optional[go.Figure]:
    if angles is None or angles.empty:
        return None
    fig = go.Figure()
    labels = {"knee": "Rodilla", "hip": "Cadera", "ankle": "Tobillo"}
    for joint, name in labels.items():
        col = f"{joint}_{side}"
        if col in angles.columns and angles[col].notna().any():
            fig.add_trace(go.Scatter(x=angles["time"], y=angles[col], name=f"{name} ({'Izq.' if side == 'L' else 'Der.'})", line=dict(width=2)))
    if len(fig.data) == 0:
        return None
    fig.update_layout(**base_layout(f"Ángulos Articulares - {'Izquierdo' if side == 'L' else 'Derecho'}", 350))
    fig.update_xaxes(title_text="Tiempo (s)")
    fig.update_yaxes(title_text="Ángulo (°)")
    return fig


def plot_com(traj: pd.DataFrame, res: Dict) -> Optional[go.Figure]:
    if traj is None or traj.empty or "CoM_Z" not in traj.columns or not traj["CoM_Z"].notna().any():
        return None
    f = res["force_df"]
    ev: JumpEvent = res["event"]
    t_to = f["time"].iloc[ev.takeoff_idx]
    t_la = f["time"].iloc[ev.landing_idx]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=traj["time"], y=traj["CoM_Z"], name="CoM/Pelvis Z", line=dict(color=COLORS["primary"], width=3)))
    fig.add_vline(x=t_to, line_dash="dash", line_color=COLORS["accent"], line_width=2, annotation_text="DESPEGUE", annotation_font_color=COLORS["accent"])
    fig.add_vline(x=t_la, line_dash="dash", line_color=COLORS["warning"], line_width=2, annotation_text="ATERRIZAJE", annotation_font_color=COLORS["warning"])
    fig.update_layout(**base_layout("Trayectoria vertical del Centro de Masa/Pelvis", 390))
    fig.update_xaxes(title_text="Tiempo (s)")
    fig.update_yaxes(title_text="Altura (mm)")
    return fig

# =============================================================================
# Interfaz principal
# =============================================================================

def format_num(x, decimals=2):
    if x is None or not np.isfinite(safe_float(x)):
        return "—"
    return f"{float(x):.{decimals}f}"


def main():
    st.markdown(
        """
        <div class="hero-header">
            <h1>Plataforma General de Análisis Biomecánico de Saltos</h1>
            <p>Vicon Nexus / Force Plates | Fuerza + trayectoria | Detección automática con control de calidad</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("## ⚙️ Configuración")
        force_file = st.file_uploader("Archivo de FUERZA.csv", type=["csv"], key="force")
        traj_file = st.file_uploader("Archivo de TRAYECTORIA.csv (opcional)", type=["csv"], key="traj")
        st.markdown("---")
        mass_input = st.number_input("Masa corporal real (kg). Dejar 0 para estimar", min_value=0.0, max_value=250.0, value=0.0, step=0.5)
        fs_force = st.number_input("Frecuencia de fuerza (Hz)", min_value=100, max_value=5000, value=1000, step=100)
        fs_traj = st.number_input("Frecuencia de trayectoria / cámara (Hz)", min_value=25, max_value=500, value=100, step=10)
        threshold = st.slider("Umbral de contacto GRF (N)", 5, 150, 20)
        cutoff = st.slider("Filtro GRF paso bajo (Hz)", 5, 150, 50)
        apply_filter = st.checkbox("Aplicar filtro a GRF", value=True)
        st.markdown("---")
        min_flight_ms = st.slider("Vuelo mínimo válido (ms)", 50, 300, 80)
        min_contact_ms = st.slider("Contacto mínimo válido (ms)", 10, 100, 30)
        st.markdown("---")
        treat_zero = st.checkbox("Trayectoria: tratar marcador 0,0,0 como perdido", value=True)
        max_interp_gap = st.slider("Interpolación máxima de huecos (frames)", 0, 30, 10)
        run = st.button("▶ CALCULAR ANÁLISIS", use_container_width=True, type="primary")

    if not run:
        info_box(
            "Sube al menos el archivo de fuerza.",
            "info",
        )
        return

    if force_file is None:
        st.error("Debes cargar el archivo de fuerza para continuar.")
        return

    try:
        force_bytes = force_file.read()
        raw_force, plates, parse_warnings = parse_force_general(force_bytes)
        f, plates, sig_warnings = prepare_force_signals(
            raw_force, plates,
            fs_force=float(fs_force), fs_traj=float(fs_traj),
            threshold=float(threshold), cutoff=float(cutoff), apply_filter=bool(apply_filter),
        )
        events, blocks, contact_mask, event_warnings = detect_jump_events(
            f, plates, fs_force=float(fs_force), threshold=float(threshold),
            min_contact_ms=float(min_contact_ms), min_flight_ms=float(min_flight_ms), max_gap_ms=20,
        )
    except Exception as e:
        st.error(f"Error procesando fuerza: {e}")
        st.exception(e)
        return

    all_warnings = parse_warnings + sig_warnings + event_warnings

    if not events:
        section("Control de calidad")
        for w in all_warnings:
            info_box(w, "warn")
        st.plotly_chart(go.Figure(go.Scatter(x=f.get("time", np.arange(len(f))), y=f.get("GRF_z", np.zeros(len(f))), name="GRF_z")), use_container_width=True, key="qc_no_events_grf")
        st.dataframe(pd.DataFrame([b.__dict__ for b in blocks]), use_container_width=True)
        return

    # Selección de evento si hay varios saltos
    if len(events) > 1:
        labels = [f"Evento {i+1}: vuelo {ev.flight_time_s*1000:.1f} ms | contacto {ev.pre_contact.active_plates} → {ev.landing_contact.active_plates}" for i, ev in enumerate(events)]
        selected_label = st.selectbox("Se detectaron varios eventos. Selecciona el salto a analizar:", labels, index=int(np.argmax([ev.flight_time_s for ev in events])))
        event = events[labels.index(selected_label)]
    else:
        event = events[0]

    res = compute_force_metrics(
        f, plates, event, blocks,
        fs_force=float(fs_force), threshold=float(threshold),
        mass_override=mass_input if mass_input > 0 else None,
    )
    all_warnings.extend(res.get("warnings", []))

    traj_clean = None
    angles_df = None
    marker_qdf = pd.DataFrame()
    kine_qc_rows: List[Dict] = []
    if traj_file is not None:
        try:
            traj_bytes = traj_file.read()
            raw_traj, traj_warnings = parse_traj_general(traj_bytes)
            traj_clean, marker_qdf, clean_warnings = clean_trajectory(
                raw_traj, fs_traj=float(fs_traj),
                first_force_frame=safe_float(raw_force["Frame"].iloc[0]),
                max_interp_gap=int(max_interp_gap), treat_zero_as_missing=bool(treat_zero),
            )
            traj_clean, angles_df, kine_qc_rows, kin_warnings = compute_kinematics(traj_clean, marker_qdf)
            all_warnings.extend(traj_warnings + clean_warnings + kin_warnings)
        except Exception as e:
            all_warnings.append(f"No se pudo procesar trayectoria: {e}")

    # Tabs
    tabs = st.tabs(["📊 Resumen", "✅ Control de calidad", "⚡ Fuerzas", "🦵 Cinemática", "📋 Datos"])

    with tabs[0]:
        section("Métricas principales del salto")
        c = st.columns(4)
        c[0].markdown(kpi("Altura de salto", format_num(res["jump_height_flight_cm"], 2), "cm"), unsafe_allow_html=True)
        c[1].markdown(kpi("Tiempo de vuelo", format_num(res["flight_time_s"] * 1000, 1), "ms"), unsafe_allow_html=True)
        c[2].markdown(kpi("Velocidad despegue", format_num(res["v_takeoff_flight_ms"], 3), "m/s por vuelo"), unsafe_allow_html=True)
        c[3].markdown(kpi("Potencia pico", format_num(res.get("peak_power_W"), 1), "W"), unsafe_allow_html=True)
        c2 = st.columns(4)
        c2[0].markdown(kpi("GRF pico", format_num(res["peak_GRF_N"], 1), "N"), unsafe_allow_html=True)
        c2[1].markdown(kpi("GRF pico / PC", format_num(res["peak_GRF_BW"], 2), "× BW"), unsafe_allow_html=True)
        c2[2].markdown(kpi("Peso corporal", format_num(res["BW_N"], 1), "N"), unsafe_allow_html=True)
        c2[3].markdown(kpi("Masa", format_num(res["mass_kg"], 1), "kg"), unsafe_allow_html=True)

        section("Vista general de fuerza")
        st.plotly_chart(plot_grf(res), use_container_width=True, key="summary_grf_overview")

        section("Tabla de variables")
        summary = pd.DataFrame([
            ["Tiempo de vuelo", "ms", res["flight_time_s"] * 1000, "OK"],
            ["Altura por tiempo de vuelo", "cm", res["jump_height_flight_cm"], "OK"],
            ["Velocidad por tiempo de vuelo", "m/s", res["v_takeoff_flight_ms"], "OK"],
            ["Velocidad por impulso", "m/s", res.get("v_takeoff_impulse_ms"), next((r["Estado"] for r in res["qc_rows"] if r["Elemento"] == "Impulso"), "REVISAR")],
            ["Altura por impulso", "cm", res.get("jump_height_impulse_cm"), next((r["Estado"] for r in res["qc_rows"] if r["Elemento"] == "Impulso"), "REVISAR")],
            ["Impulso", "N·s", res.get("impulse_Ns"), next((r["Estado"] for r in res["qc_rows"] if r["Elemento"] == "Impulso"), "REVISAR")],
            ["GRF pico total", "N", res["peak_GRF_N"], "OK"],
            ["GRF pico / PC", "× BW", res["peak_GRF_BW"], res["mass_status"]],
            ["GRF aterrizaje pico", "N", res["peak_landing_GRF_N"], "OK"],
            ["GRF aterrizaje / PC", "× BW", res["peak_landing_GRF_BW"], res["mass_status"]],
            ["Potencia pico", "W", res.get("peak_power_W"), next((r["Estado"] for r in res["qc_rows"] if r["Elemento"] == "Potencia"), "REVISAR")],
            ["Potencia relativa", "W/kg", res.get("peak_power_W_kg"), next((r["Estado"] for r in res["qc_rows"] if r["Elemento"] == "Potencia"), "REVISAR")],
            ["CoP rango X", "mm", res.get("cop_range_x_mm"), next((r["Estado"] for r in res["qc_rows"] if r["Elemento"] == "Centro de presión"), "REVISAR")],
            ["CoP rango Y", "mm", res.get("cop_range_y_mm"), next((r["Estado"] for r in res["qc_rows"] if r["Elemento"] == "Centro de presión"), "REVISAR")],
        ], columns=["Variable", "Unidad", "Resultado", "Estado"])
        summary["Resultado"] = summary["Resultado"].apply(lambda x: format_num(x, 3) if x is not None else "—")
        st.dataframe(summary, use_container_width=True, hide_index=True)

    with tabs[1]:
        section("Control de calidad del ensayo")
        qc_rows = res["qc_rows"] + kine_qc_rows
        qc_df = pd.DataFrame(qc_rows)
        if not qc_df.empty:
            qc_display = qc_df.copy()
            st.dataframe(qc_display, use_container_width=True, hide_index=True)
        if all_warnings:
            section("Advertencias")
            # Evitar mensajes repetidos
            for w in list(dict.fromkeys([str(x) for x in all_warnings if str(x).strip()])):
                kind = "bad" if "No se" in w or "no se" in w else "warn"
                info_box(w, kind)
        section("Contactos detectados")
        blocks_df = pd.DataFrame([{
            "Bloque": i + 1,
            "Inicio (s)": f["time"].iloc[b.start],
            "Fin (s)": f["time"].iloc[b.end],
            "Duración (ms)": b.duration_s * 1000,
            "Pico (N)": b.peak_n,
            "Plataformas activas": b.active_plates,
        } for i, b in enumerate(blocks)])
        st.dataframe(blocks_df.round(3), use_container_width=True, hide_index=True)

        if not marker_qdf.empty:
            section("Calidad de marcadores")
            st.dataframe(marker_qdf, use_container_width=True, hide_index=True)

    with tabs[2]:
        section("Análisis detallado de fuerzas")
        st.plotly_chart(plot_grf(res), use_container_width=True, key="forces_grf_detail")
        col1, col2 = st.columns(2)
        with col1:
            fig_pv = plot_power_velocity(res)
            if fig_pv is not None:
                st.plotly_chart(fig_pv, use_container_width=True, key="forces_power_velocity")
            else:
                info_box("Potencia no disponible o no confiable para este ensayo.", "warn")
        with col2:
            fig_cop = plot_cop(res)
            if fig_cop is not None:
                st.plotly_chart(fig_cop, use_container_width=True, key="forces_cop")
            else:
                info_box("CoP no disponible durante el contacto pre-despegue.", "warn")

        section("Componentes de fuerza")
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatter(x=f["time"], y=f["GRF_z"], name="GRF Z total", line=dict(color=COLORS["primary"], width=2)))
        fig_comp.add_trace(go.Scatter(x=f["time"], y=f["GRF_x"], name="GRF X", line=dict(width=1.5)))
        fig_comp.add_trace(go.Scatter(x=f["time"], y=f["GRF_y"], name="GRF Y", line=dict(width=1.5)))
        fig_comp.update_layout(**base_layout("GRF tridireccional", 350))
        fig_comp.update_xaxes(title_text="Tiempo (s)")
        fig_comp.update_yaxes(title_text="Fuerza (N)")
        st.plotly_chart(fig_comp, use_container_width=True, key="forces_components")

    with tabs[3]:
        section("Cinemática")
        if traj_clean is None or angles_df is None:
            info_box("No se cargó trayectoria o no pudo procesarse. El análisis de fuerza sigue siendo válido.", "warn")
        else:
            col1, col2 = st.columns(2)
            with col1:
                fig_l = plot_angles(angles_df, "L")
                if fig_l is not None:
                    st.plotly_chart(fig_l, use_container_width=True, key="kinematics_angles_left")
                else:
                    info_box("No hay ángulos izquierdos válidos.", "warn")
            with col2:
                fig_r = plot_angles(angles_df, "R")
                if fig_r is not None:
                    st.plotly_chart(fig_r, use_container_width=True, key="kinematics_angles_right")
                else:
                    info_box("No hay ángulos derechos válidos.", "warn")
            fig_com = plot_com(traj_clean, res)
            if fig_com is not None:
                st.plotly_chart(fig_com, use_container_width=True, key="kinematics_com")
            else:
                info_box("CoM/Pelvis no válido por falta de LASI/RASI o datos perdidos.", "warn")

            section("Estadísticas de ángulos")
            rows = []
            names = {"knee_L": "Rodilla Izq.", "knee_R": "Rodilla Der.", "hip_L": "Cadera Izq.", "hip_R": "Cadera Der.", "ankle_L": "Tobillo Izq.", "ankle_R": "Tobillo Der."}
            for col, label in names.items():
                if col in angles_df.columns and angles_df[col].notna().any():
                    v = angles_df[col].dropna()
                    rows.append({"Variable": label, "Mín (°)": v.min(), "Máx (°)": v.max(), "Media (°)": v.mean(), "Rango (°)": v.max() - v.min()})
            st.dataframe(pd.DataFrame(rows).round(2), use_container_width=True, hide_index=True)

    with tabs[4]:
        section("Datos procesados")
        dataset = st.selectbox("Seleccionar tabla", ["Fuerza procesada", "Trayectoria limpia", "Ángulos", "Calidad de marcadores"])
        if dataset == "Fuerza procesada":
            cols = [c for c in ["time", "Frame", "SubFrame", "GRF_x", "GRF_y", "GRF_z", "GRF_z_filt", "CoP_X", "CoP_Y"] if c in f.columns]
            st.dataframe(f[cols].head(500).round(4), use_container_width=True)
            csv = f[cols].to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Descargar fuerza procesada", data=csv, file_name="fuerza_procesada.csv", mime="text/csv")
        elif dataset == "Trayectoria limpia":
            if traj_clean is not None:
                st.dataframe(traj_clean.head(500).round(4), use_container_width=True)
                csv = traj_clean.to_csv(index=False).encode("utf-8")
                st.download_button("⬇ Descargar trayectoria limpia", data=csv, file_name="trayectoria_limpia.csv", mime="text/csv")
            else:
                st.warning("No disponible.")
        elif dataset == "Ángulos":
            if angles_df is not None:
                st.dataframe(angles_df.head(500).round(4), use_container_width=True)
                csv = angles_df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇ Descargar ángulos", data=csv, file_name="angulos.csv", mime="text/csv")
            else:
                st.warning("No disponible.")
        else:
            if not marker_qdf.empty:
                st.dataframe(marker_qdf, use_container_width=True)
            else:
                st.warning("No disponible.")


if __name__ == "__main__":
    main()
