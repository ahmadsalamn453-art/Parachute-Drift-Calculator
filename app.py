"""
SEDS Mission 05: Parachute Wind Drift & Landing Zone Command Center
Candidate: Ahmad Salman (SEDS-TASK-RCK-SRILANKA-020)
High-Tech Aerospace Telemetry UI (Streamlit + Plotly + Custom CSS)
"""

import math
from typing import Optional
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ------------------------------------------------------------------ #
# 1. PAGE CONFIGURATION & AEROSPACE DARK THEME CSS
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="SEDS Command Center | Mission 05",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Cyber-Aerospace / Skunkworks Interface
st.markdown("""
<style>
    /* Dark Theme Background */
    .stApp {
        background-color: #0A0D14;
        color: #E2E8F0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0F1420 !important;
        border-right: 1px solid #1E293B;
    }
    
    /* Header Banner */
    .header-box {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 60%, #311B92 100%);
        border: 1px solid #3730A3;
        border-radius: 14px;
        padding: 22px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }
    .header-title {
        font-size: 1.85rem;
        font-weight: 800;
        color: #38BDF8;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .header-subtitle {
        color: #94A3B8;
        font-size: 0.92rem;
        margin-top: 6px;
    }
    
    /* Custom Glassmorphism Cards */
    .telemetry-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
        margin-bottom: 12px;
    }
    .card-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748B;
        margin-bottom: 6px;
    }
    .card-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #F8FAFC;
    }
    .card-sub {
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 4px;
    }

    /* Status Pill Badges */
    .status-pass {
        color: #00E676;
        background: rgba(0, 230, 118, 0.12);
        border: 1px solid rgba(0, 230, 118, 0.35);
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .status-fail {
        color: #FF1744;
        background: rgba(255, 23, 68, 0.12);
        border: 1px solid rgba(255, 23, 68, 0.35);
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ #
# 2. CORE MATHEMATICAL DRIFT ENGINE
# ------------------------------------------------------------------ #
COMPASS_8 = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
FT_TO_M = 0.3048


def compass_point(bearing_deg: float) -> str:
    """Nearest 8-point compass label for a bearing in degrees."""
    return COMPASS_8[int(round((bearing_deg % 360.0) / 45.0)) % 8]


class ParachuteDriftCalculator:
    def __init__(
        self,
        safe_radius_m: float = 400.0,
        baseline_descent_ms: float = 5.2,
        drogue_descent_ms: Optional[float] = None,
        main_deploy_agl_m: Optional[float] = None,
        shear_exponent: float = 0.0,
        wind_ref_height_m: float = 10.0,
        safety_factor: float = 1.0,
        n_steps: int = 1000,
    ):
        self.safe_radius_m = float(safe_radius_m)
        self.baseline_descent_ms = float(baseline_descent_ms)
        self.drogue_descent_ms = None if drogue_descent_ms is None else float(drogue_descent_ms)
        self.main_deploy_agl_m = None if main_deploy_agl_m is None else float(main_deploy_agl_m)
        self.shear_exponent = float(shear_exponent)
        self.wind_ref_height_m = float(wind_ref_height_m)
        self.safety_factor = float(safety_factor)
        self.n_steps = int(n_steps)

    def _wind_factor(self, h_agl: float) -> float:
        if self.shear_exponent == 0.0:
            return 1.0
        return (max(h_agl, self.wind_ref_height_m) / self.wind_ref_height_m) ** self.shear_exponent

    def _phases(self, effective_alt_m: float):
        if self.drogue_descent_ms is None or effective_alt_m <= self.main_deploy_agl_m:
            return [(0.0, effective_alt_m, self.baseline_descent_ms)]
        return [
            (0.0, self.main_deploy_agl_m, self.baseline_descent_ms),
            (self.main_deploy_agl_m, effective_alt_m, self.drogue_descent_ms),
        ]

    def _integrate(self, effective_alt_m: float, wind_speed_ms: float):
        fall_time = 0.0
        drift = 0.0
        for h_low, h_high, speed in self._phases(effective_alt_m):
            span = h_high - h_low
            if span <= 0:
                continue
            dh = span / self.n_steps
            dt = dh / speed
            for i in range(self.n_steps):
                h_mid = h_low + (i + 0.5) * dh
                fall_time += dt
                drift += wind_speed_ms * self._wind_factor(h_mid) * dt
        return fall_time, drift

    def calculate_descent(
        self,
        deployment_alt_m: float,
        ground_elevation_m: float,
        wind_speed_ms: float,
        wind_dir_deg: float,
    ) -> dict:
        effective_alt_m = deployment_alt_m - ground_elevation_m
        fall_duration_s, total_drift_m = self._integrate(effective_alt_m, wind_speed_ms)

        drift_bearing_deg = (wind_dir_deg + 180.0) % 360.0
        drift_rad = math.radians(drift_bearing_deg)
        target_x = total_drift_m * math.sin(drift_rad)
        target_y = total_drift_m * math.cos(drift_rad)

        design_drift_m = total_drift_m * self.safety_factor

        return {
            "effective_alt_m": effective_alt_m,
            "fall_duration_s": fall_duration_s,
            "total_drift_m": total_drift_m,
            "design_drift_m": design_drift_m,
            "target_x": target_x,
            "target_y": target_y,
            "wind_speed_ms": float(wind_speed_ms),
            "wind_from_bearing_deg": wind_dir_deg % 360.0,
            "wind_from_compass": compass_point(wind_dir_deg),
            "drift_bearing_deg": drift_bearing_deg,
            "drift_compass": compass_point(drift_bearing_deg),
            "landing_bearing": f"{drift_bearing_deg:.0f}° {compass_point(drift_bearing_deg)}",
            "margin_m": self.safe_radius_m - design_drift_m,
            "is_compliant": design_drift_m <= self.safe_radius_m,
            "safe_radius_m": self.safe_radius_m,
        }

    def max_compliant_altitude_agl(self, wind_speed_ms: float) -> Optional[float]:
        def design_drift(alt: float) -> float:
            return self._integrate(alt, wind_speed_ms)[1] * self.safety_factor

        floor_alt = self.main_deploy_agl_m if self.main_deploy_agl_m else 1.0
        if design_drift(floor_alt) > self.safe_radius_m:
            return None
        ceiling = 50000.0
        if design_drift(ceiling) <= self.safe_radius_m:
            return float("inf")
        lo, hi = floor_alt, ceiling
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if design_drift(mid) <= self.safe_radius_m:
                lo = mid
            else:
                hi = mid
        return lo


# ------------------------------------------------------------------ #
# 3. SIDEBAR CONTROLS
# ------------------------------------------------------------------ #
st.sidebar.markdown("### Flight & Recovery Specs")

apogee_agl = st.sidebar.number_input("Apogee Altitude (m AGL)", min_value=100.0, max_value=2000.0, value=300.0, step=10.0)
drogue_speed = st.sidebar.number_input("Drogue Speed (m/s)", min_value=5.0, max_value=50.0, value=25.0, step=1.0)
main_deploy_ft = st.sidebar.number_input("Main Deploy Alt (ft AGL)", min_value=100.0, max_value=2000.0, value=500.0, step=50.0)
main_speed = st.sidebar.number_input("Main Speed (m/s)", min_value=2.0, max_value=15.0, value=5.2, step=0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("### Wind & Safety Parameters")
wind_dir = st.sidebar.slider("Wind Vector (deg From True N)", min_value=0, max_value=360, value=135, step=5)
safe_radius = st.sidebar.number_input("Safe Boundary Radius (m)", min_value=100.0, max_value=1000.0, value=400.0, step=25.0)
shear_exp = st.sidebar.slider("Boundary Layer Shear Exponent (p)", min_value=0.0, max_value=0.25, value=0.0, step=0.01)
safety_factor = st.sidebar.number_input("Safety Factor", min_value=1.0, max_value=2.0, value=1.0, step=0.05)

main_deploy_m = main_deploy_ft * FT_TO_M

calc = ParachuteDriftCalculator(
    safe_radius_m=safe_radius,
    baseline_descent_ms=main_speed,
    drogue_descent_ms=drogue_speed,
    main_deploy_agl_m=main_deploy_m,
    shear_exponent=shear_exp,
    safety_factor=safety_factor,
)

res_5 = calc.calculate_descent(apogee_agl, 0.0, 5.0, float(wind_dir))
res_10 = calc.calculate_descent(apogee_agl, 0.0, 10.0, float(wind_dir))

limit_5 = calc.max_compliant_altitude_agl(5.0)
limit_10 = calc.max_compliant_altitude_agl(10.0)

# ------------------------------------------------------------------ #
# 4. DASHBOARD HEADER & TELEMETRY CARDS
# ------------------------------------------------------------------ #
st.markdown("""
<div class="header-box">
    <div class="header-title">SEDS MISSION 05: LANDING ZONE COMMAND CENTER</div>
    <div class="header-subtitle">Dual-Deploy Parachute Wind Drift Analysis • SEDS-TASK-RCK-SRILANKA-020 • Lead: Ahmad Salman (Pod 1C)</div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="telemetry-card">
        <div class="card-label">Fall Duration</div>
        <div class="card-value">{res_5['fall_duration_s']:.1f}s</div>
        <div class="card-sub" style="color: #38BDF8;">Main Open: {main_deploy_m:.1f}m</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    status_color_5 = "#00E676" if res_5["is_compliant"] else "#FF1744"
    st.markdown(f"""
    <div class="telemetry-card">
        <div class="card-label">5 m/s Nominal Drift</div>
        <div class="card-value" style="color: {status_color_5};">{res_5['total_drift_m']:.1f}m</div>
        <div class="card-sub" style="color: #94A3B8;">Margin: +{res_5['margin_m']:.1f}m</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    status_color_10 = "#00E676" if res_10["is_compliant"] else "#FF1744"
    st.markdown(f"""
    <div class="telemetry-card">
        <div class="card-label">10 m/s Gust Drift</div>
        <div class="card-value" style="color: {status_color_10};">{res_10['total_drift_m']:.1f}m</div>
        <div class="card-sub" style="color: #94A3B8;">Margin: {'+' if res_10['margin_m']>=0 else ''}{res_10['margin_m']:.1f}m</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="telemetry-card">
        <div class="card-label">Drift Trajectory</div>
        <div class="card-value" style="color: #A855F7;">{res_5['drift_compass']}</div>
        <div class="card-sub" style="color: #94A3B8;">Bearing: {res_5['landing_bearing']}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ------------------------------------------------------------------ #
# 5. PLOTLY RADAR / TARGET GRAPH & VERIFICATION PANEL
# ------------------------------------------------------------------ #
col_graph, col_info = st.columns([3, 2])

with col_graph:
    st.subheader("Landing Zone Radar Scope")

    fig = go.Figure()

    # Outer 400m Safe Radius Circle
    theta = [i * (2 * math.pi / 120) for i in range(121)]
    circle_x = [safe_radius * math.cos(t) for t in theta]
    circle_y = [safe_radius * math.sin(t) for t in theta]

    fig.add_trace(go.Scatter(
        x=circle_x, y=circle_y,
        mode="lines",
        name=f"Safe Recovery Zone ({safe_radius:.0f}m)",
        line=dict(color="#FF1744", width=2, dash="dash")
    ))

    # Concentric Grid Rings
    for ring in [100, 200, 300]:
        if ring < safe_radius:
            rx = [ring * math.cos(t) for t in theta]
            ry = [ring * math.sin(t) for t in theta]
            fig.add_trace(go.Scatter(
                x=rx, y=ry, mode="lines",
                showlegend=False,
                line=dict(color="rgba(148, 163, 184, 0.15)", width=1, dash="dot")
            ))

    # Launch Pad Center
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode="markers",
        name="Launch Pad (0,0)",
        marker=dict(color="#38BDF8", size=14, symbol="cross-thin", line=dict(width=3, color="#38BDF8"))
    ))

    # 5 m/s Impact Coordinate
    fig.add_trace(go.Scatter(
        x=[res_5["target_x"]], y=[res_5["target_y"]],
        mode="markers+text",
        name="5 m/s Nominal Wind",
        text=["5 m/s Impact"],
        textposition="top center",
        marker=dict(color="#00E676" if res_5["is_compliant"] else "#FF1744", size=16, symbol="circle")
    ))

    # 10 m/s Impact Coordinate
    fig.add_trace(go.Scatter(
        x=[res_10["target_x"]], y=[res_10["target_y"]],
        mode="markers+text",
        name="10 m/s Wind Gust",
        text=["10 m/s Impact"],
        textposition="top center",
        marker=dict(color="#3B82F6" if res_10["is_compliant"] else "#F59E0B", size=16, symbol="diamond")
    ))

    # Dark Chart Styling
    max_range = max(safe_radius * 1.25, res_10["total_drift_m"] * 1.15)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        font=dict(color="#94A3B8"),
        xaxis=dict(
            title="East Offset (m)",
            range=[-max_range, max_range],
            gridcolor="rgba(30, 41, 59, 0.8)",
            zerolinecolor="#334155",
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            title="North Offset (m)",
            range=[-max_range, max_range],
            gridcolor="rgba(30, 41, 59, 0.8)",
            zerolinecolor="#334155"
        ),
        height=520,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11)
        )
    )

    st.plotly_chart(fig, use_container_width=True)

with col_info:
    st.subheader("Range Safety Audit")

    st.markdown("#### 5 m/s Nominal Case")
    if res_5["is_compliant"]:
        st.markdown(f'<span class="status-pass">COMPLIANT ({res_5["total_drift_m"]:.1f}m / {safe_radius:.0f}m)</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="status-fail">NON-COMPLIANT ({res_5["total_drift_m"]:.1f}m > {safe_radius:.0f}m)</span>', unsafe_allow_html=True)

    st.caption(f"**Max Compliant Apogee Limit:** `{limit_5:.0f} m AGL`" if limit_5 else "**Max Compliant Apogee:** `None`")

    st.markdown("#### 10 m/s Gust Case")
    if res_10["is_compliant"]:
        st.markdown(f'<span class="status-pass">COMPLIANT ({res_10["total_drift_m"]:.1f}m / {safe_radius:.0f}m)</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="status-fail">BOUNDARY BREACH ({res_10["total_drift_m"]:.1f}m > {safe_radius:.0f}m)</span>', unsafe_allow_html=True)

    st.caption(f"**Max Compliant Apogee Limit:** `{limit_10:.0f} m AGL`" if limit_10 else "**Max Compliant Apogee:** `None`")

    st.markdown("---")
    st.markdown("#### Simulation Telemetry Matrix")

    df_comp = pd.DataFrame([
        {
            "Wind Case": "5 m/s Nominal",
            "Fall Time": f"{res_5['fall_duration_s']:.1f} s",
            "Drift": f"{res_5['total_drift_m']:.1f} m",
            "Offset (E, N)": f"({res_5['target_x']:.0f}m, {res_5['target_y']:.0f}m)",
            "Status": "PASS" if res_5["is_compliant"] else "FAIL",
        },
        {
            "Wind Case": "10 m/s Gust",
            "Fall Time": f"{res_10['fall_duration_s']:.1f} s",
            "Drift": f"{res_10['total_drift_m']:.2f} m",
            "Offset (E, N)": f"({res_10['target_x']:.0f}m, {res_10['target_y']:.0f}m)",
            "Status": "PASS" if res_10["is_compliant"] else "FAIL",
        },
    ])
    st.dataframe(df_comp, hide_index=True, use_container_width=True)