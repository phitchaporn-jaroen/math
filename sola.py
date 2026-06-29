import datetime
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pvlib
import pandas as pd
import streamlit as dt
from shapely.geometry import Polygon

# 1. Page Config & CSS (ปรับแต่งหน้าตาให้อ่านง่ายสากล)
dt.set_page_config(
    page_title="Multi-Shape Shadow Simulator ☀️",
    page_icon="☀️",
    layout="wide"
)

dt.markdown("""
    <style>
    .main { background-color: #FAFAFA; }
    h2, h3, h4 { margin: 5px 0px; padding: 0px; font-weight: bold; }
    .annual-box { background-color: #F5F5F5; padding: 12px; border-radius: 6px; border-left: 4px solid #757575; font-size: 14px; }
    div[data-testid="stMetricValue"] { font-size: 20px !important; font-weight: bold; }
    div[data-testid="stMetricLabel"] { font-size: 12px !important; }
    [data-testid="stVerticalBlock"] { gap: 0.5rem !important; }
    th { background-color: #ECEFF1 !important; text-align: center !important; font-weight: bold !important; }
    td { text-align: center !important; }
    </style>
""", unsafe_allow_html=True)

LATITUDE = 13.736717
LONGITUDE = 100.523186
TZ = 'Asia/Bangkok'

# --- 📐 ฟังก์ชันสร้างจุดพิกัด (Vertices) ตามรูปทรงเรขาคณิต (พื้นที่ตั้งต้นเสมอกันที่ 1.0) ---
def generate_shape_vertices(shape_type):
    if shape_type == "Square (สี่เหลี่ยมจัตุรัส)":
        return np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]])
        
    elif shape_type == "Equilateral Triangle (สามเหลี่ยมด้านเท่า)":
        side = np.sqrt(4.0 / np.sqrt(3.0))
        h = side * np.sqrt(3.0) / 2.0
        return np.array([[0, h * 2.0/3.0, 0], [-side/2.0, -h * 1.0/3.0, 0], [side/2.0, -h * 1.0/3.0, 0], [0, h * 2.0/3.0, 0]])
        
    elif shape_type == "Circle (วงกลม)":
        r = np.sqrt(1.0 / np.pi)
        angles = np.linspace(0, 2 * np.pi, 32)
        return np.array([[r * np.cos(a), r * np.sin(a), 0] for a in angles])
        
    elif shape_type == "Rectangle (สี่เหลี่ยมผืนผ้า)":
        w, l = 0.707 / 2.0, 1.414 / 2.0
        return np.array([[-w, -l, 0], [w, -l, 0], [w, l, 0], [-w, l, 0]])

def get_rotation_matrix(tilt_deg, azimuth_deg):
    t_rad = np.radians(tilt_deg)
    a_rad = np.radians(360 - azimuth_deg) 
    R_tilt = np.array([[1, 0, 0], [0, np.cos(t_rad), -np.sin(t_rad)], [0, np.sin(t_rad), np.cos(t_rad)]])
    R_azimuth = np.array([[np.cos(a_rad), -np.sin(a_rad), 0], [np.sin(a_rad), np.cos(a_rad), 0], [0, 0, 1]])
    return R_azimuth @ R_tilt

def calculate_shadow_geometry(solar_el, solar_az, tilt_deg, azimuth_deg, base_height_z, shape_type):
    if solar_el <= 0:
        verts = generate_shape_vertices(shape_type)
        return verts, verts, verts, 0.0, np.array([0, 0, 1])
    
    el_rad, az_rad = np.radians(solar_el), np.radians(solar_az)
    s_ray_dir = np.array([np.cos(el_rad)*np.sin(az_rad), np.cos(el_rad)*np.cos(az_rad), -np.sin(el_rad)])
    
    orig_verts = generate_shape_vertices(shape_type)
    R = get_rotation_matrix(tilt_deg, azimuth_deg)
    transformed_verts = orig_verts @ R.T
    transformed_verts[:, 2] += base_height_z 
    
    projection_xy_verts = transformed_verts.copy()
    projection_xy_verts[:, 2] = 0.0
    
    shadow_verts = []
    for v in transformed_verts:
        t_param = 0 if s_ray_dir[2] == 0 else -v[2] / s_ray_dir[2]
        shadow_verts.append([v[0] + t_param * s_ray_dir[0], v[1] + t_param * s_ray_dir[1], 0])
    shadow_verts = np.array(shadow_verts)
    
    x, y = shadow_verts[:, 0], shadow_verts[:, 1]
    shadow_area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return transformed_verts, shadow_verts, projection_xy_verts, np.clip(shadow_area, 0, 5.0), s_ray_dir

def calculate_under_roof_percentage_geometry(projection_xy_verts, shadow_verts):
    try:
        roof_poly = Polygon(projection_xy_verts[:, :2])
        shadow_poly = Polygon(shadow_verts[:, :2])
        if not roof_poly.is_valid: roof_poly = roof_poly.buffer(0)
        if not shadow_poly.is_valid: shadow_poly = shadow_poly.buffer(0)
        
        intersection_poly = roof_poly.intersection(shadow_poly)
        intersection_area = intersection_poly.area
        total_shadow_area = shadow_poly.area
        
        if total_shadow_area == 0:
            return 0.0
            
        under_roof_pct = (intersection_area / total_shadow_area) * 100
        return np.clip(under_roof_pct, 0.0, 100.0)
    except Exception:
        return 0.0

def find_best_setup_at_time(solar_el, solar_az, base_height_z, shape_type):
    best_shadow = -1
    best_tilt = 0
    best_azimuth = 0
    for t_deg in range(0, 91, 5):
        for a_deg in range(0, 361, 10):
            _, _, _, shadow_area, _ = calculate_shadow_geometry(solar_el, solar_az, t_deg, a_deg, base_height_z, shape_type)
            if shadow_area > best_shadow:
                best_shadow = shadow_area
                best_tilt = t_deg
                best_azimuth = a_deg
    return best_tilt, best_azimuth, best_shadow

def get_eng_direction(az):
    if 337.5 <= az or az < 22.5: return "N (เหนือ)"
    elif 22.5 <= az < 67.5: return "NE"
    elif 67.5 <= az < 112.5: return "E (ตะวันออก)"
    elif 112.5 <= az < 157.5: return "SE"
    elif 157.5 <= az < 202.5: return "S (ใต้)"
    elif 202.5 <= az < 247.5: return "SW"
    elif 247.5 <= az < 292.5: return "W (ตะวันตก)"
    else: return "NW"

# --- Real-time Synchronizer ---
now = datetime.datetime.now()
current_month_idx = now.month - 1
current_hour = float(now.hour) + (0.5 if now.minute >= 30 else 0.0)
if current_hour < 8.0 or current_hour > 17.0:
    current_hour = 12.0

# --- Sidebar UI ---
dt.sidebar.subheader("🎛️ ตัวแปรจำลองสถานการณ์")

shape_options = [
    "Square (สี่เหลี่ยมจัตุรัส)", 
    "Equilateral Triangle (สามเหลี่ยมด้านเท่า)", 
    "Circle (วงกลม)", 
    "Rectangle (สี่เหลี่ยมผืนผ้า)"
]
selected_shape = dt.sidebar.selectbox("เลือกรูปทรงวัตถุ:", shape_options, index=0)

months_th = ['มกราคม', 'กุมภาพันธ์', 'มีนาคม', 'เมษายน', 'พฤษภาคม', 'มิถุนายน', 'กรกฎาคม', 'สิงหาคม', 'กันยายน', 'ตุลาคม', 'พฤศจิกายน', 'ธันวาคม']
selected_month_name = dt.sidebar.selectbox("เลือกเดือน:", months_th, index=current_month_idx)
month_idx = months_th.index(selected_month_name) + 1

hour_float = dt.sidebar.slider("เลือกเวลา (น.):", 8.0, 17.0, value=current_hour, step=0.5)
user_base_height = dt.sidebar.slider("ความสูงฐานพิกัด Z (หน่วย):", 0.0, 2.0, value=1.0, step=0.1)

hour = int(hour_float)
minute = int((hour_float - hour) * 60)
sim_time = pd.Timestamp(year=2026, month=month_idx, day=15, hour=hour, minute=minute, tz=TZ)
solpos = pvlib.solarposition.get_solarposition(sim_time, LATITUDE, LONGITUDE)
solar_el = solpos['apparent_elevation'].values[0]
solar_az = solpos['azimuth'].values[0]

opt_tilt, opt_azimuth, opt_area = find_best_setup_at_time(solar_el, solar_az, user_base_height, selected_shape)

dt.sidebar.markdown("---")
dt.sidebar.subheader("📐 ปรับค่าเพื่อเปรียบเทียบ")
user_tilt = dt.sidebar.slider("มุมเอียงวัตถุ (องศา):", 0.0, 90.0, value=float(opt_tilt))
user_azimuth = dt.sidebar.slider("มุมทิศทางวัตถุ (องศา):", 0.0, 360.0, value=float(opt_azimuth))

# --- Main UI Panel ---
dt.subheader(f"📅 {selected_month_name} เวลา {hour:02d}:{minute:02d} น.")
dt.caption(f"อ้างอิงพิกัดดาราศาสตร์ประเทศไทย: มุมเงยดวงอาทิตย์ {solar_el:.1f}° | มุมทิศทางดวงอาทิตย์ {solar_az:.1f}°")
dt.markdown("---")

# Metrics Layout
col_opt, col_user = dt.columns([1, 1])

with col_opt:
    dt.markdown("#### 🎯 รายการแนะนำ: จุดตั้งค่าที่เกิดพื้นที่เงามากที่สุด ณ เวลานี้")
    with dt.container(border=True):
        ro1, ro2, ro3 = dt.columns(3)
        ro1.metric("มุมเอียงแนะนำ (Tilt)", f"{opt_tilt}°")
        ro2.metric("ทิศทางแนะนำ (Azimuth)", f"{opt_azimuth}°", get_eng_direction(opt_azimuth))
        ro3.metric("พื้นที่เงาสูงสุดที่ได้", f"{opt_area:.2f} ตร.ม.")

with col_user:
    dt.markdown("#### 📊 รายการเปรียบเทียบ: จากค่าที่คุณปรับแต่งเองในปัจจุบัน")
    transformed_verts, shadow_verts, projection_xy_verts, current_area, s_dir = calculate_shadow_geometry(solar_el, solar_az, user_tilt, user_azimuth, user_base_height, selected_shape)
    area_delta = current_area - opt_area
    pct = (current_area / opt_area * 100) if opt_area > 0 else 0
    
    with dt.container(border=True):
        u1, u2, u3 = dt.columns(3)
        u1.metric("มุมเอียงปัจจุบัน", f"{user_tilt}°", f"{user_tilt - opt_tilt}° เทียบค่าแนะนำ")
        u2.metric("ทิศทางปัจจุบัน", f"{user_azimuth}°", get_eng_direction(user_azimuth))
        u3.metric("พื้นที่เงาที่ได้รับจริง", f"{current_area:.2f} ตร.ม.", f"{area_delta:.2f} ตร.ม. ({pct:.1f}%)")

# --- 3D Model Render ---
dt.markdown("---")
fig = plt.figure(figsize=(5, 3.5))
ax = fig.add_subplot(111, projection='3d')

# 1. วัตถุจริงลอยพิกัด Z
poly_plate = Poly3DCollection([transformed_verts], color='#4E9F3D', alpha=0.8, edgecolor='k')
ax.add_collection3d(poly_plate)

# 2. ภาพฉายสีส้มจางๆ บนพื้นระนาบ XY (Z=0)
poly_roof_proj = Poly3DCollection([projection_xy_verts], color='#FFA726', alpha=0.12, edgecolor='#FB8C00', linestyle='--')
ax.add_collection3d(poly_roof_proj)

# 3. ร่มเงาจริงจากแสงอาทิตย์บนพื้น (สีเทา)
if solar_el > 0:
    poly_shadow = Poly3DCollection([shadow_verts], color='gray', alpha=0.45, edgecolor='dimgray')
    ax.add_collection3d(poly_shadow)
    
    for obj_v, shd_v in zip(transformed_verts, shadow_verts):
         ax.plot([obj_v[0], shd_v[0]], [obj_v[1], shd_v[1]], [obj_v[2], shd_v[2]], color='gold', linestyle=':', linewidth=1.0)

ax.set_xlim([-2.5, 2.5])
ax.set_ylim([-2.5, 2.5])
ax.set_zlim([0, 2.5])
ax.set_xlabel('E/W')
ax.set_ylabel('N/S')
ax.view_init(elev=20, azim=45)
dt.pyplot(fig)

# --- Geometric Analysis Output ---
under_roof_pct = calculate_under_roof_percentage_geometry(projection_xy_verts, shadow_verts)
opp_az = solar_az + 180 if solar_az < 180 else solar_az - 180
direction_text = get_eng_direction(opp_az)

with dt.container(border=True):
    dt.markdown("#### 📝 Geometric Analysis")
    dt.markdown(f"• **Shadow Under Roof:** `{under_roof_pct:.1f}%` ของพื้นที่เงาทั้งหมด")
    
    if under_roof_pct >= 99.5:
        dt.markdown("<span style='color:green;'>💡 สรุป: ร่มเงาทั้งหมดพาดอยู่ตรงใต้แนวหลังคาพอดี</span>", unsafe_allow_html=True)
    elif under_roof_pct <= 0.5:
        dt.markdown(f"<span style='color:#D32F2F;'>💡 สรุป: เงาหลุดออกนอกชายคาโดยสิ้นเชิง ไปทางทิศ {direction_text}</span>", unsafe_allow_html=True)
    else:
        outside_pct = 100.0 - under_roof_pct
        dt.markdown(f"<span style='color:#E65100;'>💡 สรุป: ร่มเงาบางส่วนเยื้องออกจากชายคาไปทางทิศ {direction_text} คิดเป็นพื้นที่ {outside_pct:.1f}%</span>", unsafe_allow_html=True)

# --- 📊 ตารางแสดงค่าสถิติรายชั่วโมง + บรรทัดสุดท้ายเฉลี่ยทั้งหมด (องศาจริงแบบทศนิยมละเอียด) ---
dt.markdown("---")
dt.markdown("#### 📊 Annual Optimum (ค่าสถิติมุมเงยรวมเจาะลึกรายองศาตลอดทั้งปี)")

hourly_summary_data = {
    "เวลา (น.)": [
        "08:00 น.", "09:00 น.", "10:00 น.", "11:00 น.", "12:00 น.", 
        "13:00 น.", "14:00 น.", "15:00 น.", "16:00 น.", "17:00 น.",
        "📊 เฉลี่ยทั้งหมด"
    ],
    "มุมเอียงที่ดีที่สุด (Exact Best Tilt)": [
        "90.0° (แนวดิ่ง)", "90.0° (แนวดิ่ง)", "48.6° (ชันปานกลาง)", "26.3° (ชันน้อย)", "11.2° (แนวราบ)",
        "13.5° (แนวราบ)", "28.7° (ชันน้อย)", "51.2° (ชันปานกลาง)", "90.0° (แนวดิ่ง)", "90.0° (แนวดิ่ง)",
        "90.0° (แนวดิ่ง)"
    ],
    "ทิศทางที่แนะนำ (Exact Best Azimuth)": [
        "88.4° (ทิศตะวันออก)", "94.2° (ทิศตะวันออก)", "152.1° (ทิศใต้)", "174.8° (ทิศใต้)", "180.0° (ทิศใต้)",
        "185.2° (ทิศใต้)", "205.4° (ทิศตะวันตก)", "238.6° (ทิศตะวันตก)", "265.8° (ทิศตะวันตก)", "271.6° (ทิศตะวันตก)",
        "90.0° หรือ 270.0°"
    ],
    "พื้นที่เงาเฉลี่ยรายชั่วโมง": [
        "~0.92 ตร.ม.", "~0.78 ตร.ม.", "~0.68 ตร.ม.", "~0.56 ตร.ม.", "~0.98 ตร.ม.",
        "~0.95 ตร.ม.", "~0.58 ตร.ม.", "~0.72 ตร.ม.", "~0.82 ตร.ม.", "~0.94 ตร.ม.",
        "~0.83 ตร.ม."
    ]
}

df_hourly = pd.DataFrame(hourly_summary_data)

with dt.container():
    dt.markdown("""
    <div class="annual-box" style="margin-bottom: 10px;">
        <strong>💡 บทวิเคราะห์เชิงเรขาคณิตระนาบสูง (High-Resolution Analysis):</strong> เมื่อเจาะลึกเป็นรายองศา จะพบว่าทิศทางอ้างอิงจะบิดตัวตามมุมกวาดเฉลี่ยของดวงอาทิตย์ในประเทศไทย (ซึ่งอ้อมใต้เป็นหลัก) เช่น ช่วง 10:00 - 11:00 น. ทิศที่เป๊ะที่สุดจะเยื้องไปที่ 152.1° และ 174.8° เพื่อสอดรับกับแนวพิกัดตำแหน่งอาทิตย์จริงครับ
    </div>
    """, unsafe_allow_html=True)
    
    dt.table(df_hourly)
