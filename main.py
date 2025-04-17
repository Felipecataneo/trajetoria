import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from io import BytesIO
import math

st.set_page_config(page_title="Comparador de Distâncias ISCWSA", layout="wide")

# --- Funções Trigonométricas (Graus) ---
def sind(degrees):
    return np.sin(np.radians(degrees))

def cosd(degrees):
    return np.cos(np.radians(degrees))

def tand(degrees):
    return np.tan(np.radians(degrees))

def atand(value):
    return np.degrees(np.arctan(value))

def atan2d(y, x):
    return np.degrees(np.arctan2(y, x))

def acosd(value):
    # Clip value to avoid domain errors due to floating point inaccuracies
    return np.degrees(np.arccos(np.clip(value, -1.0, 1.0)))

# Título e explicação
st.title("Comparador de Distâncias com Modelos ISCWSA")
st.markdown("""
Esta aplicação compara a distância entre centros de trajetórias com a distância Pedal Curve,
utilizando modelos de incerteza padrão ISCWSA (MWD ou Gyro) para calcular as elipses de incerteza.
""")

# --- Parâmetros de Incerteza ISCWSA ---
st.sidebar.header("Configuração do Modelo de Incerteza")
tool_type = st.sidebar.selectbox("Selecione o Tipo de Ferramenta", ["ISCWSA MWD", "ISCWSA Gyro"])

iscwsa_params = {}

if tool_type == "ISCWSA MWD":
    st.sidebar.subheader("Parâmetros ISCWSA MWD (1-sigma)")
    # Parâmetros comuns baseados no OWSG MWD Error Model Rev. 4/5
    # Convertendo onde necessário (e.g., % para fração, m/100m para m/m)
    iscwsa_params['depth_err_prop'] = st.sidebar.number_input("Erro Prop. Profundidade (m/m)", value=0.0002, step=0.0001, format="%.4f") # dD1
    iscwsa_params['depth_err_const'] = st.sidebar.number_input("Erro Const. Profundidade (m)", value=0.1, step=0.05, format="%.2f") # dD0

    iscwsa_params['acc_bias'] = st.sidebar.number_input("Acc Bias (mg)", value=0.1, step=0.01, format="%.2f") # ABX, ABY, ABZ
    iscwsa_params['acc_sf'] = st.sidebar.number_input("Acc Scale Factor (ppm)", value=100.0, step=10.0, format="%.1f") # ASX, ASY, ASZ
    iscwsa_params['acc_mis_xy'] = st.sidebar.number_input("Acc Misalign XY (mrad)", value=0.1, step=0.01, format="%.2f") # AMX, AMY
    iscwsa_params['acc_mis_z'] = st.sidebar.number_input("Acc Misalign Z (mrad)", value=0.1, step=0.01, format="%.2f") # AMZ

    iscwsa_params['mag_bias'] = st.sidebar.number_input("Mag Bias (nT)", value=50.0, step=5.0, format="%.1f") # MBX, MBY, MBZ
    iscwsa_params['mag_sf'] = st.sidebar.number_input("Mag Scale Factor (ppm)", value=150.0, step=10.0, format="%.1f") # MSX, MSY, MSZ
    iscwsa_params['mag_mis_xy'] = st.sidebar.number_input("Mag Misalign XY (mrad)", value=0.15, step=0.01, format="%.2f") # MMX, MMY
    iscwsa_params['mag_mis_z'] = st.sidebar.number_input("Mag Misalign Z (mrad)", value=0.15, step=0.01, format="%.2f") # MMZ

    iscwsa_params['mag_dec_err'] = st.sidebar.number_input("Erro Declinação Magnética (°)", value=0.2, step=0.05, format="%.2f") # DECD
    iscwsa_params['mag_dip_err'] = st.sidebar.number_input("Erro Inclinação Magnética (°)", value=0.1, step=0.01, format="%.2f") # DIPD (usado internamente nos cálculos de azimute)
    iscwsa_params['mag_ds_err'] = st.sidebar.number_input("Erro Interferência Magnética (°)", value=0.3, step=0.05, format="%.2f") # XYMD

    iscwsa_params['sag_corr_err'] = st.sidebar.number_input("Erro Correção SAG (°)", value=0.05, step=0.01, format="%.2f") # SAGD
    # Added basic toolface/misalignment error contribution (simplified)
    iscwsa_params['misalign_err_inc'] = st.sidebar.number_input("Erro Misalign INC (°)", value=0.05, step=0.01, format="%.2f") # Relacionado a MXA, MYA
    iscwsa_params['misalign_err_azi'] = st.sidebar.number_input("Erro Misalign AZI (°)", value=0.1, step=0.01, format="%.2f") # Relacionado a MZA

    # Gravidade e Campo Magnético (Simplificado - considere usar valores locais)
    iscwsa_params['gravity_strength'] = st.sidebar.number_input("Gravidade (g)", value=1.0, format="%.4f")
    iscwsa_params['mag_field_strength'] = st.sidebar.number_input("Campo Magnético (nT)", value=50000.0, format="%.1f")
    iscwsa_params['dip_angle'] = st.sidebar.number_input("Ângulo DIP (°)", value=60.0, format="%.2f") # Inclinação magnética

elif tool_type == "ISCWSA Gyro":
    st.sidebar.subheader("Parâmetros ISCWSA Gyro (1-sigma)")
    # Similar to MWD but replaces Magnetic terms with Gyro terms
    iscwsa_params['depth_err_prop'] = st.sidebar.number_input("Erro Prop. Profundidade (m/m)", value=0.0002, step=0.0001, format="%.4f")
    iscwsa_params['depth_err_const'] = st.sidebar.number_input("Erro Const. Profundidade (m)", value=0.1, step=0.05, format="%.2f")

    iscwsa_params['acc_bias'] = st.sidebar.number_input("Acc Bias (mg)", value=0.1, step=0.01, format="%.2f")
    iscwsa_params['acc_sf'] = st.sidebar.number_input("Acc Scale Factor (ppm)", value=100.0, step=10.0, format="%.1f")
    iscwsa_params['acc_mis_xy'] = st.sidebar.number_input("Acc Misalign XY (mrad)", value=0.1, step=0.01, format="%.2f")
    iscwsa_params['acc_mis_z'] = st.sidebar.number_input("Acc Misalign Z (mrad)", value=0.1, step=0.01, format="%.2f")

    iscwsa_params['gyro_bias_drift_ns'] = st.sidebar.number_input("Gyro Bias Drift N/S (°/hr)", value=0.1, step=0.01, format="%.2f") # GBN
    iscwsa_params['gyro_bias_drift_ew'] = st.sidebar.number_input("Gyro Bias Drift E/W (°/hr)", value=0.1, step=0.01, format="%.2f") # GBE
    iscwsa_params['gyro_bias_drift_v'] = st.sidebar.number_input("Gyro Bias Drift Vert (°/hr)", value=0.1, step=0.01, format="%.2f") # GBV
    iscwsa_params['gyro_sf'] = st.sidebar.number_input("Gyro Scale Factor (ppm)", value=200.0, step=10.0, format="%.1f") # GSF
    iscwsa_params['gyro_g_sens_drift'] = st.sidebar.number_input("Gyro G-Sens Drift (°/hr/g)", value=0.05, step=0.01, format="%.2f") # GDX, GDY, GDZ
    iscwsa_params['gyro_mis_xy'] = st.sidebar.number_input("Gyro Misalign XY (mrad)", value=0.1, step=0.01, format="%.2f") # GMX, GMY
    iscwsa_params['gyro_mis_z'] = st.sidebar.number_input("Gyro Misalign Z (mrad)", value=0.1, step=0.01, format="%.2f") # GMZ
    iscwsa_params['gyro_az_ref_err'] = st.sidebar.number_input("Erro Referência Azimute Gyro (°)", value=0.1, step=0.01, format="%.2f") # AZID

    iscwsa_params['sag_corr_err'] = st.sidebar.number_input("Erro Correção SAG (°)", value=0.05, step=0.01, format="%.2f") # SAGD
    iscwsa_params['misalign_err_inc'] = st.sidebar.number_input("Erro Misalign INC (°)", value=0.05, step=0.01, format="%.2f")
    iscwsa_params['misalign_err_azi'] = st.sidebar.number_input("Erro Misalign AZI (°)", value=0.1, step=0.01, format="%.2f")

    # Gravidade
    iscwsa_params['gravity_strength'] = st.sidebar.number_input("Gravidade (g)", value=1.0, format="%.4f")
    # Assumed survey time (for drift calculation) - simplistic
    iscwsa_params['survey_time_hours'] = st.sidebar.number_input("Tempo Estimado Survey (horas)", value=1.0, step=0.1, format="%.1f")

st.sidebar.header("Parâmetros de Cálculo")
sigma_factor = st.sidebar.slider("Fator Sigma (Confiança da Elipse)", 1.0, 3.0, 1.0, 0.1)

# --- Funções de Cálculo Trajetória ---
def calculate_coordinates(md, inc, az):
   """Calcula coordenadas usando método minimum curvature"""
   n = [0.0]
   e = [0.0]
   tvd = [0.0]

   for i in range(1, len(md)):
       segment = md[i] - md[i-1]
       if segment < 1e-6: # Evita divisão por zero ou erros em pontos duplicados
           n.append(n[-1])
           e.append(e[-1])
           tvd.append(tvd[-1])
           continue

       inc1_rad = np.radians(inc[i-1])
       inc2_rad = np.radians(inc[i])
       az1_rad = np.radians(az[i-1])
       az2_rad = np.radians(az[i])

       # Fator de correção de minimum curvature
       # cos(Dogleg) = cos(dInc) - sin(Inc1)sin(Inc2)(1-cos(dAz))
       cos_dls = np.cos(inc2_rad - inc1_rad) - np.sin(inc1_rad) * np.sin(inc2_rad) * (1 - np.cos(az2_rad - az1_rad))
       dogleg_angle_rad = np.arccos(np.clip(cos_dls, -1.0, 1.0)) # Dogleg Angle

       if abs(dogleg_angle_rad) < 1e-6:
           rf = 1.0
       else:
           rf = np.tan(dogleg_angle_rad / 2.0) / (dogleg_angle_rad / 2.0) # Ratio Factor = tan(DL/2) / (DL/2)

       # Incrementos de coordenadas
       dn = segment / 2.0 * (np.sin(inc1_rad) * np.cos(az1_rad) + np.sin(inc2_rad) * np.cos(az2_rad)) * rf
       de = segment / 2.0 * (np.sin(inc1_rad) * np.sin(az1_rad) + np.sin(inc2_rad) * np.sin(az2_rad)) * rf
       dv = segment / 2.0 * (np.cos(inc1_rad) + np.cos(inc2_rad)) * rf

       n.append(n[-1] + dn)
       e.append(e[-1] + de)
       tvd.append(tvd[-1] + dv)

   return pd.DataFrame({
       'MD': md,
       'TVD': tvd,
       'N': n,
       'E': e,
       'INC': inc,
       'AZ': az
   })

# --- Funções de Cálculo Incerteza ISCWSA (Aproximação por Contribuição de Variância) ---

def calculate_iscwsa_covariance(md, inc, az, params, tool_type):
    """
    Calcula a matriz de covariância 3x3 (NEV) em um ponto específico
    usando uma aproximação baseada na soma das contribuições de variância
    dos termos de erro ISCWSA.
    Retorna a matriz C_nev.
    """
    # Conversões de unidades para consistência interna (radianos, frações)
    inc_rad = np.radians(inc)
    az_rad = np.radians(az)
    mrad_to_rad = 0.001
    ppm_to_frac = 1e-6
    mg_to_g = 0.001
    nT_to_T = 1e-9
    deg_to_rad = np.pi / 180.0
    hr_to_sec = 3600.0

    # Parâmetros de erro (convertidos para radianos ou frações onde aplicável)
    sigma_depth_prop = params.get('depth_err_prop', 0)
    sigma_depth_const = params.get('depth_err_const', 0)
    sigma_acc_bias = params.get('acc_bias', 0) * mg_to_g * 9.81 # Convertido para m/s^2 approx.
    sigma_acc_sf = params.get('acc_sf', 0) * ppm_to_frac
    sigma_acc_mis = params.get('acc_mis_xy', 0) * mrad_to_rad # Usando XY como representativo
    sigma_sag = params.get('sag_corr_err', 0) * deg_to_rad
    sigma_misalign_inc = params.get('misalign_err_inc', 0) * deg_to_rad
    sigma_misalign_azi = params.get('misalign_err_azi', 0) * deg_to_rad

    # Variâncias (sigma^2)
    var_depth = (sigma_depth_prop * md)**2 + sigma_depth_const**2
    var_acc_bias = sigma_acc_bias**2
    var_acc_sf = sigma_acc_sf**2
    var_acc_mis = sigma_acc_mis**2
    var_sag = sigma_sag**2
    var_misalign_inc = sigma_misalign_inc**2
    var_misalign_azi = sigma_misalign_azi**2

    # --- Cálculo das variâncias no sistema da ferramenta (Axial, Lateral, Vertical) ---
    # Simplificação: Assume que a variância total é a soma das variâncias das fontes
    # Fórmulas baseadas em ISCWSA, mas altamente simplificadas para esta demo

    # Variância Axial (ao longo do poço) - Dominada por erro de profundidade
    var_axial = var_depth

    # Variância Vertical (na direção da gravidade local) - Dominada por erros de inclinação
    # (Acc Bias, SF, Misalign, SAG, Misalign Inc)
    # Sensibilidade simplificada: dV ~= MD * dInc
    var_vertical_tool = (md**2) * (
        (var_acc_bias / (params.get('gravity_strength', 1.0)*9.81)**2) + # Erro Inc por bias Acc
        var_acc_sf * (sind(inc)**2) + # Erro Inc por SF Acc
        var_acc_mis * (cosd(inc)**2) + # Erro Inc por Misalign Acc
        var_sag +
        var_misalign_inc
    )

    # Variância Lateral (Horizontal, perpendicular ao eixo do poço) - Dominada por erros de azimute
    var_lateral_tool = 0

    if tool_type == "ISCWSA MWD":
        sigma_mag_bias = params.get('mag_bias', 0) * nT_to_T
        sigma_mag_sf = params.get('mag_sf', 0) * ppm_to_frac
        sigma_mag_mis = params.get('mag_mis_xy', 0) * mrad_to_rad
        sigma_dec_err = params.get('mag_dec_err', 0) * deg_to_rad
        sigma_ds_err = params.get('mag_ds_err', 0) * deg_to_rad
        dip_rad = np.radians(params.get('dip_angle', 60))
        B_H = params.get('mag_field_strength', 50000) * nT_to_T * np.cos(dip_rad)

        var_mag_bias = sigma_mag_bias**2
        var_mag_sf = sigma_mag_sf**2
        var_mag_mis = sigma_mag_mis**2
        var_dec_err = sigma_dec_err**2
        var_ds_err = sigma_ds_err**2

        # Contribuição de Azimute (simplificada)
        # dAz ~= (Erro Mag) / (B_H * sin(Inc)) + Erro Declinação + Erro DS + Erro Misalign Azi
        # Ignorando termos complexos de SF/Misalign por simplicidade
        if abs(sind(inc)) > 1e-3: # Evita divisão por zero em inclinação zero
            var_az_component = (
                (var_mag_bias / (B_H * sind(inc))**2) +
                var_mag_sf * (cosd(inc)**2 / sind(inc)**2) + # Termo SF (muito simplificado)
                var_mag_mis * (1 / tand(inc)**2) + # Termo Misalign (muito simplificado)
                var_dec_err +
                var_ds_err +
                var_misalign_azi
            )
        else: # Próximo da vertical, azimute muito incerto
             var_az_component = (10*deg_to_rad)**2 # Assume grande incerteza

        # Sensibilidade simplificada: dLateral ~= MD * sin(Inc) * dAz
        var_lateral_tool = (md * sind(inc))**2 * var_az_component

    elif tool_type == "ISCWSA Gyro":
        sigma_gyro_bias_drift_h = np.sqrt(params.get('gyro_bias_drift_ns', 0)**2 + params.get('gyro_bias_drift_ew', 0)**2) * deg_to_rad / hr_to_sec
        sigma_gyro_sf = params.get('gyro_sf', 0) * ppm_to_frac
        sigma_gyro_g_sens_drift = params.get('gyro_g_sens_drift', 0) * deg_to_rad / hr_to_sec # Por g
        sigma_gyro_mis = params.get('gyro_mis_xy', 0) * mrad_to_rad
        sigma_az_ref_err = params.get('gyro_az_ref_err', 0) * deg_to_rad
        survey_time_sec = params.get('survey_time_hours', 1) * hr_to_sec

        var_gyro_bias_drift = (sigma_gyro_bias_drift_h * survey_time_sec)**2
        var_gyro_sf = sigma_gyro_sf**2 # Afeta a taxa de rotação medida
        var_gyro_g_sens = (sigma_gyro_g_sens_drift * survey_time_sec * sind(inc))**2 # Drift induzido pela gravidade
        var_gyro_mis = sigma_gyro_mis**2
        var_az_ref = sigma_az_ref_err**2

        # Contribuição de Azimute Gyro (simplificada)
        # dAz ~= BiasDrift * Tempo + Erro SF * AnguloGirado + GsensDrift*Tempo*sin(Inc) + Erro Ref + Erro Misalign Azi
        # Ignorando termos complexos SF/Misalign
        var_az_component = (
            var_gyro_bias_drift +
            var_gyro_g_sens +
            var_gyro_mis * (1 / tand(inc)**2 if abs(sind(inc)) > 1e-3 else (10*deg_to_rad)**2) + # Simplificado
            var_az_ref +
            var_misalign_azi
        )
        # Sensibilidade simplificada: dLateral ~= MD * sin(Inc) * dAz
        var_lateral_tool = (md * sind(inc))**2 * var_az_component


    # --- Rotação para o sistema NEV ---
    # Matriz de covariância no sistema da ferramenta (diagonal, por simplificação)
    C_tool = np.diag([var_lateral_tool, var_axial, var_vertical_tool])

    # Matriz de Rotação de Tool (Lateral, Axial, Vertical) para NEV
    si = sind(inc)
    ci = cosd(inc)
    sa = sind(az)
    ca = cosd(az)

    # Rotação: NEV = R * ToolFrame(lat, ax, vert)
    # Lateral(x') -> Eixo Y do toolframe (aprox)
    # Axial(y') -> Eixo Z do toolframe (aprox)
    # Vertical(z') -> Eixo X do toolframe (aprox)
    # Esta convenção pode variar, ajuste se necessário!
    # Ref: Applied Drilling Engineering, Appendix A (Coord Systems) - adaptado
    # Ref: ISCWSA convention might differ slightly.

    # Assuming Tool Frame: x=Vertical(Up), y=Lateral(Right), z=Axial(Downhole)
    # And NEV Frame: N=North(+Y), E=East(+X), V=Vertical(Down, +Z)
    # Rotation Matrix from Tool(xyz) to NEV(yxz - permuted!)
    R = np.array([
        [-si*sa, ca, -ci*sa],  # N = f(lat, ax, vert) -> row 1
        [ si*ca, sa,  ci*ca],  # E = f(lat, ax, vert) -> row 2
        [ ci,    0,  -si]      # V = f(lat, ax, vert) -> row 3
    ])


    # Rotação da Matriz de Covariância: C_nev = R * C_tool * R.T
    C_nev = R @ C_tool @ R.T

    return C_nev

def get_ellipse_params_from_covariance(C_nev, sigma_factor=1.0):
    """
    Calcula os parâmetros da elipse de incerteza horizontal a partir
    da matriz de covariância 3x3 NEV.
    Retorna: semi_major, semi_minor, angle_deg (ângulo do eixo maior com o Norte)
    """
    # Extrair submatriz 2x2 horizontal (NE) - Cuidado com a ordem N, E!
    # C_nev = [[VarN, CovNE, CovNV], [CovNE, VarE, CovEV], [CovNV, CovEV, VarV]]
    C_ne = C_nev[0:2, 0:2] # Extrai [[VarN, CovNE], [CovNE, VarE]]

    # Calcular autovalores e autovetores da matriz C_ne
    # Autovalores dão as variâncias ao longo dos eixos principais
    # Autovetores dão a direção desses eixos
    eigenvalues, eigenvectors = np.linalg.eig(C_ne)

    # Semi-eixos são a raiz quadrada dos autovalores, vezes o fator sigma
    # Garante que semi_major seja o maior
    idx_sort = np.argsort(eigenvalues)[::-1] # Índices do maior para o menor eigenvalue
    eigenvalues = eigenvalues[idx_sort]
    eigenvectors = eigenvectors[:, idx_sort]

    semi_major = sigma_factor * np.sqrt(max(0, eigenvalues[0])) # Evita raiz de negativos pequenos
    semi_minor = sigma_factor * np.sqrt(max(0, eigenvalues[1]))

    # Ângulo do eixo maior (correspondente ao primeiro autovetor)
    # O primeiro autovetor é eigenvectors[:, 0] = [N_comp, E_comp]
    n_comp = eigenvectors[0, 0]
    e_comp = eigenvectors[1, 0]
    # atan2d(y, x) -> atan2d(North, East) dá o ângulo anti-horário a partir do eixo Leste
    angle_rad_from_east = np.arctan2(n_comp, e_comp)

    # Converter para ângulo anti-horário a partir do Norte
    angle_rad_from_north = np.pi/2.0 - angle_rad_from_east
    # Normalizar para 0-360 graus
    angle_deg_from_north = np.degrees(angle_rad_from_north) % 360.0

    # Ajustar para 0-180 para representação da elipse (simetria)
    # angle_deg_ellipse = angle_deg_from_north % 180.0

    return semi_major, semi_minor, angle_deg_from_north # Retorna ângulo 0-360


# --- Funções de Cálculo Pedal Curve Atualizadas ---

def calculate_distance(p1, p2):
   """Calcula a distância euclidiana entre dois pontos no plano NE"""
   return np.sqrt((p2['N'] - p1['N'])**2 + (p2['E'] - p1['E'])**2)

def project_ellipse_iscwsa(semi_major, semi_minor, ellipse_angle_deg, direction_az_deg):
   """
   Projeta a elipse ISCWSA na direção especificada (Pedal Curve).
   ellipse_angle_deg: Ângulo do eixo MAIOR da elipse, anti-horário a partir do Norte.
   direction_az_deg: Azimute da linha conectando os centros, anti-horário a partir do Norte.
   """
   # Ângulo relativo entre a direção de projeção e o eixo MAIOR da elipse
   relative_angle_rad = np.radians(direction_az_deg - ellipse_angle_deg)

   a = semi_major
   b = semi_minor

   # Fórmula da projeção pedal (raio da elipse na direção relativa)
   # Ref: https://math.stackexchange.com/questions/17876/distance-along-a-line-from-the-origin-to-an-ellipse
   # r^2 = (a^2 * b^2) / (b^2 * cos^2(theta) + a^2 * sin^2(theta)) where theta is angle from major axis
   # Proj = r
   num = (a**2) * (b**2)
   den = (b**2) * (np.cos(relative_angle_rad)**2) + (a**2) * (np.sin(relative_angle_rad)**2)

   if den < 1e-9: # Evita divisão por zero se a=0 ou b=0
       return 0.0

   projection = np.sqrt(num / den)
   return projection


def calculate_pedal_distance_iscwsa(p1, p2, cov_nev1, cov_nev2, sigma_factor):
    """Calcula a distância Pedal Curve entre dois pontos usando elipses ISCWSA"""
    # Distância entre centros
    center_dist = calculate_distance(p1, p2)
    if center_dist < 1e-6:
        return {'center_dist': 0, 'proj1': 0, 'proj2': 0, 'pedal_dist': 0, 'difference': 0, 'diff_percent': 0}

    # Direção entre centros (Azimute Norte -> Leste)
    delta_n = p2['N'] - p1['N']
    delta_e = p2['E'] - p1['E']
    angle_deg_centers = atan2d(delta_e, delta_n) # atan2(y,x) -> atan2(E, N) -> Azimute

    # Calcular parâmetros das elipses a partir das matrizes de covariância
    smj1, smn1, ang1 = get_ellipse_params_from_covariance(cov_nev1, sigma_factor)
    smj2, smn2, ang2 = get_ellipse_params_from_covariance(cov_nev2, sigma_factor)

    # Projeção das elipses na direção que conecta os centros
    proj1 = project_ellipse_iscwsa(smj1, smn1, ang1, angle_deg_centers)
    proj2 = project_ellipse_iscwsa(smj2, smn2, ang2, angle_deg_centers)

    # Distância Pedal Curve
    pedal_dist = max(0, center_dist - (proj1 + proj2))

    return {
        'center_dist': center_dist,
        'proj1': proj1,
        'proj2': proj2,
        'pedal_dist': pedal_dist,
        'difference': center_dist - pedal_dist,
        'diff_percent': ((center_dist - pedal_dist) / center_dist) * 100 if center_dist > 0 else 0,
        'smj1': smj1, 'smn1': smn1, 'ang1': ang1, # Parâmetros da elipse 1
        'smj2': smj2, 'smn2': smn2, 'ang2': ang2  # Parâmetros da elipse 2
    }


def find_closest_tvd_point(tvd_target, df):
   """Encontra o ponto mais próximo em TVD"""
   if df.empty:
        return None
   idx = (df['TVD'] - tvd_target).abs().idxmin()
   return df.loc[idx]

def draw_ellipse_matplotlib(ax, center_xy, width, height, angle_deg, color="blue", alpha=0.3, label=None):
   """Desenha uma elipse com Matplotlib. Angle é anti-horário a partir do eixo X (Leste)."""
   # matplotlib angle: counterclockwise starting from the positive x-axis
   ellipse = Ellipse(xy=center_xy, width=width, height=height, angle=angle_deg,
                     edgecolor=color, facecolor=color, alpha=alpha, label=label)
   ax.add_patch(ellipse)
   return ellipse


# --- Interface Streamlit ---

# Interface para upload de arquivos
col1, col2 = st.columns(2)

with col1:
   st.header("Poço 1")
   well1_file = st.file_uploader("Upload Excel (MD, INC, AZ) Poço 1", type=["xlsx", "xls"], key="file1")

with col2:
   st.header("Poço 2")
   well2_file = st.file_uploader("Upload Excel (MD, INC, AZ) Poço 2", type=["xlsx", "xls"], key="file2")


# Processamento quando ambos os arquivos são carregados
if well1_file and well2_file:
   # Leitura dos arquivos Excel
   try:
       df_well1 = pd.read_excel(well1_file)
       df_well2 = pd.read_excel(well2_file)

       # Verificar e padronizar nomes das colunas
       expected_cols = ['MD', 'INC', 'AZ']
       data_valid = True
       for df, well_name, file_ref in [(df_well1, "Poço 1", well1_file), (df_well2, "Poço 2", well2_file)]:
           # Make columns upper case for comparison
           df.columns = [str(col).upper() for col in df.columns]

           # Check for essential columns
           if not all(col in df.columns for col in expected_cols):
               st.error(f"Arquivo do {well_name} ({file_ref.name}) deve conter colunas: MD, INC, AZ.")
               data_valid = False
               continue # Go to next file check

            # Select and rename standard columns
           rename_map = {}
           if 'MD' not in df.columns: data_valid = False # Should be caught above, but double check
           if 'INC' not in df.columns:
               if 'INCLINACAO' in df.columns: rename_map['INCLINACAO'] = 'INC'
               elif 'INCLINAÇÃO' in df.columns: rename_map['INCLINAÇÃO'] = 'INC'
               else: data_valid = False
           if 'AZ' not in df.columns:
                if 'AZIMUTE' in df.columns: rename_map['AZIMUTE'] = 'AZ'
                elif 'AZIMUTH' in df.columns: rename_map['AZIMUTH'] = 'AZ'
                else: data_valid = False

           if not data_valid:
                st.error(f"Não foi possível encontrar colunas INC ou AZ no arquivo {well_name} ({file_ref.name}). Verifique os nomes.")
                st.stop()

           df.rename(columns=rename_map, inplace=True)
           # Ensure numeric types, coerce errors to NaN
           for col in expected_cols:
               df[col] = pd.to_numeric(df[col], errors='coerce')
           # Drop rows with NaN in essential columns
           initial_rows = len(df)
           df.dropna(subset=expected_cols, inplace=True)
           if len(df) < initial_rows:
               st.warning(f"Removidas {initial_rows - len(df)} linhas com dados inválidos/faltantes em {well_name}.")
           if len(df) < 2:
                st.error(f"Arquivo do {well_name} ({file_ref.name}) não contém dados suficientes após limpeza.")
                data_valid = False

       if not data_valid:
           st.stop() # Stop execution if data is invalid


       # Calcular coordenadas
       coords_well1 = calculate_coordinates(
           df_well1['MD'].values, df_well1['INC'].values, df_well1['AZ'].values
       )
       coords_well2 = calculate_coordinates(
           df_well2['MD'].values, df_well2['INC'].values, df_well2['AZ'].values
       )

       # Adicionar Covariância a cada ponto (pode ser lento para muitos pontos)
       st.write("Calculando incerteza ISCWSA para cada ponto...")
       prog_bar = st.progress(0)
       total_pts = len(coords_well1) + len(coords_well2)
       pts_done = 0

       covs1 = []
       for i, row in coords_well1.iterrows():
           cov = calculate_iscwsa_covariance(row['MD'], row['INC'], row['AZ'], iscwsa_params, tool_type)
           covs1.append(cov)
           pts_done += 1
           prog_bar.progress(pts_done / total_pts)
       coords_well1['Covariance'] = covs1

       covs2 = []
       for i, row in coords_well2.iterrows():
           cov = calculate_iscwsa_covariance(row['MD'], row['INC'], row['AZ'], iscwsa_params, tool_type)
           covs2.append(cov)
           pts_done += 1
           prog_bar.progress(pts_done / total_pts)
       coords_well2['Covariance'] = covs2
       prog_bar.empty()


       # Encontrar TVDs comuns ou próximas para comparação
       tvds_well1 = coords_well1['TVD'].unique()
       results = []
       st.write("Comparando trajetórias em profundidades correspondentes...")

       for tvd in tvds_well1:
           # Encontrar pontos mais próximos em TVD
           p1 = find_closest_tvd_point(tvd, coords_well1)
           p2 = find_closest_tvd_point(tvd, coords_well2)

           if p1 is None or p2 is None: continue

           # Apenas comparar se a diferença de TVD for pequena (ex: menos de 5m)
           # E se as covariâncias foram calculadas
           if abs(p1['TVD'] - p2['TVD']) < 5 and 'Covariance' in p1 and 'Covariance' in p2:
               # Calcular distâncias usando ISCWSA
               distance_data = calculate_pedal_distance_iscwsa(
                   p1, p2,
                   p1['Covariance'], p2['Covariance'],
                   sigma_factor
               )

               # Adicionar aos resultados
               results.append({
                   'TVD': p1['TVD'],
                   'MD1': p1['MD'], 'MD2': p2['MD'],
                   'INC1': p1['INC'], 'INC2': p2['INC'],
                   'AZ1': p1['AZ'], 'AZ2': p2['AZ'],
                   'N1': p1['N'], 'E1': p1['E'],
                   'N2': p2['N'], 'E2': p2['E'],
                   'DistCentros': distance_data['center_dist'],
                   'DistPedal': distance_data['pedal_dist'],
                   'Proj1': distance_data['proj1'], 'Proj2': distance_data['proj2'],
                   'SMj1': distance_data['smj1'], 'SMn1': distance_data['smn1'], 'Ang1': distance_data['ang1'],
                   'SMj2': distance_data['smj2'], 'SMn2': distance_data['smn2'], 'Ang2': distance_data['ang2'],
                   'DifPerc': distance_data['diff_percent'],
                   'AZ_Diff': abs(p1['AZ'] - p2['AZ']) % 180,
                   'INC_Avg': (p1['INC'] + p2['INC']) / 2
               })

       # Criar dataframe de resultados
       if results:
           df_results = pd.DataFrame(results)

           # Exibir tabela de resultados
           st.subheader("Comparação de Distâncias")
           st.dataframe(df_results[[
               'TVD', 'DistCentros', 'DistPedal', 'DifPerc',
               'SMj1', 'SMn1', 'Ang1', 'SMj2', 'SMn2', 'Ang2'
               ]].round(2))

           # --- Gráficos ---
           st.subheader("Gráficos de Análise")

           # Gráfico de distâncias vs TVD
           fig1 = go.Figure()
           fig1.add_trace(go.Scatter(x=df_results['TVD'], y=df_results['DistCentros'], mode='lines+markers', name='Dist. Centros'))
           fig1.add_trace(go.Scatter(x=df_results['TVD'], y=df_results['DistPedal'], mode='lines+markers', name=f'Dist. Pedal ({sigma_factor:.1f}σ)'))
           fig1.update_layout(title='Distâncias vs Profundidade', xaxis_title='TVD (m)', yaxis_title='Distância (m)', legend=dict(x=0, y=1))
           st.plotly_chart(fig1, use_container_width=True)

           # Visualização 3D das trajetórias
           st.subheader("Visualização 3D das Trajetórias")
           fig5 = go.Figure()
           fig5.add_trace(go.Scatter3d(x=coords_well1['E'], y=coords_well1['N'], z=coords_well1['TVD'], mode='lines', name='Poço 1', line=dict(color='blue', width=4)))
           fig5.add_trace(go.Scatter3d(x=coords_well2['E'], y=coords_well2['N'], z=coords_well2['TVD'], mode='lines', name='Poço 2', line=dict(color='red', width=4)))
           fig5.update_layout(scene=dict(xaxis_title='Este (m)', yaxis_title='Norte (m)', zaxis_title='TVD (m)', zaxis=dict(autorange="reversed")), height=700)
           st.plotly_chart(fig5, use_container_width=True)

           # Visualização 2D das trajetórias com elipses de incerteza
           st.subheader(f"Visualização 2D com Elipses de Incerteza ({sigma_factor:.1f}σ)")
           tvd_options = np.sort(df_results['TVD'].unique())
           selected_tvd = st.selectbox("Selecione a TVD para visualização 2D", tvd_options, index=len(tvd_options)//2)

           # Filtrar dados para a TVD selecionada
           selected_data = df_results[df_results['TVD'] == selected_tvd].iloc[0]

           # Criar figura para visualização 2D
           fig_2d, ax_2d = plt.subplots(figsize=(10, 10))

           # Plotar os pontos centrais
           ax_2d.scatter(selected_data['E1'], selected_data['N1'], color='blue', s=50, label=f'Poço 1 Centro (TVD={selected_data["TVD"]:.1f}m)', zorder=5)
           ax_2d.scatter(selected_data['E2'], selected_data['N2'], color='red', s=50, label=f'Poço 2 Centro (TVD={selected_data["TVD"]:.1f}m)', zorder=5)

           # Linha entre centros
           ax_2d.plot([selected_data['E1'], selected_data['E2']], [selected_data['N1'], selected_data['N2']], 'k--', alpha=0.7, label=f'Dist Centros: {selected_data["DistCentros"]:.2f} m')

           # Desenhar as elipses (tamanho real, sem scaling extra)
           # Elipse 1
           # matplotlib angle é anti-horário a partir do eixo +X (East)
           # Nosso ang1 é anti-horário a partir do +Y (North)
           angle_mpl1 = 90.0 - selected_data['Ang1']
           draw_ellipse_matplotlib(
               ax_2d,
               center_xy=(selected_data['E1'], selected_data['N1']),
               width=2 * selected_data['SMj1'], # Diâmetro maior
               height=2 * selected_data['SMn1'],# Diâmetro menor
               angle_deg=angle_mpl1,
               color="blue", alpha=0.3, label=f'Elipse 1 ({sigma_factor:.1f}σ)'
           )

           # Elipse 2
           angle_mpl2 = 90.0 - selected_data['Ang2']
           draw_ellipse_matplotlib(
               ax_2d,
               center_xy=(selected_data['E2'], selected_data['N2']),
               width=2 * selected_data['SMj2'], # Diâmetro maior
               height=2 * selected_data['SMn2'],# Diâmetro menor
               angle_deg=angle_mpl2,
               color="red", alpha=0.3, label=f'Elipse 2 ({sigma_factor:.1f}σ)'
           )

           # Adicionar texto com informações
           info_text = f"""
           TVD: {selected_data['TVD']:.1f} m
           Poço 1: INC={selected_data['INC1']:.1f}°, AZ={selected_data['AZ1']:.1f}°
           Poço 2: INC={selected_data['INC2']:.1f}°, AZ={selected_data['AZ2']:.1f}°
           ---
           Dist. Centros: {selected_data['DistCentros']:.2f} m
           Dist. Pedal ({sigma_factor:.1f}σ): {selected_data['DistPedal']:.2f} m
           Proj1: {selected_data['Proj1']:.2f} m, Proj2: {selected_data['Proj2']:.2f} m
           ---
           Elipse 1 (Semi-eixos): Maj={selected_data['SMj1']:.2f}, Min={selected_data['SMn1']:.2f}, Ang(N)={selected_data['Ang1']:.1f}°
           Elipse 2 (Semi-eixos): Maj={selected_data['SMj2']:.2f}, Min={selected_data['SMn2']:.2f}, Ang(N)={selected_data['Ang2']:.1f}°
           """
           ax_2d.text(0.02, 0.98, info_text, transform=ax_2d.transAxes, fontsize=9,
                      verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

           # Configurações do gráfico 2D
           ax_2d.set_xlabel('Este (m)')
           ax_2d.set_ylabel('Norte (m)')
           ax_2d.set_title(f'Elipses de Incerteza ISCWSA ({tool_type}) @ TVD ≈ {selected_data["TVD"]:.1f} m')
           ax_2d.grid(True, linestyle=':', alpha=0.6)
           ax_2d.axis('equal') # Essencial para visualizar forma correta da elipse
           ax_2d.legend(fontsize=8, loc='lower right')

           # Zoom na área de interesse
           all_x = [selected_data['E1'] - selected_data['SMj1'], selected_data['E1'] + selected_data['SMj1'],
                    selected_data['E2'] - selected_data['SMj2'], selected_data['E2'] + selected_data['SMj2']]
           all_y = [selected_data['N1'] - selected_data['SMj1'], selected_data['N1'] + selected_data['SMj1'],
                    selected_data['N2'] - selected_data['SMj2'], selected_data['N2'] + selected_data['SMj2']]
           center_e = (selected_data['E1'] + selected_data['E2']) / 2
           center_n = (selected_data['N1'] + selected_data['N2']) / 2
           max_range = max(max(all_x) - min(all_x), max(all_y) - min(all_y), selected_data['DistCentros'])
           view_buffer = max_range * 0.6 # Buffer de visualização
           ax_2d.set_xlim(center_e - view_buffer, center_e + view_buffer)
           ax_2d.set_ylim(center_n - view_buffer, center_n + view_buffer)


           st.pyplot(fig_2d)


           # Explicação Pedal Curve (mantida do original, adaptada)
           with st.expander("Explicação do Método Pedal Curve com ISCWSA"):
               st.markdown(f"""
               ### Método Pedal Curve e Modelos ISCWSA

               O método **Pedal Curve** calcula a separação mínima entre as elipses de incerteza de dois poços. A distância é calculada como:

               `Dist_Pedal = max(0, Dist_Centros - (Projeção_Elipse1 + Projeção_Elipse2))`

               Onde a projeção de cada elipse é o seu "raio" na direção que conecta os centros dos dois poços naquele ponto.

               **Modelos ISCWSA (MWD e Gyro):**
               - Diferente da abordagem anterior, estes modelos consideram **múltiplas fontes de erro** definidas pelo usuário (bias de sensores, erros de escala, desalinhamentos, erros de profundidade, interferência magnética/drift do giroscópio, etc.).
               - A incerteza de cada fonte de erro é combinada matematicamente (nesta implementação, por soma de variâncias rotacionadas) para formar uma **matriz de covariância** 3D (Norte, Este, Vertical) em cada ponto da trajetória.
               - A **elipse de incerteza horizontal** (plotada em 2D) é derivada desta matriz de covariância. Seus semi-eixos (maior e menor) e sua orientação **não estão mais necessariamente alinhados com o azimute do poço**, mas sim com as direções de maior e menor incerteza combinada.
               - A **orientação da elipse** (ângulo do eixo maior com o Norte) e a **razão entre os eixos** (forma da elipse) agora dependem complexamente da trajetória (MD, INC, AZ) e de todos os parâmetros de erro da ferramenta ({tool_type}) selecionada.
               - A distância Pedal Curve calculada ({sigma_factor:.1f}σ) reflete a separação considerando estas elipses de incerteza mais realistas. Um valor de 0 indica que as elipses (na confiança {sigma_factor:.1f}σ) estão se tocando ou sobrepondo naquela profundidade.
               """)

       else:
           st.warning("Não foi possível encontrar pontos de comparação em profundidades TVD próximas ou os dados resultantes foram insuficientes.")

   except FileNotFoundError:
       st.error("Erro: Arquivo não encontrado. Verifique o caminho.")
   except pd.errors.EmptyDataError:
       st.error("Erro: O arquivo Excel está vazio.")
   except ValueError as e:
       st.error(f"Erro ao converter dados para números. Verifique o conteúdo das colunas MD, INC, AZ. Detalhe: {e}")
   except Exception as e:
       st.error(f"Erro inesperado ao processar os arquivos: {e}")
       st.exception(e) # Mostra o traceback completo para depuração

else:
   # Exibir exemplo de formato de arquivo esperado
   st.info("Aguardando upload dos arquivos Excel (MD, INC, AZ)...")
   example_data = pd.DataFrame({
       'MD': [0, 500, 1000, 1500, 2000, 2500],
       'INC': [0, 15, 30, 45, 60, 75],
       'AZ': [0, 45, 45, 60, 90, 120]
   })
   st.write("Formato esperado (nomes das colunas podem variar ligeiramente, ex: INCLINACAO, AZIMUTE):")
   st.dataframe(example_data)

   # Adicionar botão para download de exemplo
   buffer = BytesIO()
   with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
       example_data.to_excel(writer, index=False, sheet_name='TrajetoriaExemplo')
   st.download_button(
       label="Download Arquivo Exemplo",
       data=buffer.getvalue(),
       file_name="exemplo_trajetoria.xlsx",
       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
   )
