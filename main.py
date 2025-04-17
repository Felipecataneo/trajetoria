import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from io import BytesIO
import math

st.set_page_config(page_title="Comparador de Distâncias Pedal Curve", layout="wide")

# Título e explicação
st.title("Comparador de Distâncias em Trajetórias Direcionais")
st.markdown("""
Esta aplicação permite comparar a distância entre centros de trajetórias com a distância calculada pelo método Pedal Curve,
que considera as elipses de incerteza. O método Pedal Curve é utilizado no Compass para análise de anti-colisão.
""")

# Parâmetros de incerteza do MWD padrão
st.sidebar.header("Parâmetros de Incerteza MWD")
bias_n = st.sidebar.number_input("Bias-N (m/100m)", value=0.35, step=0.05, format="%.2f")
bias_e = st.sidebar.number_input("Bias-E (m/100m)", value=0.35, step=0.05, format="%.2f")
bias_v = st.sidebar.number_input("Bias-V (m/100m)", value=0.35, step=0.05, format="%.2f")
drfr = st.sidebar.number_input("DRFR (m/100m)", value=0.30, step=0.05, format="%.2f")
dref = st.sidebar.number_input("DREF (deg)", value=0.50, step=0.05, format="%.2f")
az_mn = st.sidebar.number_input("AZ MN (deg)", value=0.50, step=0.05, format="%.2f")
inc_err = st.sidebar.number_input("INC (deg)", value=0.15, step=0.05, format="%.2f")
toolface = st.sidebar.number_input("Toolface (deg)", value=0.40, step=0.05, format="%.2f")

# Funções de cálculo para trajetória e incerteza
def deg_to_rad(deg):
    return deg * np.pi / 180.0

def rad_to_deg(rad):
    return rad * 180.0 / np.pi

def calculate_coordinates(md, inc, az):
    """Calcula coordenadas usando método minimum curvature"""
    n = [0]
    e = [0]
    tvd = [0]
   
    for i in range(1, len(md)):
        segment = md[i] - md[i-1]
        inc1_rad = deg_to_rad(inc[i-1])
        inc2_rad = deg_to_rad(inc[i])
        az1_rad = deg_to_rad(az[i-1])
        az2_rad = deg_to_rad(az[i])
       
        # Fator de correção de minimum curvature
        cos_dls = np.cos(inc2_rad - inc1_rad) - np.sin(inc1_rad) * np.sin(inc2_rad) * (1 - np.cos(az2_rad - az1_rad))
        dls = np.arccos(min(max(cos_dls, -1.0), 1.0))  # Evitar erros de domínio
       
        if abs(dls) < 1e-6:
            rf = 1.0
        else:
            rf = 2.0 * np.tan(dls/2) / dls
       
        # Incrementos de coordenadas
        dn = segment/2 * (np.sin(inc1_rad) * np.cos(az1_rad) + np.sin(inc2_rad) * np.cos(az2_rad)) * rf
        de = segment/2 * (np.sin(inc1_rad) * np.sin(az1_rad) + np.sin(inc2_rad) * np.sin(az2_rad)) * rf
        dv = segment/2 * (np.cos(inc1_rad) + np.cos(inc2_rad)) * rf
       
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

def calculate_ellipse_axes(md, inc, az, error_params):
    """Calcula os semi-eixos da elipse de incerteza"""
    inc_rad = deg_to_rad(inc)
   
    # Parâmetros de erro
    bias_n, bias_e, bias_v, drfr, dref, az_mn, inc_err, toolface = error_params
   
    # Cálculo simplificado da elipse de incerteza
    sigma_r = np.sqrt((md * drfr/100)**2 + (np.sin(inc_rad) * dref * md/100)**2)
    sigma_a = np.sqrt((md * drfr/100)**2 + (np.cos(inc_rad) * dref * md/100)**2)
    sigma_v = bias_v * md/100 + inc_err * md/100 * np.sin(inc_rad)
   
    return sigma_r, sigma_a, sigma_v

def calculate_distance(p1, p2):
    """Calcula a distância euclidiana entre dois pontos"""
    return np.sqrt((p2['N'] - p1['N'])**2 + (p2['E'] - p1['E'])**2)

def project_ellipse(sigma_r, sigma_a, well_az, direction_az):
    """Projeta a elipse na direção especificada"""
    rel_angle = deg_to_rad(direction_az - well_az)
   
    # Projeção da elipse (pedal curve)
    numerator = sigma_r * sigma_a
    denominator = np.sqrt((sigma_a * np.sin(rel_angle))**2 + (sigma_r * np.cos(rel_angle))**2)
   
    return numerator / denominator

def calculate_pedal_distance(p1, p2, error_params):
    """Calcula a distância Pedal Curve entre dois pontos"""
    # Distância entre centros
    center_dist = calculate_distance(p1, p2)
   
    # Direção entre centros
    angle_rad = np.arctan2(p2['E'] - p1['E'], p2['N'] - p1['N'])
    angle_deg = rad_to_deg(angle_rad)
   
    # Calcular semi-eixos das elipses
    sigma_r1, sigma_a1, _ = calculate_ellipse_axes(p1['MD'], p1['INC'], p1['AZ'], error_params)
    sigma_r2, sigma_a2, _ = calculate_ellipse_axes(p2['MD'], p2['INC'], p2['AZ'], error_params)
   
    # Projeção das elipses
    proj1 = project_ellipse(sigma_r1, sigma_a1, p1['AZ'], angle_deg)
    proj2 = project_ellipse(sigma_r2, sigma_a2, p2['AZ'], angle_deg)
   
    # Distância Pedal Curve
    pedal_dist = max(0, center_dist - (proj1 + proj2))
   
    return {
        'center_dist': center_dist,
        'proj1': proj1,
        'proj2': proj2,
        'pedal_dist': pedal_dist,
        'difference': center_dist - pedal_dist,
        'diff_percent': ((center_dist - pedal_dist) / center_dist) * 100 if center_dist > 0 else 0
    }

def find_closest_tvd_point(tvd_target, df):
    """Encontra o ponto mais próximo em TVD"""
    idx = (df['TVD'] - tvd_target).abs().idxmin()
    return df.loc[idx]

def draw_ellipse(ax, pos, width, height, angle, color="blue", alpha=0.3):
    """Desenha uma elipse com os parâmetros especificados"""
    ellipse = Ellipse(pos, width, height, angle=angle, color=color, alpha=alpha)
    ax.add_patch(ellipse)
    return ellipse

# Interface para upload de arquivos
col1, col2 = st.columns(2)

with col1:
    st.header("Poço 1")
    well1_file = st.file_uploader("Upload do Excel com MD, INC, AZ do Poço 1", type=["xlsx", "xls"])

with col2:
    st.header("Poço 2")
    well2_file = st.file_uploader("Upload do Excel com MD, INC, AZ do Poço 2", type=["xlsx", "xls"])

# Processamento quando ambos os arquivos são carregados
if well1_file and well2_file:
    # Leitura dos arquivos Excel
    try:
        df_well1 = pd.read_excel(well1_file)
        df_well2 = pd.read_excel(well2_file)
       
        # Verificar e padronizar nomes das colunas
        expected_cols = ['MD', 'INC', 'AZ']
       
        for df, well_name in [(df_well1, "Poço 1"), (df_well2, "Poço 2")]:
            # Verificar se as colunas esperadas existem (ignorando case)
            cols = [col for col in df.columns if col.upper() in [ec.upper() for ec in expected_cols]]
           
            if len(cols) != len(expected_cols):
                st.error(f"O arquivo do {well_name} deve conter as colunas: MD, INC, AZ (Profundidade Medida, Inclinação, Azimute)")
                st.stop()
           
            # Renomear colunas para o formato esperado
            col_mapping = {}
            for col in df.columns:
                if col.upper() == 'MD':
                    col_mapping[col] = 'MD'
                elif col.upper() in ['INC', 'INCLINACAO', 'INCLINAÇÃO']:
                    col_mapping[col] = 'INC'
                elif col.upper() in ['AZ', 'AZIMUTH', 'AZIMUTE']:
                    col_mapping[col] = 'AZ'
           
            df.rename(columns=col_mapping, inplace=True)
       
        # Exibir dados carregados
        st.subheader("Dados Carregados")
        col1, col2 = st.columns(2)
       
        with col1:
            st.write("Poço 1:")
            st.dataframe(df_well1[['MD', 'INC', 'AZ']])
       
        with col2:
            st.write("Poço 2:")
            st.dataframe(df_well2[['MD', 'INC', 'AZ']])
       
        # Calcular coordenadas
        coords_well1 = calculate_coordinates(
            df_well1['MD'].values,
            df_well1['INC'].values,
            df_well1['AZ'].values
        )
       
        coords_well2 = calculate_coordinates(
            df_well2['MD'].values,
            df_well2['INC'].values,
            df_well2['AZ'].values
        )
       
        # Parâmetros de erro MWD
        error_params = (bias_n, bias_e, bias_v, drfr, dref, az_mn, inc_err, toolface)
       
        # Encontrar TVDs comuns ou próximas para comparação
        tvds_well1 = coords_well1['TVD'].unique()
       
        results = []
       
        for tvd in tvds_well1:
            # Encontrar pontos mais próximos em TVD
            p1 = find_closest_tvd_point(tvd, coords_well1)
            p2 = find_closest_tvd_point(tvd, coords_well2)
           
            # Apenas comparar se a diferença de TVD for pequena (menos de 5m)
            if abs(p1['TVD'] - p2['TVD']) < 5:
                # Calcular distâncias
                distance_data = calculate_pedal_distance(p1, p2, error_params)
               
                # Adicionar aos resultados
                results.append({
                    'TVD': p1['TVD'],
                    'MD1': p1['MD'],
                    'MD2': p2['MD'],
                    'INC1': p1['INC'],
                    'INC2': p2['INC'],
                    'AZ1': p1['AZ'],
                    'AZ2': p2['AZ'],
                    'N1': p1['N'],
                    'E1': p1['E'],
                    'N2': p2['N'],
                    'E2': p2['E'],
                    'DistCentros': distance_data['center_dist'],
                    'DistPedal': distance_data['pedal_dist'],
                    'Proj1': distance_data['proj1'],
                    'Proj2': distance_data['proj2'],
                    'DifPerc': distance_data['diff_percent'],
                    'AZ_Diff': abs(p1['AZ'] - p2['AZ']) % 180,  # Diferença angular menor que 180
                    'INC_Avg': (p1['INC'] + p2['INC']) / 2
                })
       
        # Criar dataframe de resultados
        if results:
            df_results = pd.DataFrame(results)
           
            # Exibir tabela de resultados
            st.subheader("Comparação de Distâncias")
            st.dataframe(df_results[['TVD', 'DistCentros', 'DistPedal', 'DifPerc', 'INC1', 'INC2', 'AZ1', 'AZ2']])
           
            # Gráfico de distâncias vs TVD
            st.subheader("Distâncias vs Profundidade")
           
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=df_results['TVD'],
                y=df_results['DistCentros'],
                mode='lines+markers',
                name='Distância entre Centros'
            ))
            fig1.add_trace(go.Scatter(
                x=df_results['TVD'],
                y=df_results['DistPedal'],
                mode='lines+markers',
                name='Distância Pedal Curve'
            ))
           
            fig1.update_layout(
                title='Comparação de Distâncias vs Profundidade',
                xaxis_title='TVD (m)',
                yaxis_title='Distância (m)',
                legend=dict(x=0, y=1, traceorder='normal'),
                height=500
            )
           
            st.plotly_chart(fig1, use_container_width=True)
           
            # Gráfico da diferença percentual vs TVD
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=df_results['TVD'],
                y=df_results['DifPerc'],
                mode='lines+markers',
                name='Diferença Percentual'
            ))
           
            fig2.update_layout(
                title='Diferença Percentual vs Profundidade',
                xaxis_title='TVD (m)',
                yaxis_title='Diferença Percentual (%)',
                height=400
            )
           
            st.plotly_chart(fig2, use_container_width=True)
           
            # Visualização dos efeitos de inclinação e azimute
            col1, col2 = st.columns(2)
           
            with col1:
                # Efeito da diferença de azimute
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(
                    x=df_results['AZ_Diff'],
                    y=df_results['DifPerc'],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=df_results['INC_Avg'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title='Inclinação Média (°)')
                    ),
                    name='Pontos'
                ))
               
                fig3.update_layout(
                    title='Efeito da Diferença de Azimute',
                    xaxis_title='Diferença de Azimute (°)',
                    yaxis_title='Diferença Percentual (%)',
                    height=500
                )
               
                st.plotly_chart(fig3, use_container_width=True)
           
            with col2:
                # Efeito da inclinação média
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(
                    x=df_results['INC_Avg'],
                    y=df_results['DifPerc'],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=df_results['AZ_Diff'],
                        colorscale='Plasma',
                        showscale=True,
                        colorbar=dict(title='Diferença de Azimute (°)')
                    ),
                    name='Pontos'
                ))
               
                fig4.update_layout(
                    title='Efeito da Inclinação Média',
                    xaxis_title='Inclinação Média (°)',
                    yaxis_title='Diferença Percentual (%)',
                    height=500
                )
               
                st.plotly_chart(fig4, use_container_width=True)
           
            # Visualização 3D das trajetórias
            st.subheader("Visualização 3D das Trajetórias")
           
            fig5 = go.Figure()
            fig5.add_trace(go.Scatter3d(
                x=coords_well1['E'],
                y=coords_well1['N'],
                z=coords_well1['TVD'],
                mode='lines+markers',
                name='Poço 1',
                marker=dict(size=4),
                line=dict(width=3, color='blue')
            ))
           
            fig5.add_trace(go.Scatter3d(
                x=coords_well2['E'],
                y=coords_well2['N'],
                z=coords_well2['TVD'],
                mode='lines+markers',
                name='Poço 2',
                marker=dict(size=4),
                line=dict(width=3, color='red')
            ))
           
            fig5.update_layout(
                scene=dict(
                    xaxis_title='Este (m)',
                    yaxis_title='Norte (m)',
                    zaxis_title='TVD (m)',
                    zaxis=dict(autorange="reversed")  # Profundidade aumenta para baixo
                ),
                height=800
            )
           
            st.plotly_chart(fig5, use_container_width=True)
           
            # Visualização 2D das trajetórias com elipses de incerteza
            st.subheader("Visualização 2D com Elipses de Incerteza")
           
            # Opções para selecionar a profundidade TVD para visualização
            tvd_options = np.sort(df_results['TVD'].unique())
            selected_tvd = st.selectbox("Selecione a profundidade TVD para visualização", tvd_options)
           
            # Filtrar dados para a TVD selecionada
            selected_data = df_results[df_results['TVD'] == selected_tvd].iloc[0]
           
            # Criar figura para visualização 2D
            fig, ax = plt.subplots(figsize=(10, 8))
           
            # Plotar os pontos
            ax.scatter(selected_data['E1'], selected_data['N1'], color='blue', s=50, label='Poço 1')
            ax.scatter(selected_data['E2'], selected_data['N2'], color='red', s=50, label='Poço 2')
           
            # Linha entre centros
            ax.plot([selected_data['E1'], selected_data['E2']], [selected_data['N1'], selected_data['N2']], 'k--', alpha=0.7)
           
            # Calcular elipses
            # Obter os parâmetros da elipse para o ponto 1
            sigma_r1, sigma_a1, _ = calculate_ellipse_axes(
                selected_data['MD1'],
                selected_data['INC1'],
                selected_data['AZ1'],
                error_params
            )
           
            # Obter os parâmetros da elipse para o ponto 2
            sigma_r2, sigma_a2, _ = calculate_ellipse_axes(
                selected_data['MD2'],
                selected_data['INC2'],
                selected_data['AZ2'],
                error_params
            )
           
            # Desenhar as elipses - multiplicamos por um fator para torná-las mais visíveis
            scale_factor = 5  # Para visualização mais clara
            draw_ellipse(
                ax,
                (selected_data['E1'], selected_data['N1']),
                sigma_a1 * scale_factor,
                sigma_r1 * scale_factor,
                selected_data['AZ1'],
                "blue",
                0.3
            )
           
            draw_ellipse(
                ax,
                (selected_data['E2'], selected_data['N2']),
                sigma_a2 * scale_factor,
                sigma_r2 * scale_factor,
                selected_data['AZ2'],
                "red",
                0.3
            )
           
            # Adicionar linha de texto com informações
            info_text = f"""
            TVD: {selected_data['TVD']:.1f} m
            Inclinação Poço 1: {selected_data['INC1']:.1f}°, Azimute Poço 1: {selected_data['AZ1']:.1f}°
            Inclinação Poço 2: {selected_data['INC2']:.1f}°, Azimute Poço 2: {selected_data['AZ2']:.1f}°
            Distância entre Centros: {selected_data['DistCentros']:.2f} m
            Dist. Pedal Curve: {selected_data['DistPedal']:.2f} m
            Diferença: {selected_data['DistCentros'] - selected_data['DistPedal']:.2f} m ({selected_data['DifPerc']:.1f}%)
            """
           
            ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
           
            # Configurações do gráfico
            ax.set_xlabel('Este (m)')
            ax.set_ylabel('Norte (m)')
            ax.set_title(f'Elipses de Incerteza na TVD = {selected_data["TVD"]:.1f} m')
            ax.grid(True)
            ax.axis('equal')
            ax.legend()
           
            # Adicionar anotação explicando as elipses (escaladas para visualização)
            ax.annotate(f'Elipses escaladas por fator {scale_factor}x para melhor visualização',
                       xy=(0.5, 0.01), xycoords='axes fraction',
                       ha='center', fontsize=8)
           
            # Exibir o gráfico no Streamlit
            st.pyplot(fig)
           
            # Informações técnicas sobre o método Pedal Curve
            with st.expander("Explicação do Método Pedal Curve"):
                st.markdown("""
                ### Método Pedal Curve

                O método **Pedal Curve** (ou Curva Pedal) para cálculo de distância entre elipses de incerteza é utilizado em softwares de anti-colisão como o Compass para representar a separação efetiva entre poços, levando em conta a incerteza posicional de cada trajetória.

                **Como funciona:**
                1. Para cada ponto da trajetória, calcula-se uma elipse de incerteza baseada em parâmetros do MWD/LWD utilizado
                2. A orientação e forma da elipse dependem da inclinação e azimute do poço naquele ponto
                3. O método projeta estas elipses na direção que conecta os centros dos dois poços
                4. A distância Pedal Curve é calculada como: Distância entre centros - (Projeção da elipse 1 + Projeção da elipse 2)

                **Por que a distância é diferente da distância entre centros:**
                - Em poços com alta inclinação, a elipse é tipicamente mais "alongada" perpendicular ao eixo do poço
                - Quando os azimutes dos poços são diferentes, as elipses têm orientações diferentes
                - A projeção destas elipses na direção entre os centros resulta em diferentes contribuições para a redução da distância

                O efeito da inclinação e diferença de azimute na distância calculada é o que está sendo visualizado nesta aplicação.
                """)
        else:
            st.warning("Não foi possível encontrar pontos de comparação nas mesmas profundidades.")
       
    except Exception as e:
        st.error(f"Erro ao processar os arquivos: {e}")
        st.exception(e)
else:
    # Exibir exemplo de formato de arquivo esperado
    st.info("Aguardando upload dos arquivos Excel...")
   
    # Criar exemplo de formato
    example_data = pd.DataFrame({
        'MD': [0, 500, 1000, 1500, 2000, 2500],
        'INC': [0, 15, 30, 45, 60, 75],
        'AZ': [0, 45, 45, 60, 90, 120]
    })
   
    st.write("Formato esperado dos arquivos Excel:")
    st.dataframe(example_data)
   
    # Adicionar botão para download de exemplo
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        example_data.to_excel(writer, index=False, sheet_name='Trajetória')
   
    st.download_button(
        label="Download do arquivo de exemplo",
        data=buffer.getvalue(),
        file_name="exemplo_trajetoria.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
