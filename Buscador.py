# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from collections import Counter
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# --- CONFIGURACIÓN DE LA RUTA_RELATIVA ---
RUTA_CSV = 'Geotodo.csv' 

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Geotodo - Suite Ultimate",
    page_icon="🎲",
    layout="wide"
)

st.title("🎲 Geotodo - Suite de Análisis Avanzado (Corregido)")

# --- FUNCIÓN PARA CARGAR DATOS ---
@st.cache_resource
def cargar_datos_geotodo(_ruta_csv):
    try:
        if not os.path.exists(_ruta_csv):
            st.error(f"❌ Error: No se encontró el archivo {_ruta_csv}.")
            st.stop()
        df = pd.read_csv(_ruta_csv, sep=';', encoding='latin-1')
        df.rename(columns={'Fecha': 'Fecha', 'Tarde/Noche': 'Tipo_Sorteo', 'Fijo': 'Fijo', '1er Corrido': 'Primer_Corrido', '2do Corrido': 'Segundo_Corrido'}, inplace=True)
        df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
        df.dropna(subset=['Fecha'], inplace=True)
        df['Tipo_Sorteo'] = df['Tipo_Sorteo'].astype(str).str.strip().str.upper().map({'MAÑANA': 'M', 'M': 'M', 'MANANA': 'M', 'TARDE': 'T', 'T': 'T', 'NOCHE': 'N', 'N': 'N'}).fillna('OTRO')
        df_fijos = df[['Fecha', 'Tipo_Sorteo', 'Fijo']].copy()
        df_fijos = df_fijos.rename(columns={'Fijo': 'Numero'})
        df_fijos['Numero'] = pd.to_numeric(df_fijos['Numero'], errors='coerce').astype(int)
        draw_order_map = {'M': 0, 'T': 1, 'N': 2}
        df_fijos['draw_order'] = df_fijos['Tipo_Sorteo'].map(draw_order_map).fillna(3)
        df_fijos['sort_key'] = df_fijos['Fecha'] + pd.to_timedelta(df_fijos['draw_order'], unit='h')
        df_fijos = df_fijos.sort_values(by='sort_key').reset_index(drop=True)
        return df_fijos
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        st.stop()

# --- FUNCIÓN 1: PATRONES ---
def analizar_siguientes(df_fijos, numero_busqueda, ventana_sorteos):
    indices = df_fijos[df_fijos['Numero'] == numero_busqueda].index.tolist()
    if not indices: return None, 0 
    lista_s = []
    for idx in indices:
        i = idx + 1; f = idx + ventana_sorteos + 1
        if i < len(df_fijos): lista_s.extend(df_fijos.iloc[i:f]['Numero'].tolist())
    c = Counter(lista_s)
    r = pd.DataFrame.from_dict(c, orient='index', columns=['Frecuencia'])
    r['Probabilidad (%)'] = (r['Frecuencia'] / sum(lista_s) * 100).round(2) if lista_s else 0
    
    # --- CORRECCIÓN AQUÍ: Usar list comprehension en lugar de .apply en el índice ---
    r['Número'] = [f"{int(x):02d}" for x in r.index]
    
    return r.sort_values('Frecuencia', ascending=False), len(indices)

# --- FUNCIÓN 2: ALMANAQUE ---
def analizar_almanaque(df_fijos, dia_inicio, dia_fin, meses_atras):
    fecha_hoy = datetime.now()
    bloques_validos = []
    nombres_bloques = []
    
    for offset in range(1, meses_atras + 1):
        f_obj = fecha_hoy - relativedelta(months=offset)
        try:
            f_i = datetime(f_obj.year, f_obj.month, dia_inicio)
            f_f = datetime(f_obj.year, f_obj.month, dia_fin)
            if f_i > f_f: continue
            df_b = df_fijos[(df_fijos['Fecha'] >= f_i) & (df_fijos['Fecha'] <= f_f)]
            if not df_b.empty:
                bloques_validos.append(df_b)
                nombres_bloques.append(f"{f_i.strftime('%d/%m')}-{f_f.strftime('%d/%m')}")
        except: continue

    if not bloques_validos: return None, "Sin datos.", None, None, None, None, None, None, None, None, None, None
        
    df_total = pd.concat(bloques_validos)
    df_total['Decena'] = df_total['Numero'] // 10
    df_total['Unidad'] = df_total['Numero'] % 10
    
    cnt_d = df_total['Decena'].value_counts().reindex(range(10), fill_value=0)
    cnt_u = df_total['Unidad'].value_counts().reindex(range(10), fill_value=0)
    
    def clasificar(serie):
        df_t = serie.sort_values(ascending=False).reset_index()
        df_t.columns = ['Digito', 'Frecuencia']
        conds = [(df_t.index < 3), (df_t.index < 6)]
        vals = ['🔥 Caliente', '🟡 Tibio']
        df_t['Estado'] = np.select(conds, vals, default='🧊 Frío')
        mapa = {r['Digito']: r['Estado'] for _, r in df_t.iterrows()}
        return df_t, mapa

    df_dec, mapa_d = clasificar(cnt_d)
    df_uni, mapa_u = clasificar(cnt_u)
    
    hot_d = df_dec[df_dec['Estado']=='🔥 Caliente']['Digito'].tolist()
    hot_u = df_uni[df_uni['Estado']=='🔥 Caliente']['Digito'].tolist()
    lista_3x3 = [{'Número': f"{d*10+u:02d}", 'Veces': len(df_total[df_total['Numero']==d*10+u])} for d in hot_d for u in hot_u]
    df_3x3 = pd.DataFrame(lista_3x3).sort_values('Veces', ascending=False)

    ranking = []
    for n, v in df_total['Numero'].value_counts().items():
        d = n//10; u=n%10
        p = f"{mapa_d.get(d,'?')} + {mapa_u.get(u,'?')}"
        ranking.append({'Número': f"{n:02d}", 'Frecuencia': v, 'Perfil': p})
    df_rank = pd.DataFrame(ranking).sort_values('Frecuencia', ascending=False)
    
    tend = df_rank['Perfil'].value_counts().reset_index()
    tend.columns = ['Perfil', 'Frecuencia']
    top_p = tend.iloc[0]['Perfil'] if not tend.empty else "N/A"

    tend_nums = []
    if top_p != "N/A" and " + " in top_p:
        p_dec, p_uni = top_p.split(" + ")
        decs_obj = df_dec[df_dec['Estado'] == p_dec]['Digito'].tolist()
        unis_obj = df_uni[df_uni['Estado'] == p_uni]['Digito'].tolist()
        for d in decs_obj:
            for u in unis_obj:
                n = d*10 + u
                tend_nums.append({'Número': f"{n:02d}", 'Sugerencia (Perfil Tendencia)': f"{p_dec} x {p_uni}"})
    df_tend_nums = pd.DataFrame(tend_nums)

    pers_num = []
    nums_unicos = df_total['Numero'].unique()
    for n in nums_unicos:
        c = sum(1 for b in bloques_validos if n in b['Numero'].values)
        if c == len(bloques_validos):
            perfil_val = df_rank[df_rank['Número']==f"{n:02d}"]['Perfil']
            p = perfil_val.values[0] if not perfil_val.empty else "Desconocido"
            pers_num.append({'Número': f"{n:02d}", 'Perfil': p})
    
    if pers_num:
        df_pers_num = pd.DataFrame(pers_num).sort_values('Número').reset_index(drop=True)
    else:
        df_pers_num = pd.DataFrame(columns=['Número', 'Perfil'])

    sets_perfiles = []
    for df_b in bloques_validos:
        perfiles_en_bloque = set()
        for row in df_b.itertuples():
            d = row.Numero // 10
            u = row.Numero % 10
            ed = mapa_d.get(d, '?')
            eu = mapa_u.get(u, '?')
            perfiles_en_bloque.add(f"{ed} + {eu}")
        sets_perfiles.append(perfiles_en_bloque)
    
    persistentes_perfiles = set.intersection(*sets_perfiles) if sets_perfiles else set()
        
    return df_total, df_dec, df_uni, df_3x3, df_rank, nombres_bloques, df_pers_num, tend, top_p, df_tend_nums, persistentes_perfiles

# --- FUNCIÓN 3: PROPUESTA ---
def generar_sugerencia(df, dias, gap):
    fh = datetime.now()
    df_t = df[df['Fecha'] >= fh - timedelta(days=dias)].copy()
    if df_t.empty: return pd.DataFrame()
    df_t['Dec'] = df_t['Numero']//10; df_t['Uni'] = df_t['Numero']%10
    td = df_t['Dec'].value_counts().head(3).index.tolist()
    tu = df_t['Uni'].value_counts().head(3).index.tolist()
    res = []
    for n in [d*10+u for d in td for u in tu]:
        df_n = df[df['Numero']==n]
        if not df_n.empty:
            g = (fh - df_n['Fecha'].max()).days
            if g >= gap: res.append({'Número': f"{n:02d}", 'Gap': g, 'Estado': "⚡ Muy" if g>gap*1.5 else "✅ Op"})
    return pd.DataFrame(res).sort_values('Gap', ascending=False)

# --- FUNCIÓN 4: SECUENCIA ---
def buscar_seq(df, part, type_, seq):
    try:
        p = [x.strip().upper() for x in seq.replace(',', ' ').split() if x.strip()]
    except: return None, "Error."
    if len(p)==0 or len(p)>5: return None, "Inválido."
    if type_=='digito': v=set(range(10))
    elif type_=='paridad': v={'P','I'}
    elif type_=='altura': v={'A','B'}
    else: return None, "Desconocido."
    try:
        if type_=='digito':
            p = [int(x) for x in p]
            if any(x not in v for x in p): return None, "0-9."
        else:
            if any(x not in v for x in p): return None, f"Usa {', '.join(v)}."
    except: return None, "Conv."
    
    l = []
    for x in df['Numero']:
        val = x//10 if part=='Decena' else x%10
        if type_=='digito': l.append(val)
        elif type_=='paridad': l.append('P' if val%2==0 else 'I')
        elif type_=='altura': l.append('A' if val>=5 else 'B')
    lp = len(p); dat = {}
    for i in range(len(l)-lp):
        if l[i:i+lp]==p:
            sig = l[i+lp]; r = df.iloc[i+lp]
            e = f"{r['Numero']:02d} ({r['Fecha'].strftime('%d/%m/%Y')})"
            if sig not in dat: dat[sig] = {'c':0, 'e':[]}
            dat[sig]['c']+=1
            if len(dat[sig]['e'])<3 and e not in dat[sig]['e']: dat[sig]['e'].append(e)
    if not dat: return None, "No."
    rows = [{'Siguiente':k, 'Frecuencia':v['c'], 'Ejemplos':", ".join(v['e']), 'Prob':0} for k,v in dat.items()]
    df_r = pd.DataFrame(rows)
    df_r['Prob'] = (df_r['Frecuencia']/df_r['Frecuencia'].sum()*100).round(2)
    return df_r.sort_values('Frecuencia', ascending=False), None

# --- MAIN ---
def main():
    df = cargar_datos_geotodo(RUTA_CSV)
    st.sidebar.header("⚙️ Panel")
    with st.sidebar.expander("📂 Datos", expanded=True):
        def m(l, i, ic):
            if not l.empty:
                f = l['Fecha'].max(); n = l[l['Fecha']==f]['Numero'].values[0]
                st.metric(f"{ic} {i}", f"{f.strftime('%d/%m')}", delta=f"{n:02d}")
        m(df[df['Tipo_Sorteo']=='M'], "Mañana", "🌅")
        m(df[df['Tipo_Sorteo']=='T'], "Tarde", "🌞")
        m(df[df['Tipo_Sorteo']=='N'], "Noche", "🌙")

    with st.sidebar.expander("📝 Agregar", expanded=False):
        f = st.date_input("Fecha:", datetime.now().date(), format="DD/MM/YYYY", label_visibility="collapsed")
        s = st.radio("Sesión:", ["Mañana (M)", "Tarde (T)", "Noche (N)"], horizontal=True, label_visibility="collapsed")
        c1, c2 = st.columns(2)
        with c1: fj = st.number_input("Fijo", 0, 99, 0, format="%02d", label_visibility="collapsed")
        with c2: c1v = st.number_input("1er", 0, 99, 0, format="%02d", label_visibility="collapsed")
        p2 = st.number_input("2do", 0, 99, 0, format="%02d", label_visibility="collapsed")
        if st.button("💾", type="primary", use_container_width=True):
            cd = s.split('(')[1].replace(')', '')
            try:
                with open(RUTA_CSV, 'a', encoding='latin-1') as file: file.write(f"{f.strftime('%d/%m/%Y')};{cd};{fj};{c1v};{p2}\n")
                st.success("Ok"); time.sleep(1); st.rerun()
            except Exception as e: st.error(e)

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄"): st.cache_resource.clear(); st.rerun()
    st.sidebar.subheader("🎲 Modo")
    modo = st.sidebar.radio("Filtro:", ["General", "Mañana", "Tarde", "Noche"])
    if "Mañana" in modo: dfa = df[df['Tipo_Sorteo']=='M'].copy(); t="Mañana"
    elif "Tarde" in modo: dfa = df[df['Tipo_Sorteo']=='T'].copy(); t="Tarde"
    elif "Noche" in modo: dfa = df[df['Tipo_Sorteo']=='N'].copy(); t="Noche"
    else: dfa = df.copy(); t="General"
    if dfa.empty: st.stop()
    
    tabs = st.tabs(["🔍 Patrones", "📅 Almanaque Ultimate", "🧠 Propuesta", "🔗 Secuencia"])

    # 1. PATRONES
    with tabs[0]:
        st.subheader(f"Patrones: {t}")
        c1, c2 = st.columns(2)
        with c1: n = st.number_input("Disparador:", 0, 99, 40, format="%02d")
        with c2: v = st.slider("Ventana:", 1, 30, 15)
        if st.button("🔍", key="b1"): st.session_state['sb1'] = True
        if st.session_state.get('sb1'):
            r, tot = analizar_siguientes(dfa, n, v)
            if r is None: st.error("No salió.")
            else: st.dataframe(r.head(20), column_config={"Frecuencia": st.column_config.ProgressColumn("Frecuencia", format="%d", min_value=0, max_value=int(r['Frecuencia'].max()))}, hide_index=True)

    # 2. ALMANAQUE
    with tabs[1]:
        st.subheader(f"Almanaque (Persistencia Total): {t}")
        c_r, c_m = st.columns(2)
        with c_r:
            ca, cb = st.columns(2)
            with ca: di = st.number_input("Ini:", 1, 31, 16)
            with cb: dfi = st.number_input("Fin:", 1, 31, 31)
        with c_m: ma = st.slider("Meses Atrás:", 1, 12, 4)
        
        if st.button("📊", key="b_al"): st.session_state['sal'] = True
        if st.session_state.get('sal'):
            if di > dfi: st.error("Error fechas.")
            else:
                _, dec, uni, comb, rank, noms, pers_n, tend, top_p, tend_nums, pers_p = analizar_almanaque(dfa, di, dfi, ma)
                
                if noms: st.success(f"📅 Bloques: {', '.join(noms)}")
                else: st.error("❌ Sin bloques válidos.")

                cd, cu = st.columns(2)
                with cd: st.markdown("### 🔢 Decenas"); st.dataframe(dec, hide_index=True)
                with cu: st.markdown("### 🔢 Unidades"); st.dataframe(uni, hide_index=True)
                
                st.markdown("---")
                col_t1, col_t2 = st.columns([1, 2])
                with col_t1:
                    st.markdown("### 🔥 Tendencia por Perfil")
                    if not tend.empty:
                        mv = int(tend['Frecuencia'].max())
                        st.dataframe(tend, column_config={"Frecuencia": st.column_config.ProgressColumn("Frecuencia", format="%d", min_value=0, max_value=mv)}, hide_index=True)
                        st.info(f"Dominante: **{top_p}**")
                
                with col_t2:
                    st.markdown("### 💡 Sugerencias")
                    st.dataframe(tend_nums, hide_index=True)

                with st.expander("🛡️ Análisis de Persistencia"):
                    p1, p2 = st.columns(2)
                    with p1:
                        st.markdown("#### 📌 Persistencia de NÚMERO")
                        st.caption("Estos números específicos salieron en TODOS los meses.")
                        st.dataframe(pers_n, hide_index=True)
                    with p2:
                        st.markdown("#### 🏷️ Persistencia de PERFIL")
                        st.caption("Estas ETIQUETAS aparecieron en TODOS los meses.")
                        if pers_p:
                            st.dataframe(pd.DataFrame(list(pers_p), columns=["Perfil Persistente"]), hide_index=True)
                        else:
                            st.info("Ningún perfil se repite en todos los meses.")

                with st.expander("📋 Ranking General"):
                    st.dataframe(rank.head(20), hide_index=True)

    # 3. PROPUESTA
    with tabs[2]:
        st.subheader(f"Sincronización: {t}")
        c1, c2 = st.columns(2)
        with c1: dt = st.number_input("Días Tendencia:", 5, 60, 15)
        with c2: dg = st.number_input("Gap Mínimo:", 1, 90, 10)
        if st.button("🧠", key="b_pr"): st.session_state['spr'] = True
        if st.session_state.get('spr'):
            p = generar_sugerencia(dfa, dt, dg)
            if p.empty: st.warning("No.")
            else: st.dataframe(p, hide_index=True)

    # 4. SECUENCIA
    with tabs[3]:
        st.subheader(f"Secuencia: {t}")
        c1, c2, c3 = st.columns(3)
        with c1: part = st.selectbox("Parte:", ["Decena", "Unidad"])
        with c2: type_ = st.selectbox("Análisis:", ["Dígito (0-9)", "Paridad (P/I)", "Altura (A/B)"])
        with c3: seq = st.text_input("Secuencia:")
        if st.button("🔗", key="b_seq"): st.session_state['sseq'] = True
        if st.session_state.get('sseq') and seq:
            r, e = buscar_seq(dfa, part, type_.lower().replace('(','').replace(')','').split(' ')[0], seq)
            if e: st.warning(e)
            else: st.dataframe(r, column_config={"Siguiente": st.column_config.TextColumn("Sig", width="small"), "Frecuencia": st.column_config.ProgressColumn("Frec", format="%d", min_value=0, max_value=int(r['Frecuencia'].max())), "Prob": st.column_config.NumberColumn("Prob", format="%.2f%%"), "Ejemplos": st.column_config.TextColumn("Hist", width="large")}, hide_index=True)

if __name__ == "__main__":
    main()