import os
import re
import inspect
import datetime
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from sora.extra import draw_ellipse


def _ordinal(n):
    """Retorna o sufixo ordinal em formato LaTeX para um dia do mês (ex: 7 -> 7^{th})."""
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}$^{{\\text{{{suffix}}}}}$"


def _latex_escape(text):
    """Escapa caracteres especiais do LaTeX em strings comuns."""
    if text is None:
        return ""
    text = str(text)
    replacements = {
        '&': r'\&',
        '%': r'\%',
        '#': r'\#',
        '_': r'\_',
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def _to_float(val, default=np.nan):
    """Extrai o valor numérico (float) de um Astropy Quantity, PhysicalData ou número puro."""
    if val is None:
        return default
    if hasattr(val, 'value'):
        return float(val.value)
    try:
        return float(val)
    except Exception:
        return default


def _format_dms_latex(coord_obj, is_lon=False):
    """
    Formata uma coordenada Astropy Angle/Longitude/Latitude no formato do relatório:
    ex: 50^\\circ 15' 57.24'' W  ou  24^\\circ 34' 25.14'' S
    """
    val = coord_obj.deg
    if is_lon:
        if val > 180:
            val -= 360
        hem = 'E' if val >= 0 else 'W'
    else:
        hem = 'N' if val >= 0 else 'S'

    abs_val = abs(val)
    d = int(abs_val)
    m_full = (abs_val - d) * 60.0
    m = int(m_full)
    s = (m_full - m) * 60.0
    s_str = f"{s:.4f}".rstrip('0').rstrip('.') if f"{s:.4f}" != "0.0000" else "00"
    if '.' in s_str and len(s_str.split('.')[1]) < 2:
        s_str = f"{s:.2f}"
    return f"{d}$^\\circ$ {m:02d}' {s_str}'' {hem}"


def _compute_radial_velocities(occ):
    """
    Calcula as velocidades radiais de imersão e emersão (km/s) para cada corda,
    seguindo exatamente a implementação de `occ.check_velocities()` do SORA.
    """
    rad_vels = {}
    if hasattr(occ, 'fitted_params') and occ.fitted_params:
        center = np.array([occ.fitted_params['center_f'][0], occ.fitted_params['center_g'][0]])
        major_axis = occ.fitted_params['equatorial_radius'][0]
        minor_axis = major_axis * (1.0 - occ.fitted_params['oblateness'][0])
    else:
        center = np.array([0.0, 0.0])
        major_axis = _to_float(getattr(occ.body, 'radius', None), 50.0)
        minor_axis = major_axis

    for name, chord in occ.chords.items():
        rad_vels[name] = {'immersion': None, 'emersion': None, 'vel_used': getattr(chord.lightcurve, 'vel', None)}
        for key in ['immersion', 'emersion']:
            t_inst = getattr(chord.lightcurve, key, None)
            if t_inst is not None:
                try:
                    vals = chord.get_fg(time=t_inst, vel=True)
                    x, y = vals[:2] - center
                    ang = np.arctan((-x / y) * np.power(minor_axis / major_axis, 2)) + np.pi / 2.0
                    observer_vec = np.array([np.cos(ang), np.sin(ang)])
                    normal_vel = np.abs(np.dot(observer_vec, np.array(vals[2:])) / np.linalg.norm(observer_vec))
                    rad_vels[name][key] = float(normal_vel)
                except Exception:
                    rad_vels[name][key] = float(abs(getattr(chord.lightcurve, 'vel', 0.0)))
    return rad_vels


def _extract_astrometric_results(occ, occ_full_text):
    """
    Extrai os dados da posição astrométrica refinada e offsets da efeméride
    gerados por `occ.new_astrometric_position()`.
    """
    astrom_text = ""
    if hasattr(occ, 'fitted_params') and occ.fitted_params:
        try:
            astrom_text = occ.new_astrometric_position(verbose=False)
        except Exception:
            astrom_text = occ_full_text
    else:
        astrom_text = occ_full_text

    res = {}

    # Ephemeris offset (km): X = 42.3 km +/- 8.6 km; Y = -22.0 km +/- 10.6 km
    m_km = re.search(
        r'Ephemeris offset \(km\):\s*X\s*=\s*([-\d\.]+)(?:\s*km)?\s*\+/-\s*([-\d\.]+)(?:\s*km)?;\s*Y\s*=\s*([-\d\.]+)(?:\s*km)?\s*\+/-\s*([-\d\.]+)',
        astrom_text
    )
    if m_km:
        res['x_km'] = f"${float(m_km.group(1)):.1f} \\pm {float(m_km.group(2)):.1f}$ km"
        res['y_km'] = f"${float(m_km.group(3)):.1f} \\pm {float(m_km.group(4)):.1f}$ km"

    # Ephemeris offset (mas): da_cos_dec = 11.903 +/- 2.407; d_dec = -6.189 +/- 2.996
    m_mas = re.search(
        r'Ephemeris offset \(mas\):\s*da_cos_dec\s*=\s*([-\d\.]+)\s*\+/-\s*([-\d\.]+);\s*d_dec\s*=\s*([-\d\.]+)\s*\+/-\s*([-\d\.]+)',
        astrom_text
    )
    if m_mas:
        res['da_cos_dec'] = f"${float(m_mas.group(1)):.3f} \\pm {float(m_mas.group(2)):.3f}$ mas"
        res['d_dec'] = f"${float(m_mas.group(3)):.3f} \\pm {float(m_mas.group(4)):.3f}$ mas"

    # Astrometric object position at time 2026-05-14 03:34:26.120 for reference 'geocenter'
    m_epoch = re.search(
        r'Astrometric object position at time\s+([\d\-\s\.:]+)\s+for reference\s+[\'"]?([^\'"\n]+)[\'"]?',
        astrom_text
    )
    if m_epoch:
        res['epoch'] = f"{m_epoch.group(1).strip()} UTC"
        res['reference'] = m_epoch.group(2).strip()

    # RA = 11 16 05.6294515 +/- 2.412 mas; DEC = -6 23 13.033410 +/- 3.001 mas
    m_pos = re.search(
        r'RA\s*=\s*([\d\s\.]+)\s*\+/-\s*([\d\.]+)\s*mas;\s*DEC\s*=\s*([-\+\d\s\.]+)\s*\+/-\s*([\d\.]+)\s*mas',
        astrom_text
    )
    if m_pos:
        res['ra'] = m_pos.group(1).strip()
        res['ra_err'] = f"$\\pm {float(m_pos.group(2)):.3f}$ mas"
        res['dec'] = m_pos.group(3).strip()
        res['dec_err'] = f"$\\pm {float(m_pos.group(4)):.3f}$ mas"
        res['ra_full'] = f"{res['ra']} ({float(m_pos.group(2)):.3f} mas)"
        res['dec_full'] = f"{res['dec']} ({float(m_pos.group(4)):.3f} mas)"

    return res


def _auto_resolve_chi2(occ, lc_chi2_dict, caller_ns):
    """
    Resolve automaticamente os objetos ChiSquare de imersão e emersão para cada corda,
    aceitando no `lc_chi2_dict` (ou buscando automaticamente no namespace do Jupyter Notebook):
      - O nome da corda (ex: 'Johannes')
      - O nome da variável da curva de luz (ex: 'lc1', 'lc2')
      - O objeto LightCurve diretamente (ex: lc1)
      - As variáveis padrão do notebook: `obs1_chi2`, `obs1_chi2_i`, `obs1_chi2_e`, etc.
    """
    resolved = {}
    lc_chi2_dict = lc_chi2_dict or {}

    lc_id_to_idx = {}
    lc_id_to_varname = {}
    for var_name, var_val in caller_ns.items():
        m = re.match(r'^lc(\d+)$', var_name)
        if m:
            lc_id_to_idx[id(var_val)] = m.group(1)
            lc_id_to_varname[id(var_val)] = var_name

    for idx_enum, (name, chord) in enumerate(occ.chords.items(), start=1):
        lc = chord.lightcurve
        lc_num = lc_id_to_idx.get(id(lc), str(idx_enum))
        lc_varname = lc_id_to_varname.get(id(lc), f"lc{idx_enum}")

        entry = None
        for candidate_key in [name, lc_varname, f"lc{idx_enum}", f"obs{lc_num}", f"c{lc_num}", lc]:
            try:
                if candidate_key in lc_chi2_dict:
                    entry = lc_chi2_dict[candidate_key]
                    break
            except TypeError:
                pass

        if entry is None and caller_ns:
            auto_i = caller_ns.get(f"obs{lc_num}_chi2_i") or caller_ns.get(f"lc{lc_num}_chi2_i")
            auto_e = caller_ns.get(f"obs{lc_num}_chi2_e") or caller_ns.get(f"lc{lc_num}_chi2_e")
            auto_both = caller_ns.get(f"obs{lc_num}_chi2") or caller_ns.get(f"lc{lc_num}_chi2")
            if auto_i is not None or auto_e is not None:
                entry = {
                    'immersion': auto_i if auto_i is not None else getattr(lc, 'chisquare', None),
                    'emersion': auto_e if auto_e is not None else getattr(lc, 'chisquare', None),
                }
            elif auto_both is not None:
                entry = auto_both

        if entry is None:
            entry = getattr(lc, 'chisquare', None)

        if isinstance(entry, dict):
            chi2_i = entry.get('immersion', getattr(lc, 'chisquare', None))
            chi2_e = entry.get('emersion', getattr(lc, 'chisquare', None))
        elif isinstance(entry, (tuple, list)) and len(entry) == 2:
            chi2_i, chi2_e = entry[0], entry[1]
        else:
            chi2_i = entry
            chi2_e = entry

        resolved[name] = {'immersion': chi2_i, 'emersion': chi2_e}

    return resolved


def generate_sora_latex_report(
    occ,
    ellipse_chi2=None,
    lc_chi2_dict=None,
    observer_metadata=None,
    extra_observers=None,
    author="João David Moreira",
    report_date=None,
    preliminary=True,
    pred_map_file="pred_map.png",
    output_tex="relatorio_ocultacao.tex",
    figures_dir="figures_report",
    chord_colors=None,
    ellipse_sigma_plot=1,
    lucky_star_url=None,
    occultation_portal_url=None,
    portal_url=None,
    star_comment=None,
    campaign_comment=None,
):
    """
    Gera automaticamente um relatório completo em LaTeX (.tex) e todas as figuras
    associadas a partir de uma análise de ocultação estelar no SORA.

    Parâmetros principais adicionais
    --------------------------------
    lucky_star_url : str, opcional
        URL da predição/evento no projeto Lucky Star (ex: "https://lesia.obspm.fr/lucky-star/...").
    occultation_portal_url : str, opcional
        URL da campanha/evento no Occultation Portal (ex: "https://occultation.tug.tubitak.gov.tr/...").
    portal_url : str, opcional
        Atalho genérico para URL do evento (se contiver 'lucky-star' ou 'obspm' associa ao Lucky Star,
        caso contrário associa ao Occultation Portal).
    """
    os.makedirs(figures_dir, exist_ok=True)
    observer_metadata = observer_metadata or {}
    extra_observers = extra_observers or []

    # Tratamento do atalho portal_url caso o usuário passe um único link
    if portal_url:
        if ("lucky-star" in portal_url.lower() or "obspm.fr" in portal_url.lower()) and not lucky_star_url:
            lucky_star_url = portal_url
        elif not occultation_portal_url:
            occultation_portal_url = portal_url

    caller_ns = {}
    try:
        frame = inspect.currentframe()
        if frame is not None and frame.f_back is not None:
            caller_ns = frame.f_back.f_globals.copy()
            caller_ns.update(frame.f_back.f_locals)
    except Exception:
        pass

    if ellipse_chi2 is None and 'ellipse_chi2' in caller_ns:
        ellipse_chi2 = caller_ns['ellipse_chi2']

    resolved_chi2 = _auto_resolve_chi2(occ, lc_chi2_dict, caller_ns)

    # 1. Captura do print(occ) completo (usado no Listing 1 e como fallback de parsing)
    occ_full_text = str(occ)

    # 2. Dados básicos do evento, corpo e estrela
    star = occ.star
    body = occ.body

    tca_dt = occ.tca.to_datetime()
    months_en = [
        "", "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    month_name = months_en[tca_dt.month]
    day_ord = _ordinal(tca_dt.day)
    year_str = str(tca_dt.year)
    event_date_text = f"{year_str} {month_name} {tca_dt.day}"

    if report_date is None:
        now = datetime.date.today()
        report_date = f"{months_en[now.month]} {now.day}, {now.year}"

    body_short_name = getattr(body, 'shortname', None) or body.name
    m_num_name = re.match(r'^(\d+)\s+([^(]+)', body.name.strip())
    if m_num_name:
        body_num = m_num_name.group(1)
        body_clean = m_num_name.group(2).strip()
        body_paren_name = f"({body_num}) {body_clean}"
        body_prose_name = f"{body_num} {body_clean}"
        title_body_name = body_clean
    else:
        body_paren_name = body.name.strip()
        body_prose_name = body.name.strip()
        title_body_name = body_short_name

    body_class = getattr(body, 'orbit_class', 'Asteroid')

    pos_chords = [(name, ch) for name, ch in occ.chords.items() if ch.status() == 'positive']
    neg_chords = [(name, ch) for name, ch in occ.chords.items() if ch.status() == 'negative']
    all_chords = list(occ.chords.items())

    num_words = {
        0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four',
        5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten'
    }
    n_pos = len(pos_chords)
    n_neg = len(neg_chords)
    n_tot = len(all_chords)
    n_pos_word = num_words.get(n_pos, str(n_pos))
    n_tot_word = num_words.get(n_tot, str(n_tot))

    # 3. Extração de parâmetros da Estrela (Tabela 1)
    star_code = getattr(star, 'code', 'Unknown')
    star_cat = "Gaia DR3" if "GaiaDR3" in getattr(star, '_catalogue', 'GaiaDR3') else getattr(star, '_catalogue', 'Gaia DR3')
    ra_hms = star.ra.to_string(u.hourangle, sep=' ', precision=5, pad=True)
    dec_dms = star.dec.to_string(u.deg, sep=' ', precision=4, alwayssign=True, pad=True)
    astrometric_pos_icrs = f"{ra_hms} {dec_dms}"

    ra_err_mas = star.errors['ra'].to(u.mas).value if hasattr(star.errors['ra'], 'to') else float(star.errors['ra'])
    dec_err_mas = star.errors['dec'].to(u.mas).value if hasattr(star.errors['dec'], 'to') else float(star.errors['dec'])
    m_star_err = re.search(
        r'ICRS star coordinate at J[\d\.]+:\s*\nRA=\S+\s*\+/-\s*([\d\.]+)\s*(\w+),\s*DEC=\S+\s*\+/-\s*([\d\.]+)\s*(\w+)',
        occ_full_text
    )
    if m_star_err:
        ra_err_val = float(m_star_err.group(1))
        dec_err_val = float(m_star_err.group(3))
    else:
        ra_err_val = ra_err_mas
        dec_err_val = dec_err_mas
    star_pos_error_str = f"RA = {ra_err_val:.4f} mas; DEC = {dec_err_val:.4f} mas"

    mags = star.mag
    line1_bands = [b for b in ['G', 'V', 'R', 'B'] if b in mags]
    if 'G' in mags and 'R' in mags and 'B' in mags:
        line1_bands = ['G', 'R', 'B']
    line2_bands = [b for b in ['J', 'H', 'K'] if b in mags]
    mag_line1 = "; ".join([f"{b} = {mags[b]:.3f}" for b in line1_bands])
    mag_line2 = "; ".join([f"{b} = {mags[b]:.3f}" for b in line2_bands])

    # Distância do objeto e diâmetro estelar em mas e km (sempre exibidos juntos!)
    if len(all_chords) > 0 and hasattr(all_chords[0][1].lightcurve, 'd_star'):
        d_star_km = _to_float(all_chords[0][1].lightcurve.d_star, 0.0)
        obj_dist_au = _to_float(all_chords[0][1].lightcurve.dist, 0.0)
    else:
        coord_obj = body.ephem.get_position(occ.tca)
        obj_dist_au = float(coord_obj.distance.to(u.au).value)
        try:
            d_star_km = float(star.apparent_diameter(distance=obj_dist_au, band='V').to(u.km).value)
        except Exception:
            d_star_km = 0.0

    kerv = star.kervella()
    star_diam_mas = None
    if kerv and 'V' in kerv:
        star_diam_mas = _to_float(kerv['V'])
    elif kerv:
        star_diam_mas = _to_float(list(kerv.values())[0])

    # Se d_star_km não veio da lightcurve mas temos star_diam_mas e obj_dist_au, convertemos mas -> km
    if (d_star_km == 0.0 or np.isnan(d_star_km)) and star_diam_mas is not None and not np.isnan(star_diam_mas):
        # 1 mas na distância D (AU) equivale a D * 149597870.7 * (pi / (180 * 3600 * 1000)) km
        d_star_km = float(star_diam_mas * obj_dist_au * u.au.to(u.km) * u.mas.to(u.rad))

    if star_diam_mas is not None and not np.isnan(star_diam_mas):
        star_diam_str = f"{star_diam_mas:.4f} mas / {d_star_km:.2f} km (Kervella et al. 2004)"
    else:
        star_diam_str = "---"

    # 4. Parâmetros físicos do Corpo (Tabela 1)
    diam_val = _to_float(getattr(body.diameter, 'value', None))
    diam_unc = _to_float(getattr(body.diameter, 'uncertainty', None), 0.0)
    if not np.isnan(diam_val):
        body_diam_str = f"${diam_val:.2f} \\pm {diam_unc:.2f}$ km"
    else:
        body_diam_str = "---"

    H_val = _to_float(getattr(body.H, 'value', None))
    H_str = f"{H_val:.2f}" if not np.isnan(H_val) else "---"

    alb_val = _to_float(getattr(body.albedo, 'value', None))
    alb_unc = _to_float(getattr(body.albedo, 'uncertainty', None), 0.0)
    if not np.isnan(alb_val):
        body_alb_str = f"${alb_val:.3f} \\pm {alb_unc:.3f}$"
    else:
        body_alb_str = "---"

    rot_val = _to_float(getattr(body.rotation, 'value', None))
    body_rot_str = f"{rot_val:.3f} h" if not np.isnan(rot_val) else "---"

    spkid = getattr(body, 'spkid', '')
    ephem_str = f"Horizons (SPKID = {spkid})" if spkid else "Horizons"

    # 5. Circunstâncias da Ocultação (Tabela 2)
    tca_iso = f"{occ.tca.iso} UTC"
    vel_val = _to_float(occ.vel, 0.0)
    ca_val = _to_float(occ.ca, 0.0)
    pa_val = _to_float(occ.pa, 0.0)

    # 6. Velocidades radiais para cada corda
    rad_vels = _compute_radial_velocities(occ)

    # 7. Geração das Figuras
    if not os.path.exists(pred_map_file):
        try:
            occ.plot_occ_map(nameimg=os.path.splitext(pred_map_file)[0])
        except Exception as e:
            print(f"[Aviso] Não foi possível gerar {pred_map_file} automaticamente: {e}")

    # 7.1 Curvas Positivas
    lc_section_data = []
    for name, chord in pos_chords:
        lc = chord.lightcurve
        meta = observer_metadata.get(name, {})
        site_title = meta.get('site_name', name)
        obs_title = meta.get('observers', name)
        subsection_title = f"{site_title} - {obs_title}" if site_title != obs_title else site_title

        chi2_i = resolved_chi2[name]['immersion']
        chi2_e = resolved_chi2[name]['emersion']

        def _extract_lc_fit_row(chi2_obj, key):
            t_obj = getattr(lc, key, None)
            t_err = _to_float(getattr(lc, f"{key}_err", 0.0), 0.0)
            if t_obj is not None:
                t_hms = t_obj.iso.split()[1]
                t_str = f"{t_hms} ({t_err:.3f})"
            else:
                t_str = "---"

            if chi2_obj is not None and key in getattr(chi2_obj, 'data', {}):
                s1 = chi2_obj.get_nsigma(sigma=1)
                c_min = float(s1['chi2_min'])
                npts = int(chi2_obj.npts)
                nparam = int(chi2_obj.nparam)
                cpdf = c_min / (npts - nparam) if (npts - nparam) != 0 else 0.0
                if t_obj is not None:
                    t_str = f"{t_hms} ({s1[key][1]:.3f})"
            elif hasattr(lc, 'chisquare') and lc.chisquare is not None:
                s1 = lc.chisquare.get_nsigma(sigma=1)
                c_min = float(s1['chi2_min'])
                npts = int(lc.chisquare.npts)
                nparam = int(lc.chisquare.nparam)
                cpdf = c_min / (npts - nparam) if (npts - nparam) != 0 else 0.0
            else:
                c_min, npts, cpdf = 0.0, 0, 0.0

            disp = float(np.mean(lc.lc_sigma)) if hasattr(lc, 'lc_sigma') else 0.0
            lam0 = _to_float(getattr(lc, 'lambda_0', 0.700), 0.700)
            dlam = _to_float(getattr(lc, 'delta_lambda', 0.300), 0.300)
            dstar = _to_float(getattr(lc, 'd_star', d_star_km), d_star_km)
            if star_diam_mas is not None and not np.isnan(star_diam_mas):
                dstar_both_str = f"{dstar:.2f} km ({star_diam_mas:.4f} mas)"
            else:
                dstar_both_str = f"{dstar:.2f} km"

            vrad = rad_vels.get(name, {}).get(key)
            if vrad is None:
                vrad = abs(_to_float(getattr(lc, 'vel', vel_val), vel_val))

            return {
                'time_str': t_str,
                'chi2_min': f"{c_min:.3f}",
                'npts': f"{npts}",
                'chi2_pdf': f"{cpdf:.3f}",
                'dispersion': f"{disp:.3f}",
                'lambda_str': f"{lam0:.3f} ({dlam:.3f})",
                'dstar_str': dstar_both_str,
                'vrad_str': f"{vrad:.3f} km/s",
            }

        row_ing = _extract_lc_fit_row(chi2_i, 'immersion')
        row_egr = _extract_lc_fit_row(chi2_e, 'emersion')

        fig_lc_path = os.path.join(figures_dir, f"lc_fit_{name.replace(' ', '_')}.png")
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)

        tref_iso_date = lc.tref.iso.split()[0] if hasattr(lc, 'tref') else tca_iso.split()[0]

        for col_idx, (key, chi2_obj) in enumerate([('immersion', chi2_i), ('emersion', chi2_e)]):
            ax_top = axes[0, col_idx]
            ax_bot = axes[1, col_idx]

            plt.sca(ax_top)
            try:
                lc.plot_lc(ax=ax_top)
                lc.plot_model(ax=ax_top)
            except Exception:
                pass

            t_center = None
            if chi2_obj is not None and key in getattr(chi2_obj, 'data', {}):
                t_center = chi2_obj.get_nsigma(sigma=1)[key][0]
            elif getattr(lc, key, None) is not None and hasattr(lc, 'tref'):
                t_center = (getattr(lc, key).jd - lc.tref.jd) * 86400.0

            if t_center is not None:
                half_win = max(_to_float(getattr(lc, 'exptime', 0.5), 0.5) * 6.0, 2.0)
                ax_top.set_xlim(t_center - half_win, t_center + half_win)

            ax_top.set_xlabel(f"Seconds from {tref_iso_date} 00:00:00", fontsize=10)
            ax_top.set_ylabel("Normalized Flux", fontsize=10)
            ax_top.tick_params(labelsize=9)
            ax_top.legend(ncol=2, fontsize=9, loc='best')

            plt.sca(ax_bot)
            if chi2_obj is not None and key in getattr(chi2_obj, 'data', {}):
                chi2_obj.plot_chi2(key, ax=ax_bot)
                ax_bot.set_xlabel(key, fontsize=14)
                ax_bot.set_ylabel("Chi2", fontsize=14)
                ax_bot.tick_params(labelsize=11)
            else:
                ax_bot.text(0.5, 0.5, f"No {key} chi2 data", ha='center', va='center', transform=ax_bot.transAxes)

        plt.tight_layout()
        fig.savefig(fig_lc_path, bbox_inches='tight')
        plt.close(fig)

        lc_section_data.append({
            'name': name,
            'subsection_title': subsection_title,
            'ingress': row_ing,
            'egress': row_egr,
            'fig_path': fig_lc_path,
        })

    # 7.2 Curvas Negativas (gera gráfico da curva de luz normalizada + dados da observação)
    neg_section_data = []
    for name, chord in neg_chords:
        lc = chord.lightcurve
        meta = observer_metadata.get(name, {})
        site_title = meta.get('site_name', name)
        obs_title = meta.get('observers', name)
        subsection_title = f"{site_title} - {obs_title} (Negative)" if site_title != obs_title else f"{site_title} (Negative)"

        fig_neg_path = os.path.join(figures_dir, f"lc_neg_{name.replace(' ', '_')}.png")
        tref_iso_date = lc.tref.iso.split()[0] if hasattr(lc, 'tref') else tca_iso.split()[0]

        fig_n, ax_n = plt.subplots(figsize=(10, 4.5), dpi=300)
        plt.sca(ax_n)
        try:
            lc.plot_lc(ax=ax_n)
        except Exception:
            pass
        ax_n.set_xlabel(f"Seconds from {tref_iso_date} 00:00:00", fontsize=11)
        ax_n.set_ylabel("Normalized Flux", fontsize=11)
        ax_n.tick_params(labelsize=10)
        ax_n.legend(loc='best', fontsize=10)
        plt.tight_layout()
        fig_n.savefig(fig_neg_path, bbox_inches='tight')
        plt.close(fig_n)

        t_ini = f"{lc.initial_time.iso.split()[1]}" if hasattr(lc, 'initial_time') else "---"
        t_end = f"{lc.end_time.iso.split()[1]}" if hasattr(lc, 'end_time') else "---"
        exptime_s = f"{_to_float(getattr(lc, 'exptime', 0.0)):.4f} s"
        cycle_s = f"{_to_float(getattr(lc, 'cycle', 0.0)):.4f} s"
        npts_n = f"{len(lc.time)}" if hasattr(lc, 'time') else "---"
        disp_n = f"{float(np.std(lc.flux)):.3f}" if hasattr(lc, 'flux') and len(lc.flux) > 0 else "---"

        neg_section_data.append({
            'name': name,
            'subsection_title': subsection_title,
            't_ini': t_ini,
            't_end': t_end,
            'exptime': exptime_s,
            'cycle': cycle_s,
            'npts': npts_n,
            'dispersion': disp_n,
            'fig_path': fig_neg_path,
        })

    # 7.3 Figura da Elipse Ajustada + Cordas + Ponto Central com Barras de Erro (center_f, center_g)
    fig_ellipse_path = os.path.join(figures_dir, "fitted_ellipse.png")
    default_colors = ['blue', 'orange', 'salmon', 'cyan', 'green', 'purple', 'brown']
    chord_colors = chord_colors or {}

    fig_el, ax_el = plt.subplots(figsize=(7, 6), dpi=300)
    plt.sca(ax_el)
    for idx, (name, chord) in enumerate(all_chords):
        c_color = chord_colors.get(name, default_colors[idx % len(default_colors)])
        st = chord.status()
        if st == 'positive':
            chord.plot_chord(segment='positive', color=c_color, lw=2, linestyle='-', label=name, zorder=4)
            chord.plot_chord(segment='error', color='red', lw=3, linestyle='-', zorder=3)
        else:
            chord.plot_chord(segment='negative', color=c_color, lw=2, linestyle='--', label=f"{name} (negative)", zorder=2)

    if ellipse_chi2 is not None:
        draw_ellipse(**ellipse_chi2.get_values(), center_dot=True, color='black', lw=2, zorder=5)
        try:
            draw_ellipse(**ellipse_chi2.get_values(sigma=ellipse_sigma_plot), color='grey', alpha=0.5, zorder=1)
        except Exception:
            pass
    elif hasattr(occ, 'fitted_params') and occ.fitted_params:
        params_best = {k: v[0] for k, v in occ.fitted_params.items()}
        draw_ellipse(**params_best, center_dot=True, color='black', lw=2, zorder=5)

    # Plota explicitamente o ponto central (center_f, center_g) da elipse ajustada e suas barras de incerteza (1-sigma)
    if hasattr(occ, 'fitted_params') and occ.fitted_params:
        cf, dcf = occ.fitted_params['center_f'][0], occ.fitted_params['center_f'][1]
        cg, dcg = occ.fitted_params['center_g'][0], occ.fitted_params['center_g'][1]
        ax_el.errorbar(
            cf, cg, xerr=dcf, yerr=dcg,
            fmt='o', color='black', markersize=6, elinewidth=1.5, capsize=3,
            label=f"Center ({cf:.1f}, {cg:.1f}) km", zorder=6
        )
        req = occ.fitted_params['equatorial_radius'][0]
        margin = req * 1.55
        ax_el.set_xlim(cf - margin, cf + margin)
        ax_el.set_ylim(cg - margin, cg + margin)

    ax_el.set_xlabel("f (km)", fontsize=18)
    ax_el.set_ylabel("g (km)", fontsize=18)
    ax_el.tick_params(labelsize=13)
    ax_el.legend(ncol=2, loc='upper center', fontsize=10)
    plt.tight_layout()
    fig_el.savefig(fig_ellipse_path, bbox_inches='tight')
    plt.close(fig_el)

    # 7.4 Figura dos gráficos de Chi2 dos parâmetros da Elipse
    fig_el_chi2_path = os.path.join(figures_dir, "ellipse_chi2_params.png")
    has_el_chi2_plot = False
    if ellipse_chi2 is not None:
        has_el_chi2_plot = True
        fig_c = plt.figure(figsize=(11, 13), dpi=300)
        gs = fig_c.add_gridspec(3, 4)
        ax1 = fig_c.add_subplot(gs[0, 0:2])
        ax2 = fig_c.add_subplot(gs[0, 2:4])
        ax3 = fig_c.add_subplot(gs[1, 0:2])
        ax4 = fig_c.add_subplot(gs[1, 2:4])
        ax5 = fig_c.add_subplot(gs[2, 1:3])

        param_axes = [
            ('center_f', ax1),
            ('center_g', ax2),
            ('oblateness', ax3),
            ('position_angle', ax4),
            ('equatorial_radius', ax5),
        ]
        for p_name, p_ax in param_axes:
            if p_name in ellipse_chi2.data:
                plt.sca(p_ax)
                ellipse_chi2.plot_chi2(p_name, ax=p_ax)
                p_ax.set_xlabel(p_name, fontsize=13)
                p_ax.set_ylabel("Chi2", fontsize=13)
                p_ax.tick_params(labelsize=10)

        plt.tight_layout()
        fig_c.savefig(fig_el_chi2_path, bbox_inches='tight')
        plt.close(fig_c)

    # 8. Montagem da Tabela 3 (Observational circumstances — agora incluindo Cycle Time!)
    table3_rows = []
    for extra in extra_observers:
        table3_rows.append({
            'site': _latex_escape(extra.get('site_name', '---')),
            'lon': extra.get('lon', '---'),
            'lat': extra.get('lat', '---'),
            'alt': extra.get('alt', '---'),
            'observers': _latex_escape(extra.get('observers', '---')),
            'aperture': _latex_escape(extra.get('aperture', '---')),
            'detector': _latex_escape(extra.get('detector', '---')),
            'exptime': _latex_escape(extra.get('exptime', '---')),
            'cycletime': _latex_escape(extra.get('cycletime', '---')),
            'status': _latex_escape(extra.get('status', '---')),
            'filter': _latex_escape(extra.get('filter', 'Clear')),
        })

    for name, chord in all_chords:
        obs = chord.observer
        lc = chord.lightcurve
        meta = observer_metadata.get(name, {})

        lon_latex = _format_dms_latex(obs.site.lon, is_lon=True)
        lat_latex = _format_dms_latex(obs.site.lat, is_lon=False)
        alt_m = f"{obs.site.height.to(u.m).value:.2f}"

        exptime_val = getattr(lc, 'exptime', None)
        exptime_str = f"{_to_float(exptime_val):.3f}" if exptime_val is not None else "---"

        cycle_val = getattr(lc, 'cycle', None)
        cycle_str = f"{_to_float(cycle_val):.4f}" if cycle_val is not None and not np.isnan(_to_float(cycle_val)) else "---"

        st_cap = chord.status().capitalize()
        table3_rows.append({
            'site': _latex_escape(meta.get('site_name', obs.name)),
            'lon': lon_latex,
            'lat': lat_latex,
            'alt': alt_m,
            'observers': _latex_escape(meta.get('observers', obs.name)),
            'aperture': _latex_escape(meta.get('aperture', '---')),
            'detector': _latex_escape(meta.get('detector', '---')),
            'exptime': meta.get('exptime', exptime_str),
            'cycletime': meta.get('cycletime', cycle_str),
            'status': _latex_escape(meta.get('status', st_cap)),
            'filter': _latex_escape(meta.get('filter', 'Clear')),
        })

    # 9. Montagem da Tabela de Resumo das Cordas (Positivas + Negativas)
    table8_rows = []
    for name, chord in pos_chords:
        lc = chord.lightcurve
        meta = observer_metadata.get(name, {})
        site_label = _latex_escape(meta.get('site_name', name))

        chi2_i = resolved_chi2[name]['immersion']
        chi2_e = resolved_chi2[name]['emersion']

        imm_err = chi2_i.get_nsigma(sigma=1)['immersion'][1] if (chi2_i is not None and 'immersion' in getattr(chi2_i, 'data', {})) else _to_float(getattr(lc, 'immersion_err', 0.0), 0.0)
        eme_err = chi2_e.get_nsigma(sigma=1)['emersion'][1] if (chi2_e is not None and 'emersion' in getattr(chi2_e, 'data', {})) else _to_float(getattr(lc, 'emersion_err', 0.0), 0.0)

        imm_str = f"{lc.immersion.iso} ({imm_err:.3f})" if getattr(lc, 'immersion', None) is not None else "---"
        eme_str = f"{lc.emersion.iso} ({eme_err:.3f})" if getattr(lc, 'emersion', None) is not None else "---"
        table8_rows.append((site_label, imm_str, eme_str, "Positive"))

    for name, chord in neg_chords:
        lc = chord.lightcurve
        meta = observer_metadata.get(name, {})
        site_label = _latex_escape(meta.get('site_name', name))
        t_ini = f"{lc.initial_time.iso} (start)" if hasattr(lc, 'initial_time') else "Not occulted"
        t_end = f"{lc.end_time.iso} (end)" if hasattr(lc, 'end_time') else "Not occulted"
        table8_rows.append((site_label, t_ini, t_end, "Negative"))

    # 10. Montagem da Tabela 9 (Best ellipse from fit with 1-sigma error bar + raio equivalente + albedo)
    fp = getattr(occ, 'fitted_params', {})
    def _fmt_fp(key):
        if key in fp:
            return f"{fp[key][0]:.3f} ({fp[key][1]:.3f})"
        return "---"

    table9_rows = [
        ("Center\\_f [km]", _fmt_fp('center_f')),
        ("Center\\_g [km]", _fmt_fp('center_g')),
        ("R$_{equat}$ [km]", _fmt_fp('equatorial_radius')),
        ("Oblateness", _fmt_fp('oblateness')),
        ("Position Angle [deg]", _fmt_fp('position_angle')),
    ]
    if 'equatorial_radius' in fp and 'oblateness' in fp:
        r_pol = fp['equatorial_radius'][0] * (1.0 - fp['oblateness'][0])
        r_eqv = np.sqrt(fp['equatorial_radius'][0] * r_pol)
        table9_rows.append(("R$_{polar}$ [km]", f"{r_pol:.3f}"))
        table9_rows.append(("R$_{equiv}$ [km]", f"{r_eqv:.3f}"))

    # 11. Extração da Posição Astrométrica Refinada (new_astrometric_position)
    astrom_data = _extract_astrometric_results(occ, occ_full_text)

    # 12. Formatação dos links para Lucky Star e Occultation Portal
    if lucky_star_url:
        ls_latex = f"ERC \\href{{{lucky_star_url}}}{{*Lucky Star*}} project [3] (\\url{{{lucky_star_url}}})"
    else:
        ls_latex = "ERC *Lucky Star* project [3]"

    if occultation_portal_url:
        op_latex = f"\\href{{{occultation_portal_url}}}{{*Occultation Portal*}} website [4] (\\url{{{occultation_portal_url}}})"
    else:
        op_latex = "*Occultation Portal* website [4]"

    # 13. Construção do código LaTeX (.tex)
    prelim_str = r"\\ \textbf{PRELIMINARY}" if preliminary else ""
    tex_lines = []
    tex_lines.append(r"% =============================================================================")
    tex_lines.append(r"% Relatório de Ocultação Estelar gerado automaticamente via SORA")
    tex_lines.append(r"% =============================================================================")
    tex_lines.append(r"\documentclass[11pt,a4paper]{article}")
    tex_lines.append(r"\usepackage[utf8]{inputenc}")
    tex_lines.append(r"\usepackage[T1]{fontenc}")
    tex_lines.append(r"\usepackage[margin=2.2cm]{geometry}")
    tex_lines.append(r"\usepackage{graphicx}")
    tex_lines.append(r"\usepackage{amsmath,amssymb}")
    tex_lines.append(r"\usepackage{array}")
    tex_lines.append(r"\usepackage{listings}")
    tex_lines.append(r"\usepackage{float}")
    tex_lines.append(r"\usepackage{caption}")
    tex_lines.append(r"\usepackage[colorlinks=true,linkcolor=blue,urlcolor=blue,citecolor=blue]{hyperref}")
    tex_lines.append(r"")
    tex_lines.append(r"% Configuração do ambiente listings para o output textual do SORA (Listing 1)")
    tex_lines.append(r"\lstset{")
    tex_lines.append(r"    basicstyle=\ttfamily\scriptsize,")
    tex_lines.append(r"    frame=single,")
    tex_lines.append(r"    breaklines=true,")
    tex_lines.append(r"    columns=fullflexible,")
    tex_lines.append(r"    keepspaces=true,")
    tex_lines.append(r"    showstringspaces=false")
    tex_lines.append(r"}")
    tex_lines.append(r"")
    tex_lines.append(f"\\title{{\\textbf{{Stellar occultation by {_latex_escape(title_body_name)} on {month_name} {day_ord}, {year_str}}}{prelim_str}}}")
    tex_lines.append(f"\\author{{{_latex_escape(author)}}}")
    tex_lines.append(f"\\date{{{_latex_escape(report_date)}}}")
    tex_lines.append(r"")
    tex_lines.append(r"\begin{document}")
    tex_lines.append(r"")
    tex_lines.append(r"\maketitle")
    tex_lines.append(r"")
    tex_lines.append(r"% =============================================================================")
    tex_lines.append(r"% SEÇÃO 1: PREDIÇÃO E CAMPANHA OBSERVACIONAL")
    tex_lines.append(r"% =============================================================================")
    tex_lines.append(r"\section{Prediction and observational campaign}")
    tex_lines.append(r"")
    tex_lines.append(r"\begin{figure}[H]")
    tex_lines.append(r"    \centering")
    tex_lines.append(f"    \\includegraphics[width=0.75\\textwidth]{{{pred_map_file}}}")
    tex_lines.append(r"    \caption{Prediction of the occultation}")
    tex_lines.append(r"    \label{fig:prediction}")
    tex_lines.append(r"\end{figure}")
    tex_lines.append(r"")
    tex_lines.append(
        f"The stellar occultation by the {_latex_escape(body_class)} asteroid {_latex_escape(body_prose_name)} "
        f"presented in this report was predicted using the star positions from the Gaia Data Release 3 (Gaia DR3) "
        f"catalogue [1] together with the Numerical Integration of the Motion of an Asteroid (NIMA) ephemeris [2]. "
        f"The prediction was made available through the {ls_latex}, and the observational campaign "
        f"was coordinated using the {op_latex}."
    )
    tex_lines.append(r"")
    if n_neg > 0:
        neg_word = num_words.get(n_neg, str(n_neg))
        neg_noun = "chord" if n_neg == 1 else "chords"
        neg_clause = f" and {neg_word} negative {neg_noun}"
    else:
        neg_clause = ""
    pos_noun = "chord" if n_pos == 1 else "chords"
    tex_lines.append(
        f"The occultation occurred on {event_date_text} and was successfully recorded from {n_tot_word} observing sites, "
        f"providing {n_pos_word} positive {pos_noun}{neg_clause} across the asteroid's silhouette. These observations allowed the "
        f"determination of the object's apparent limb and the adjustment of an elliptical shape model, as well as an "
        f"astrometric refinement of the predicted ephemeris."
    )
    tex_lines.append(r"")
    tex_lines.append(
        f"Table 1 summarizes the physical properties of {_latex_escape(body_prose_name)} and the astrometric and "
        f"photometric properties of the occulted Gaia DR3 star. Table 3 presents the observational circumstances "
        f"for each participating observatory, including their geographical coordinates, observing conditions, and "
        f"acquisition parameters."
    )
    tex_lines.append(r"")
    tex_lines.append(r"% --- Tabela 1: Dados do Objeto e da Estrela ---")
    tex_lines.append(r"\begin{table}[H]")
    tex_lines.append(r"    \centering")
    tex_lines.append(r"    \caption{Object and star data.}")
    tex_lines.append(r"    \vspace{0.2cm}")
    tex_lines.append(r"    \begin{tabular}{cc}")
    tex_lines.append(r"        \hline\hline")
    tex_lines.append(f"        Source & {star_code} \\\\")
    tex_lines.append(f"        Catalogue & {_latex_escape(star_cat)} \\\\")
    tex_lines.append(f"        Astrometric position (ICRS) & {astrometric_pos_icrs} \\\\")
    tex_lines.append(f"        Star position error & {star_pos_error_str} \\\\")
    tex_lines.append(f"        Magnitudes & {mag_line1} \\\\")
    if mag_line2:
        tex_lines.append(f"         & {mag_line2} \\\\")
    tex_lines.append(f"        Star diameter & {star_diam_str} \\\\")
    tex_lines.append(r"        \hline\hline")
    tex_lines.append(f"        Object & {_latex_escape(body_paren_name)} \\\\")
    tex_lines.append(f"        Object class & {_latex_escape(body_class)} \\\\")
    tex_lines.append(f"        Diameter & {body_diam_str} \\\\")
    tex_lines.append(f"        Absolute magnitude ($H$) & {H_str} \\\\")
    tex_lines.append(f"        Albedo & {body_alb_str} \\\\")
    tex_lines.append(f"        Rotation period & {body_rot_str} \\\\")
    tex_lines.append(f"        Ephemeris & {_latex_escape(ephem_str)} \\\\")
    tex_lines.append(r"        \hline")
    tex_lines.append(r"    \end{tabular}")
    tex_lines.append(r"    \label{tab:object_star}")
    tex_lines.append(r"\end{table}")
    tex_lines.append(r"")
    tex_lines.append(r"% --- Tabela 2: Circunstâncias da Ocultação ---")
    tex_lines.append(r"\begin{table}[H]")
    tex_lines.append(r"    \centering")
    tex_lines.append(r"    \caption{Occultation circumstances.}")
    tex_lines.append(r"    \vspace{0.2cm}")
    tex_lines.append(r"    \begin{tabular}{cc}")
    tex_lines.append(r"        \hline\hline")
    tex_lines.append(f"        Epoch & {tca_iso} \\\\")
    tex_lines.append(f"        Shadow velocity & ${vel_val:.2f}\\,\\text{{km}}\\,\\text{{s}}^{{-1}}$ \\\\")
    tex_lines.append(f"        Object distance & {obj_dist_au:.2f} au \\\\")
    tex_lines.append(f"        Closest approach & {ca_val:.3f} arcsec \\\\")
    tex_lines.append(f"        Position angle & ${pa_val:.2f}^\\circ$ \\\\")
    tex_lines.append(f"        Positive chords & {n_pos} \\\\")
    tex_lines.append(f"        Negative chords & {n_neg} \\\\")
    if lucky_star_url:
        tex_lines.append(f"        Lucky Star URL & \\url{{{lucky_star_url}}} \\\\")
    if occultation_portal_url:
        tex_lines.append(f"        Occultation Portal URL & \\url{{{occultation_portal_url}}} \\\\")
    tex_lines.append(r"        \hline")
    tex_lines.append(r"    \end{tabular}")
    tex_lines.append(r"    \label{tab:circumstances}")
    tex_lines.append(r"\end{table}")
    tex_lines.append(r"")
    tex_lines.append(r"\clearpage")
    tex_lines.append(r"")
    tex_lines.append(r"% --- Tabela 3: Circunstâncias Observacionais (incluindo Exposure Time e Cycle Time) ---")
    tex_lines.append(r"\begin{table}[H]")
    tex_lines.append(r"    \centering")
    tex_lines.append(r"    \caption{Observational circumstances.}")
    tex_lines.append(r"    \vspace{0.2cm}")
    tex_lines.append(r"    \footnotesize")
    tex_lines.append(r"    \begin{tabular}{cccccc}")
    tex_lines.append(r"        \hline\hline")
    tex_lines.append(r"         & \textbf{Longitude [$^\circ\,'\,'']$} & & \textbf{Aperture (mm)} & \textbf{Exp. Time (s)} & \textbf{Status} \\")
    tex_lines.append(r"        \textbf{Site} & \textbf{Latitude [$^\circ\,'\,'']$} & \textbf{Observers} & \textbf{Detector} & \textbf{Cycle Time (s)} & \textbf{Filter} \\")
    tex_lines.append(r"         & \textbf{Altitude (m)} & & & & \\")
    tex_lines.append(r"        \hline")
    for idx, r in enumerate(table3_rows):
        tex_lines.append(f"        {r['site']} & {r['lon']} & {r['observers']} & {r['detector']} & {r['exptime']} & {r['status']} \\\\")
        tex_lines.append(f"         & {r['lat']} & & {r['aperture']} & {r['cycletime']} & {r['filter']} \\\\")
        tex_lines.append(f"         & {r['alt']} & & & & \\\\")
        tex_lines.append(r"        \hline")
    tex_lines.append(r"    \end{tabular}")
    tex_lines.append(r"    \label{tab:obs_circumstances}")
    tex_lines.append(r"\end{table}")
    tex_lines.append(r"")
    tex_lines.append(r"\clearpage")
    tex_lines.append(r"")
    tex_lines.append(r"% =============================================================================")
    tex_lines.append(r"% SEÇÃO 2: DETERMINAÇÃO DOS INSTANTES DO CORPO PRINCIPAL")
    tex_lines.append(r"% =============================================================================")
    tex_lines.append(r"\section{Main body instants determination}")
    tex_lines.append(r"")
    star_diam_text = f"{star_diam_mas:.4f} mas ({d_star_km:.2f} km)" if star_diam_mas is not None else f"{d_star_km:.2f} km"
    extra_star_sentence = f" {star_comment}" if star_comment else ""
    tex_lines.append(
        f"The light curves were modeled using the expected limb velocity, calculated from a preliminary ellipse fit. "
        f"The limb-darkened stellar angular diameter was estimated from the surface-brightness relations of "
        f"Kervella et al. (2004) as {star_diam_text} at the distance of {_latex_escape(body_prose_name)} "
        f"({obj_dist_au:.2f} au).{extra_star_sentence}"
    )
    if campaign_comment:
        tex_lines.append(r"")
        tex_lines.append(campaign_comment)
    else:
        tex_lines.append(r"")
        obs_noun = "observation was" if n_pos == 1 else "observations were"
        neg_sentence = (
            f" Additionally, {n_neg} negative observation(s) provided constraints on the asteroid's limb projection."
            if n_neg > 0 else ""
        )
        tex_lines.append(
            f"In total, {n_pos} positive {obs_noun} used for the determination of the instants of immersion "
            f"and emersion of the main body.{neg_sentence}"
        )
    tex_lines.append(r"")

    # 2.X Subseções das Cordas Positivas
    for idx, lc_info in enumerate(lc_section_data):
        tex_lines.append(r"\clearpage")
        tex_lines.append(f"\\subsection{{{_latex_escape(lc_info['subsection_title'])}}}")
        tex_lines.append(r"")
        tex_lines.append(r"\begin{table}[H]")
        tex_lines.append(r"    \centering")
        tex_lines.append(f"    \\caption{{Results for instants determination for the {_latex_escape(lc_info['name'])} light curve.}}")
        tex_lines.append(r"    \vspace{0.15cm}")
        tex_lines.append(r"    \footnotesize")
        tex_lines.append(r"    \begin{tabular}{ccccccccc}")
        tex_lines.append(r"        \hline\hline")
        tex_lines.append(r"        Instant & Time (UTC) ($1\sigma$) & $\chi^2_{min}$ & Npts & $\chi^2_{pdf}$ & Dispersion & $\lambda$ ($\mu$m) & $d_{\text{star}}$ & $v_{rad}$ \\")
        tex_lines.append(r"        \hline")
        ing = lc_info['ingress']
        egr = lc_info['egress']
        tex_lines.append(
            f"        Ingress & {ing['time_str']} & {ing['chi2_min']} & {ing['npts']} & {ing['chi2_pdf']} & "
            f"{ing['dispersion']} & {ing['lambda_str']} & {ing['dstar_str']} & {ing['vrad_str']} \\\\"
        )
        tex_lines.append(
            f"        Egress & {egr['time_str']} & {egr['chi2_min']} & {egr['npts']} & {egr['chi2_pdf']} & "
            f"{egr['dispersion']} & {egr['lambda_str']} & {egr['dstar_str']} & {egr['vrad_str']} \\\\"
        )
        tex_lines.append(r"        \hline")
        tex_lines.append(r"    \end{tabular}")
        tex_lines.append(r"\end{table}")
        tex_lines.append(r"")
        tex_lines.append(r"\begin{figure}[H]")
        tex_lines.append(r"    \centering")
        tex_lines.append(f"    \\includegraphics[width=0.98\\textwidth]{{{lc_info['fig_path']}}}")
        tex_lines.append(r"    \caption{$\chi^2$ graphs for ingress and egress fits and modeled light curves.}")
        tex_lines.append(r"\end{figure}")
        tex_lines.append(r"")

    # 2.Y Subseções das Cordas Negativas (se existirem)
    for idx, neg_info in enumerate(neg_section_data):
        tex_lines.append(r"\clearpage")
        tex_lines.append(f"\\subsection{{{_latex_escape(neg_info['subsection_title'])}}}")
        tex_lines.append(r"")
        tex_lines.append(
            f"No secondary or main body occultation drop was detected in the light curve recorded at "
            f"{_latex_escape(neg_info['name'])}. The normalized light curve is presented below and its "
            f"sky-plane trajectory is used as a negative chord constraint."
        )
        tex_lines.append(r"")
        tex_lines.append(r"\begin{table}[H]")
        tex_lines.append(r"    \centering")
        tex_lines.append(f"    \\caption{{Summary of the negative light curve for {_latex_escape(neg_info['name'])}.}}")
        tex_lines.append(r"    \vspace{0.15cm}")
        tex_lines.append(r"    \small")
        tex_lines.append(r"    \begin{tabular}{cccccc}")
        tex_lines.append(r"        \hline\hline")
        tex_lines.append(r"        Initial Time (UTC) & End Time (UTC) & Exp. Time & Cycle Time & Npts & Dispersion ($\sigma$) \\")
        tex_lines.append(r"        \hline")
        tex_lines.append(
            f"        {neg_info['t_ini']} & {neg_info['t_end']} & {neg_info['exptime']} & "
            f"{neg_info['cycle']} & {neg_info['npts']} & {neg_info['dispersion']} \\\\"
        )
        tex_lines.append(r"        \hline")
        tex_lines.append(r"    \end{tabular}")
        tex_lines.append(r"\end{table}")
        tex_lines.append(r"")
        tex_lines.append(r"\begin{figure}[H]")
        tex_lines.append(r"    \centering")
        tex_lines.append(f"    \\includegraphics[width=0.85\\textwidth]{{{neg_info['fig_path']}}}")
        tex_lines.append(f"    \\caption{{Normalized light curve recorded at {_latex_escape(neg_info['name'])} (negative chord).}}")
        tex_lines.append(r"\end{figure}")
        tex_lines.append(r"")

    tex_lines.append(r"\clearpage")
    tex_lines.append(r"")
    tex_lines.append(r"% =============================================================================")
    tex_lines.append(r"% SEÇÃO 3: AJUSTE DE FORMA (ELIPSE) E POSIÇÃO ASTROMÉTRICA")
    tex_lines.append(r"% =============================================================================")
    tex_lines.append(r"\section{Fit shape and astrometric position}")
    tex_lines.append(r"")
    tex_lines.append(
        r"Using the instants of ingress and egress presented in the previous section, along with the "
        r"constraints provided by any negative chords, the chords are projected in the sky plane, "
        r"allowing the ellipse fit and the determination of the refined astrometric position of the body."
    )
    tex_lines.append(r"")
    tex_lines.append(r"% --- Tabela de Resumo dos Instantes e Cordas (Positivas e Negativas) ---")
    tex_lines.append(r"\begin{table}[H]")
    tex_lines.append(r"    \centering")
    tex_lines.append(r"    \caption{Summary of the chords and instants obtained for the light curves.}")
    tex_lines.append(r"    \vspace{0.2cm}")
    tex_lines.append(r"    \begin{tabular}{lccc}")
    tex_lines.append(r"        \hline\hline")
    tex_lines.append(r"        Site & Immersion / Start (UTC) & Emersion / End (UTC) & Status \\")
    tex_lines.append(r"        \hline")
    for site_label, imm_str, eme_str, st_label in table8_rows:
        tex_lines.append(f"        {site_label} & {imm_str} & {eme_str} & {st_label} \\\\")
    tex_lines.append(r"        \hline")
    tex_lines.append(r"    \end{tabular}")
    tex_lines.append(r"    \label{tab:summary_instants}")
    tex_lines.append(r"\end{table}")
    tex_lines.append(r"")
    tex_lines.append(r"\vspace{0.3cm}")
    tex_lines.append(r"")
    tex_lines.append(r"% --- Tabela dos Parâmetros da Elipse Ajustada ---")
    tex_lines.append(r"\begin{table}[H]")
    tex_lines.append(r"    \centering")
    tex_lines.append(r"    \caption{Best ellipse from fit with $1\sigma$ error bar.}")
    tex_lines.append(r"    \vspace{0.2cm}")
    tex_lines.append(r"    \begin{tabular}{lc}")
    tex_lines.append(r"        \hline\hline")
    tex_lines.append(r"        Parameter & Value ($1\sigma$) \\")
    tex_lines.append(r"        \hline")
    for param_name, param_val in table9_rows:
        tex_lines.append(f"        {param_name} & {param_val} \\\\")
    tex_lines.append(r"        \hline")
    tex_lines.append(r"    \end{tabular}")
    tex_lines.append(r"    \label{tab:best_ellipse}")
    tex_lines.append(r"\end{table}")
    tex_lines.append(r"")

    # Tabela de Posição Astrométrica Refinada (extraída da análise do SORA)
    if astrom_data:
        tex_lines.append(r"\vspace{0.3cm}")
        tex_lines.append(r"")
        tex_lines.append(r"% --- Tabela da Posição Astrométrica Refinada pelo SORA ---")
        tex_lines.append(r"\begin{table}[H]")
        tex_lines.append(r"    \centering")
        tex_lines.append(f"    \\caption{{Refined astrometric position of {_latex_escape(body_paren_name)} and ephemeris offset obtained from SORA.}}")
        tex_lines.append(r"    \vspace{0.2cm}")
        tex_lines.append(r"    \begin{tabular}{lc}")
        tex_lines.append(r"        \hline\hline")
        tex_lines.append(r"        Parameter & Value ($1\sigma$) \\")
        tex_lines.append(r"        \hline")
        if 'epoch' in astrom_data:
            tex_lines.append(f"        Reference Epoch & {astrom_data['epoch']} \\\\")
        if 'reference' in astrom_data:
            tex_lines.append(f"        Reference Center & {_latex_escape(astrom_data['reference'])} \\\\")
        if 'ra' in astrom_data:
            tex_lines.append(f"        Astrometric RA (ICRS) & {astrom_data['ra']} ({astrom_data['ra_err']}) \\\\")
        if 'dec' in astrom_data:
            tex_lines.append(f"        Astrometric DEC (ICRS) & {astrom_data['dec']} ({astrom_data['dec_err']}) \\\\")
        if 'da_cos_dec' in astrom_data:
            tex_lines.append(f"        Offset $\\Delta\\alpha\\cos\\delta$ & {astrom_data['da_cos_dec']} \\\\")
        if 'd_dec' in astrom_data:
            tex_lines.append(f"        Offset $\\Delta\\delta$ & {astrom_data['d_dec']} \\\\")
        if 'x_km' in astrom_data:
            tex_lines.append(f"        Offset $X$ & {astrom_data['x_km']} \\\\")
        if 'y_km' in astrom_data:
            tex_lines.append(f"        Offset $Y$ & {astrom_data['y_km']} \\\\")
        tex_lines.append(r"        \hline")
        tex_lines.append(r"    \end{tabular}")
        tex_lines.append(r"    \label{tab:astrometric_position}")
        tex_lines.append(r"\end{table}")
        tex_lines.append(r"")

    tex_lines.append(r"\clearpage")
    tex_lines.append(r"")
    tex_lines.append(r"\begin{figure}[H]")
    tex_lines.append(r"    \centering")
    tex_lines.append(f"    \\includegraphics[width=0.80\\textwidth]{{{fig_ellipse_path}}}")
    tex_lines.append(r"    \caption{Best fitted ellipse plotted in black with its fitted center $(f_0, g_0)$ and positive/negative chords.}")
    tex_lines.append(r"    \label{fig:best_ellipse}")
    tex_lines.append(r"\end{figure}")
    tex_lines.append(r"")

    if has_el_chi2_plot:
        tex_lines.append(r"\clearpage")
        tex_lines.append(r"")
        tex_lines.append(r"\begin{figure}[H]")
        tex_lines.append(r"    \centering")
        tex_lines.append(f"    \\includegraphics[width=0.85\\textwidth]{{{fig_el_chi2_path}}}")
        tex_lines.append(r"    \caption{$\chi^2$ plots of the fitted ellipse parameters.}")
        tex_lines.append(r"    \label{fig:ellipse_chi2}")
        tex_lines.append(r"\end{figure}")
        tex_lines.append(r"")

    tex_lines.append(r"\clearpage")
    tex_lines.append(r"")
    tex_lines.append(r"% =============================================================================")
    tex_lines.append(r"% LISTING 1: LOG COMPLETO DO SORA (print(occ))")
    tex_lines.append(r"% =============================================================================")
    tex_lines.append(r"\begin{lstlisting}[caption={SORA occultation analysis output.}, label={lst:sora_output}]")
    tex_lines.append(occ_full_text.strip())
    tex_lines.append(r"\end{lstlisting}")
    tex_lines.append(r"")
    tex_lines.append(r"\end{document}")

    tex_content = "\n".join(tex_lines) + "\n"
    with open(output_tex, "w", encoding="utf-8") as f:
        f.write(tex_content)

    print(f"[Sucesso] Relatório LaTeX salvo em: {output_tex}")
    print(f"[Sucesso] Figuras salvas no diretório: {figures_dir}/")
    return output_tex
