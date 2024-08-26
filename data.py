import pandas as pd
import os


def format_update_date(df):
    return df[0][0].strftime("%d/%m/%Y")


def rmv_col_suffixe(columns):
    return [col.split('.')[0] if '.' in col else col for col in columns]


def split_by_unnamed_columns(df):
    data = {}
    start_idx = 0

    for i, col in enumerate(df.columns):
        if 'Unnamed:' in col:
            sub_df = df.iloc[:, start_idx:i].copy()
            sub_df.columns = rmv_col_suffixe(sub_df.columns)
            key = sub_df.columns[0]
            sub_df.columns = ['Date'] + sub_df.columns[1:].tolist()
            data[key] = sub_df
            start_idx = i + 1

    if start_idx < len(df.columns):
        sub_df = df.iloc[:, start_idx:].copy()
        sub_df.columns = rmv_col_suffixe(sub_df.columns)
        key = sub_df.columns[0]
        sub_df.columns = ['Date'] + sub_df.columns[1:].tolist()
        data[key] = sub_df

    return data


def split_by_parts_columns(df):
    result = {}
    start_idx = 0

    for i, col in enumerate(df.columns):
        if df[col].isnull().all():
            sub_df = df.iloc[:, start_idx:i].copy()
            sub_df.columns = rmv_col_suffixe(sub_df.columns)
            result[col] = sub_df.dropna(how="all", axis=0)
            start_idx = i + 1

    if start_idx < len(df.columns):
        sub_df = df.iloc[:, start_idx:].copy()
        sub_df.columns = rmv_col_suffixe(sub_df.columns)
        key = df.columns[start_idx - 1]
        result[key] = sub_df.dropna(how="all", axis=0)

    return result


def format_time_series_dict(df):
    return {key: sub_df.dropna() for key, sub_df in split_by_unnamed_columns(df).items()}


def arrow_percentage(x):
    try:
        return f"{abs(x) * 100:.2f}%"
    except Exception:
        return x


def format_arrow_value(x):
    if x > 0:
        return (
            f'<div style="display:flex; align-items:center; justify-content:center; height:100%; width:100%; box-sizing:border-box;">'
            f'<div style="display:flex; align-items:center; justify-content:center; color:rgb(19, 115, 51);">'
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="14" height="14" style="fill:rgb(19, 115, 51);">'
            f'<path d="M4 12l1.41 1.41L11 7.83V20h2V7.83l5.58 5.59L20 12l-8-8-8 8z"></path></svg>'
            f'<span style="margin-left:4px;">{arrow_percentage(x)}</span>'
            f'</div>'
            f'</div>'
        )
    elif x < 0:
        return (
            f'<div style="display:flex; align-items:center; justify-content:center; height:100%; width:100%; box-sizing:border-box;">'
            f'<div style="display:flex; align-items:center; justify-content:center; color:rgb(165, 14, 14);">'
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="14" height="14" style="fill:rgb(165, 14, 14);">'
            f'<path d="M4 12l1.41 1.41L11 7.83V20h2V7.83l5.58 5.59L20 12l-8-8-8 8z" transform="rotate(180,12,12)"></path></svg>'
            f'<span style="margin-left:4px;">{arrow_percentage(x)}</span>'
            f'</div>'
            f'</div>'
        )
    else:
        return (
            f'<div style="display:flex; align-items:center; justify-content:center; height:100%; width:100%; '
            f'box-sizing:border-box; color:rgb(60, 64, 67);">'
            f'{arrow_percentage(x)}'
            f'</div>'
        )


def format_index_global_data(df):
    df['Fixing'] = df['Fixing'].apply(
        lambda x: f"{x * 100:.2f}%" if x < 1 else f"{x:,.2f}".replace(',', ' ').replace('.', ',')
    )
    df['Performance YTD_tmp'] = df['Performance YTD']
    df['Performance YTD'] = df['Performance YTD'].apply(format_arrow_value)
    return df


def format_other_metrics_data(df):
    df = df.T.copy()

    percent_col = ['Performance depuis création',
                   'Performance annualisée',
                   'Volatilité annualisée',
                   'Maximum DrawDown']

    level_col = ['Plus haute VL',
                 'Plus basse VL',
                 'Ratio de Sharpe']

    df[percent_col] = df[percent_col].map(lambda x: f"{x * 100:.2f}%")
    df[level_col] = df[level_col].map(lambda x: round(x, 2))

    return df.T


def format_calendar_value(x):
    if x == 0:
        return "-"
    elif x < 1:
        return f"{x * 100:.2f}%"
    elif x == 999:
        return ""
    else:
        return x


def format_calendar_dict(df):
    return {key: sub_df.map(format_calendar_value) for key, sub_df in split_by_parts_columns(df).items()}


def compute_position_metric(df):
    mask = (df['Poche'] != "Couverture")

    metric_dict = {
        'Nominal': df.loc[mask, "Nominal"].sum(),
        "TRI": (df.loc[mask, "Nominal"] @ df.loc[mask, "TRI"]) / df.loc[mask, "Nominal"].sum(),
        "MtM": (df["Nominal"] @ df["MtM"]) / df["Nominal"].sum(),
        "Upside": (df["Nominal"] @ df["Upside"]) / df["Nominal"].sum(),
        "Duration": (df["Nominal"] @ df["Duration"]) / df["Nominal"].sum(),
        "Delta": (df["Nominal"] @ df["Delta"]) / df["Nominal"].sum(),
        "Vega": (df["Nominal"] @ df["Vega"]) / df["Nominal"].sum(),
        "VaR": (df["Nominal"] @ df["Vega"]) / df["Nominal"].sum()
    }

    metric_dict.update({col: "" for col in df.columns if col not in metric_dict.keys()})

    metric_df = pd.DataFrame([metric_dict])
    return pd.concat([df, metric_df]).reset_index(drop=True)


def format_date_short(dt, default_value=" - "):
    month_to_str = {
        1: "janv", 2: "févr", 3: "mars", 4: "avr", 5: "mai", 6: "juin",
        7: "juil", 8: "août", 9: "sept", 10: "oct", 11: "nov", 12: "déc"
    }
    try:
        if not pd.isnull(dt):
            month = month_to_str[dt.month]
            return f"{dt.day}-{month}-{str(dt.year)[2:]}"
        else:
            return default_value

    except Exception:
        return " "


def format_date_long(dt):
    month_to_str_long = {
        1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril", 5: "Mai", 6: "Juin",
        7: "Juillet", 8: "Août", 9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"
    }
    try:
        if not pd.isnull(dt):
            month = month_to_str_long[dt.month]
            return f"{dt.day} {month} {dt.year}"
        else:
            return " - "

    except Exception:
        return " "


def format_percentage(x):
    try:
        if not pd.isnull(x):
            return f"{x * 100:.2f}%"
        else:
            return " - "
    except Exception:
        return x


def format_percentage_product(x):
    try:
        if not pd.isnull(x):
            value = x * 100
            if value == int(value):
                return f"{value:.0f}%"
            else:
                return f"{value:.2f}%"
        else:
            return " - "
    except Exception:
        return x


def format_position_data(df):
    df = compute_position_metric(df)

    numeric_format_cols = ['Nominal']
    for col in numeric_format_cols:
        if col in df.columns:
            df[col + '_tmp'] = df[col]
            df[col] = df[col].apply(lambda x: f"{x:,.0f}".replace(',', ' '))

    percent_cols = ['TRI', 'Coupon(s) restant(s)', 'ERP', 'Delta', 'Vega', 'VaR']
    for col in percent_cols:
        if col in df.columns:
            df[col + '_tmp'] = df[col]
            df[col] = df[col].apply(format_percentage)

    arrow_cols = ['MtM', 'Upside']
    for col in arrow_cols:
        if col in df.columns:
            df[col + '_tmp'] = df[col]
            df[col] = df[col].apply(format_arrow_value)

    df["Duration"] = df["Duration"].round(2)
    df["Date de trade_tmp"] = df["Date de trade"]
    df['Date de trade'] = df['Date de trade'].apply(format_date_short)

    return df


def format_notes_data():
    return {
        'TRI': "TRI calculé en prenant en compte la totalité des coupons perçus/à percevoir et la date de "
               "maturité/autocall, sur les niveaux actuels de marché.  TRI global calculé hors produits de couverture.",
        "ERP": "ERP = Probabilité de rappel",
        'Delta': "Delta calculé comme l'impact produit d'un choc instantané de +1% sur le sous-jacent",
        'VaR': "VaR historique 20 jours, seuil 99%, calculée sur 5 ans",
        "Vega": "Vega calculé comme l'impact produit d'un choc instantané de +1% sur la volatilité du sous-jacent"
    }


def format_worst_best_data(df):
    df["Performance MTD_tmp"] = df["Performance MTD"].copy()
    df["Performance MTD"] = df["Performance MTD"].apply(format_arrow_value)
    return df


def format_details_data(df):
    for col in ["MtM", "Delta", "Vega"]:
        if col in df.columns:
            df[col + '_tmp'] = df[col]

            if col == "MtM":
                df["MtM"] = df["MtM"].apply(format_arrow_value)
            else:
                df[col] = df[col].apply(lambda x: f"{x * 100:.2f}%")

    return df


def format_flow_data(df):
    df["Année"] = df["Année"].dt.year
    df = df[~df.apply(lambda row: row.astype(str).str.contains('-').any(), axis=1)]
    return df


def format_flow_dict(df):
    return {key: format_flow_data(sub_df) for key, sub_df in split_by_parts_columns(df).items()}


def format_upcoming_flows_data(df):
    numeric_format_cols = ['FLUX EUR', "NOMINAL"]
    for col in numeric_format_cols:
        if col in df.columns:
            df[col + '_tmp'] = df[col]
            df[col] = df[col].apply(lambda x: f"{x:,.2f}".replace(',', ' '))

    percent_cols = ['FLUX %']
    for col in percent_cols:
        if col in df.columns:
            df[col + '_tmp'] = df[col]
            df[col] = df[col].apply(format_percentage)

    date_cols = ['DATE DE CONSTATATION', "DATE DE PAIEMENT"]
    for col in date_cols:
        if col in df.columns:
            df[col + '_tmp'] = df[col]
            df[col] = df[col].apply(format_date_short)

    return df.dropna()


def format_upcoming_flow_dict(df):
    return {key: format_upcoming_flows_data(sub_df) for key, sub_df in split_by_parts_columns(df).items()}


def format_exposure_data(df):
    df = df.dropna(how='all', axis=1)
    df_dict = {}

    for i in range(0, len(df.columns), 2):
        sub_df = df.iloc[:, i:i + 2].dropna()
        sub_df.columns = ["labels", "values"]

        if len(sub_df) > 1 :
            df_dict[df.columns[i]] = sub_df

    return df_dict


def format_shock_data(df):
    shock_data = df["shock"]
    shock_greek = df["shock_greek"].astype(float)

    spot_df = shock_data.iloc[1:].copy()
    spot_df.columns = [0 if col == 0.0 else col for col in shock_data.iloc[0]]
    spot_df = spot_df.astype(float)

    vol_df = shock_data.T.iloc[1:].copy()
    vol_df.columns = [0 if col == 0.0 else col for col in shock_data.T.iloc[0]]
    vol_df = vol_df.astype(float)

    return {
        "spot": spot_df,
        "vol": vol_df,
        "greek": shock_greek
    }


def format_product_data(df):
    df["NOMINAL" + '_tmp'] = df["NOMINAL"]
    df["NOMINAL"] = df["NOMINAL"].apply(lambda x: f"{x:,.0f}€".replace(',', ' '))

    date_cols = ['DATE DE MATURITÉ', "DATE DE TRADE", "DATE DE REMBOURS.", "DATE PROCH. CONSTAT. COUPON"]
    for col in date_cols:
        if col in df.columns:
            df[col] = df[col].apply(format_date_long)

    num_cols = ['STRIKE', 'SPOT']
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: '{:,.2f}'.format(x).replace(',', ' '))

    percent_col = ['BARRIÈRE AUTOCALL', 'TRI', 'MTM', 'PROCHAIN COUPON', 'COUPONS PERÇUS', 'COUPONS À PERCEVOIR']
    for col in percent_col:
        if col in df.columns:
            df[col] = df[col].apply(format_percentage_product)

    return df


def format_transparisation_data(df):
    df["STRIKE"] = df["STRIKE"].round(1)
    return df


def format_cashflow_data(df):
    df["FLUX %" + '_tmp'] = df["FLUX %"]
    df["FLUX %"] = df["FLUX %"].apply(format_percentage)

    date_cols = ['DATE DE CONSTAT.', 'DATE DE PAIEMENT']
    for col in date_cols:
        if col in df.columns:
            df[col + '_tmp'] = df[col]
            df[col] = df[col].apply(format_date_short)

    num_cols = ['FIXING', "FLUX EUR"]
    for col in num_cols:
        if col in df.columns:
            df[col + '_tmp'] = df[col]
            df[col] = df[col].apply(lambda x: '{:,.2f}'.format(x).replace(',', ' '))

    return df


def format_product_dict(product_dict):
    for product, df in product_dict.items():
        df = df.copy()

        df["product"] = format_product_data(df["product"])

        if not df["transparisation"].empty:
            df["transparisation"] = format_transparisation_data(df["transparisation"])

        if not df["cashflow"].empty:
            df["cashflow"] = format_cashflow_data(df["cashflow"])

        product_dict[product] = df

    return product_dict


def format_name_data_tmp(df):
    return df.set_index(df.columns[0]).to_dict()[1]


def format_subscription_data(df):
    df = df.copy()
    num_col = ["SOUSCRIPTIONS", "RACHATS", "SOLDE NET"]
    for col in num_col:
        if col in df.columns:
            df[col + '_tmp'] = df[col]
            df[col] = df[col].apply(lambda x: f"{x:,.2f}".replace(',', ' ') if not pd.isnull(x) else "")

    df["DATE"] = df["DATE"].apply(lambda x: format_date_short(x, default_value=" "))

    return df


def format_parts_data(df):
    df = df.T.copy()

    num_col = ["FRAIS DE GESTION MAX.", "FRAIS D'ENTRÉE MAX.", "FRAIS DE SORTIE MAX.", "COMMISSION DE SURPERFORMANCE"]
    for col in num_col:
        if col in df.columns:
            df[col] = df[col].apply(format_percentage_product)

    return df.T


def format_specifications_data(df):
    df = df.copy()
    df["DATE DE CRÉATION"] = df["DATE DE CRÉATION"].apply(format_date_long)
    return df


def process_data(df):
    processors = {
        "date": format_update_date,
        'line_perf': format_time_series_dict,
        'delta_action': format_time_series_dict,
        'perf_calendar': format_calendar_dict,
        'other_metric': format_other_metrics_data,
        'index_global': format_index_global_data,
        'position': format_position_data,
        "worst_best": format_worst_best_data,
        'details': format_details_data,
        "flow": format_flow_dict,
        "upcoming_flows": format_upcoming_flow_dict,
        "exposures": format_exposure_data,
        'name': format_name_data_tmp,
        "product": format_product_dict,
        "subscription": format_subscription_data,
        "parts": format_parts_data,
        "specifications": format_specifications_data
    }

    for key, func in processors.items():
        if key in df and df[key] is not None:
            df[key] = func(df[key])

    if df.get('shock') is not None:
        df['shock'] = format_shock_data(df)

    df['notes'] = format_notes_data()

    return df
#%%
