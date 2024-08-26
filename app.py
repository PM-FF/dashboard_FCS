import os
import io
import base64

import pandas as pd

import dash
from dash import html, dcc, dash_table, MATCH, callback_context
from dash.dependencies import Input, Output, State, ALL

import dash_bootstrap_components as dbc
import dash_mantine_components as dmc

import plotly.colors
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate

from flask import Flask, send_file, make_response

import urllib.parse

from Dashboard.sidebar.data import process_data


def set_fund_name(df):
    return df["specifications"]["FONDS"][0].upper()


# style.py

def set_one_color_background(df, column, base_color="#F5A701"):
    col_max = df[column].max()
    col_min = df[column].min()

    styles = []
    for index, row in df.iterrows():
        if col_max != col_min:
            percentage = ((row[column] - col_min) / (col_max - col_min)) * 100
        else:
            percentage = 100

        background_style = f'linear-gradient(90deg, {base_color} 0%, white {percentage}%)'

        style = {
            'if': {
                'filter_query': f"{{{column}}} eq {row[column]}",
                'column_id': f"{column.split('_')[0]}"
            },
            'background': background_style,
            'paddingBottom': 2,
            'paddingTop': 2
        }
        styles.append(style)

    return styles


def set_two_color_background(df, column, color_above='#3D9970', color_below='#FF4136'):
    n_bins = 100
    bounds = [i * (1.0 / n_bins) for i in range(n_bins + 1)]
    col_max = df[column].max()
    col_min = df[column].min()
    ranges = [
        ((col_max - col_min) * i) + col_min
        for i in bounds
    ]
    midpoint = 0

    styles = []
    for i in range(1, len(bounds)):
        min_bound = ranges[i - 1]
        max_bound = ranges[i]
        min_bound_percentage = bounds[i - 1] * 100
        max_bound_percentage = bounds[i] * 100

        style = {
            'if': {
                'filter_query': (
                        '{{{column}}} >= {min_bound}' +
                        (' && {{{column}}} < {max_bound}' if (i < len(bounds) - 1) else '')
                ).format(column=column, min_bound=min_bound, max_bound=max_bound),
                'column_id': f"{column.split('_')[0]}"
            },
            'paddingBottom': 2,
            'paddingTop': 2
        }
        if min_bound >= midpoint:
            background = (
                """
                linear-gradient(90deg,
                white 0%,
                white 50%,
                {color_above} 50%,
                {color_above} {midpoint_percentage}%,
                white {max_bound_percentage}%,
                white 100%)
                """.format(
                    midpoint_percentage=(max_bound_percentage / 2),
                    max_bound_percentage=max_bound_percentage,
                    color_above=color_above
                )
            )
            style['background'] = background
        else:
            background = (
                """
                linear-gradient(90deg,
                white 0%,
                white {min_bound_percentage}%,
                {color_below} 50%,
                {color_below} {min_bound_percentage}%,
                white 50%,
                white 100%)
                """.format(
                    midpoint_percentage=(min_bound_percentage / 2),
                    min_bound_percentage=min_bound_percentage,
                    color_below=color_below
                )
            )
            style['background'] = background
        styles.append(style)

    return styles


# figure.py

def get_table(df, col_params, id=None, tooltip_header=None, markdown=False):
    style_data_conditional = [
        {
            'if': {'column_id': 'index'},
            'backgroundColor': 'white',
            'color': 'rgba(0, 22, 64, 0.8)',
            'fontWeight': 'bold',
            'fontSize': '32px !important',
            'font-family': 'Roboto',
            'textAlign': 'left',
            'padding': '20px'
        },
        {
            'if': {'state': 'active'},
            'backgroundColor': 'inherit !important',
            'color': 'inherit !important',
            'border': 'none !important'
        },
        {
            'if': {'column_id': 'ISIN'},
            'cursor': 'pointer'
        }

    ]

    # TODO : if id="product-table" : ...
    if len(col_params) == 16:
        for i in range(len(df)):
            if df.loc[i, 'ISIN'] == '':
                for column in df.columns:
                    style_data_conditional.append(
                        {
                            'if': {
                                'column_id': column,
                                'row_index': i
                            },

                            'background': '#f2f2f2',
                            'borderTop': '2px solid black',
                            'fontWeight': 'bold'
                        }
                    )

    if id == "subscription-table":
        for i in range(len(df)):
            if df.loc[i, 'DATE'] == ' ':
                for column in df.columns:
                    style_data_conditional.append(
                        {
                            'if': {
                                'column_id': column,
                                'row_index': i
                            },

                            'background': '#f2f2f2',
                            'borderTop': '2px solid black',
                            'fontWeight': 'bold'
                        }
                    )

    table_params = {
        'data': df.reset_index().to_dict('records'),
        'columns': col_params,
        'tooltip_delay': 0,
        'tooltip_duration': None,
        'export_format': 'xlsx',
        'export_headers': 'display',
        'sort_action': 'custom',
        "sort_mode": 'single',
        "sort_by": [],

        'style_table': {'height': '600px'},

        'style_cell': {
            'textAlign': 'center',
            'backgroundColor': 'white',
            'color': 'black',
            'fontSize': '16px',
            'font-family': 'Roboto',
            'height': '60px',
            'minWidth': '60px',
            'width': '80px',
            'maxWidth': '150px',
            'whiteSpace': 'normal',
            "border": "none"
        },
        'style_header': {
            'backgroundColor': 'rgba(0, 22, 64, 1)',
            'fontWeight': 'bold',
            'fontSize': '34px',
            'font-family': 'Roboto',
            'color': 'white',
            "padding": "10px",
            "border": "1px",
            'borderBottom': '2px solid #f5a701',
            'letterSpacing': '2px'
        },
        'style_data_conditional': style_data_conditional
    }

    if id is not None:
        table_params['id'] = id

    if tooltip_header is not None:
        table_params['tooltip_header'] = tooltip_header

    if markdown:
        table_params['markdown_options'] = {'html': True}

    return dash_table.DataTable(**table_params)


def get_calendar_conditional_styles(df):
    style_data_conditional = [
        {
            'if': {
                'column_id': 'Year',
                'row_index': 'even'
            },
            'color': '#fff',
            'backgroundColor': 'rgba(0, 22, 64, 1)',
            'borderBottom': 'none',
            'textAlign': 'center',
            'verticalAlign': 'bottom'
        },
        {
            'if': {
                'column_id': 'Year',
                'row_index': 'odd',
            },
            'color': 'white',
            'backgroundColor': 'rgba(0, 22, 64, 1)',
            'borderTop': 'none',
            'textAlign': 'center',
            'verticalAlign': 'middle'
        },
        {
            'if': {'state': 'active'},
            'backgroundColor': 'transparent !important',
            'color': 'inherit',
            'border': 'none !important'
        },
        {
            'if': {'column_id': df.columns[-1]},
            'borderLeft': '3px solid rgba(0, 22, 64, 1)'
        }
    ]

    for i in range(len(df)):
        if i % 4 < 2:
            for column in df.columns:
                if column != 'Year':
                    style_data_conditional.append(
                        {
                            'if': {
                                'column_id': column,
                                'row_index': i
                            },
                            'backgroundColor': 'white'
                        }
                    )
        else:
            for column in df.columns:
                if column != 'Year':
                    style_data_conditional.append(
                        {
                            'if': {
                                'column_id': column,
                                'row_index': i
                            },
                            'backgroundColor': '#f2f2f2'
                        }
                    )
    return style_data_conditional


def get_calendar_table(df, col_params):
    return dash_table.DataTable(
        data=df.reset_index().to_dict('records'),
        id=f"calendar-table",
        columns=col_params,
        export_format='xlsx',
        export_headers='display',
        style_table={'height': '600px',
                     'width': '100%',
                     'border': 'none'},
        style_cell={
            'textAlign': 'center',
            'backgroundColor': 'white',
            'color': 'black',
            'fontSize': '20px',
            'font-family': 'Roboto',
            'height': '60px',
            'minWidth': '80px', 'width': '80px', 'maxWidth': '80px',
            'whiteSpace': 'normal',
            "border": "none"
        },
        style_header={
            'backgroundColor': 'rgba(0, 22, 64, 1)',
            'fontWeight': 'bold',
            'fontSize': '28px',
            'font-family': 'Roboto',
            'color': '#ffff'

        },
        style_data_conditional=get_calendar_conditional_styles(df)
        # style_header_conditional=[
        #     {
        #         'if': {'column_id': df.columns[0]},
        #         'backgroundColor': 'white',
        #         'color': 'black'
        #     }
        # ]
    )


def get_date_picker(df, part):
    df = df[part].copy()
    return dcc.DatePickerSingle(
        id={'type': 'rebase-date-picker', 'index': part},
        min_date_allowed=df['Date'].min(),
        max_date_allowed=df['Date'].max(),
        initial_visible_month=df['Date'].min(),
        date=str(df['Date'].min()),
        style={
            'backgroundColor': '#f0f0f0',
            'color': '#000',
            'fontSize': '10px'
        },
        with_portal=True,
        display_format='YYYY-MM-DD'
    )


def create_figure(add_yaxis=False, legend_pos="middle"):
    grid_color = '#e3e6ee'
    background_color = '#ffffff'
    tick_pad = 20
    layout = {
        'height': 700,
        'plot_bgcolor': background_color,
        'xaxis': {
            'title': '',
            'gridcolor': grid_color,
            'ticklen': tick_pad,
            'tickfont': {'size': 14},
            'zerolinecolor': '#B5B8BE'
        },
        'yaxis': {
            'title': '',
            'autorange': True,
            'gridcolor': grid_color,
            'tickfont': {'size': 14},
            'ticklen': tick_pad,
            'zerolinecolor': '#B5B8BE'
        },
        'legend': {
            'x': 0.5, 'y': -0.3, 'xanchor': 'center', 'orientation': 'v'
        },
        'autosize': True,
        'margin': dict(l=50, r=50, b=50, t=50),
    }

    # Ajouter un second axe Y si demandé
    if add_yaxis:
        layout['yaxis2'] = {
            'title': '',
            'autorange': True,
            'gridcolor': "white",
            'tickfont': {'size': 14},
            'ticklen': tick_pad,
            'overlaying': 'y',
            'side': 'right'
        }

    # Ajuster la position de la légende si spécifié
    if legend_pos == "right":
        layout['legend'] = {
            'x': 0.8,
            'y': -0.3,
            'xanchor': 'right',
            'orientation': 'v'
        }

    return go.Figure(layout=go.Layout(**layout))


def get_exposure_select_box(df_dict, side, selected_input=None):
    selected_input = list(data['exposures'].keys())[0 if side == "left" else 1] \
        if selected_input is None else selected_input

    return dmc.Select(
        placeholder="SELECTIONNER UNE CATÉGORIE D'EXPOSITION",
        id=f"exposure_select_box_{side}",
        searchable=False,
        data=[{"value": key, "label": key} for key in df_dict.keys()],
        className=f"exposure_select_box",
        value=selected_input
    )


def get_sensitivity_select_box(df_dict):
    label_dict = {
        "spot": "SENSIBILITÉ AU SPOT",
        "vol": "SENSIBILITÉ À LA VOLATILITÉ",
        "greek": "SENSIBILITÉ AU DELTA ET AU VEGA"
    }

    return dmc.Select(
        placeholder="SELECTIONNER UNE CATÉGORIE DE SENSIBILITÉ",
        id="sensitivity_select_box",
        data=[{"value": key, "label": label_dict[key]} for key in df_dict.keys()],
        className="sensitivity_select_box",
        value="spot"
    )


def get_product_select_box(df_dict, default_product):
    # TODO : get name value directly from the dict not with a global variable
    default_product = next(iter(data['name'].values()), None) if default_product is None else default_product
    return dmc.Select(
        placeholder="SELECTIONNER UN PRODUITS",
        value=default_product,
        id="product_select_box",
        searchable=True,
        data=[{"value": value, "label": key} for key, value in df_dict.items()],
        className="product_select_box",
    )


def get_slider(df, slider_value, id):
    if isinstance(df, dict):
        values = [float(x) for x in list(df.keys())]
    else:
        values = df

    keys = [f"+{int(val * 100)}%" if val > 0 else f"{int(val * 100)}%" if val < 0 else "0%" for val in values]
    marks = {val: key for key, val in zip(keys, values)}

    return dcc.Slider(
        id=id,
        value=slider_value,
        min=min(marks.keys()),
        max=max(marks.keys()),
        marks={v: {'label': k} for v, k in marks.items()},
        step=None
    )


# data.py

def rebase_dataframe(df, date_column, rebase_date):
    rebase_datetime = pd.to_datetime(rebase_date)
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.loc[df[date_column] >= rebase_datetime]
    return df, rebase_datetime


def normalize_series(df, date_column, series_name, base_value):
    series_data = df[series_name]
    if base_value is not None and not df.empty:
        series_data = 100 * series_data / base_value
    return series_data


# figure.py

def add_traces_to_scatter_fig(fig, df, date_column, mode, line_shape, color_dict, label_dict, line_width,
                              add_yaxis=False):
    if add_yaxis:
        first = True
        for col in color_dict.keys():
            if col != date_column:
                fig.add_trace(go.Scatter(
                    x=df[date_column],
                    y=df[col],
                    mode=mode,
                    name=label_dict.get(col, col),
                    line=dict(shape=line_shape, color=color_dict[col], width=line_width),
                    yaxis='y2' if first else 'y'
                ))
                first = False
    else:
        for col in color_dict.keys():
            if col != date_column:
                fig.add_trace(go.Scatter(
                    x=df[date_column],
                    y=df[col],
                    mode=mode,
                    name=label_dict.get(col, col),
                    line=dict(shape=line_shape, color=color_dict[col], width=line_width)
                ))


def add_traces_to_bar_fig(fig, df, date_column, color_dict, label_dict):
    for col in color_dict.keys():
        if col != date_column and col in color_dict:
            fig.add_trace(go.Bar(
                x=df[date_column],
                y=df[col],
                name=label_dict.get(col, col),
                marker=dict(color=color_dict[col], cornerradius="5%")
            ))


def get_label_and_color_dict(df, ptf_ticker):
    df_equity = df[(df['Ticker'].str.contains("Equity", case=False)) & (df['Ticker'] == ptf_ticker)]
    df_non_equity = df[~df['Ticker'].str.contains("Equity", case=False)]
    df_filtered = pd.concat([df_equity, df_non_equity])

    color_dict = {}
    label_dict = {}

    color_list = ["#17202A", "#B3B6B7", "#F5A701", "#00164F"]
    for i, (_, row) in enumerate(df_filtered.iterrows()):
        ticker = row['Ticker']
        label = row['Label']

        color = color_list[i] if i < len(color_list) else "#00164F"
        color_dict[ticker] = color

        label_dict[ticker] = label

    return color_dict, label_dict


def get_perf_plot(df, part, rebase_date=None):
    params = {
        'x': 'Date',
        'mode': 'lines',
        'line_shape': 'linear',
        'graph_id': {'type': 'performance-graph', 'index': part},
        'line_width': 3
    }

    ptf_ticker = [col for col in df.columns if 'Equity' in col][0]

    color_dict, label_dict = get_label_and_color_dict(data['labels'], ptf_ticker)

    fig = create_figure(add_yaxis=False)

    if rebase_date:
        df, rebase_datetime = rebase_dataframe(df, params['x'], rebase_date)
        for col in df.columns:
            if col != params['x'] and col in color_dict:
                base_value = df.loc[df[params['x']] == rebase_datetime, col].iloc[0]
                df[col] = normalize_series(df, params['x'], col, base_value).copy()

    add_traces_to_scatter_fig(fig, df,
                              params['x'],
                              params['mode'],
                              params['line_shape'],
                              color_dict,
                              label_dict,
                              params['line_width'],
                              add_yaxis=False)

    return dcc.Graph(id=params['graph_id'], figure=fig)


def get_line_perf_tabs(df):
    tab_labels = [f"PART - {part}" for part in df.keys()]

    tabs_list = dmc.TabsList([dmc.Tab(label, value=str(index)) for index, label in enumerate(tab_labels)])
    tabs_panels = [
        dmc.TabsPanel(
            children=[
                get_perf_plot(df[part], part),
                html.Div(get_date_picker(df, part), id='date-picker-container')
            ],
            value=str(index)
        )
        for index, part in enumerate(df.keys())
    ]

    tabs = dmc.Tabs(
        [tabs_list, *tabs_panels],
        color="orange",
        orientation="horizontal",
        value="0",
        id="line-perf-tabs-container"
    )

    return tabs


def get_delta_action_plot(df, part):
    params = {
        'x': 'Date',
        'mode': 'lines',
        'line_shape': 'linear',
        'graph_id': f'delta-action-graph',
        'line_width': 3
    }

    ptf_ticker = [col for col in df.columns if 'Equity' in col][0]

    color_dict = {
        'Tendance Delta': '#F5A701',
        ptf_ticker: "#00164F"
    }

    label_dict = {
        'Tendance Delta': "Proxy delta (RHS)",
        ptf_ticker: f"Valeur liquidative (Part {part})"
    }

    fig = create_figure(add_yaxis=True)

    add_traces_to_scatter_fig(fig, df,
                              params['x'],
                              params['mode'],
                              params['line_shape'],
                              color_dict,
                              label_dict,
                              params['line_width'],
                              add_yaxis=True)

    return dcc.Graph(id=params['graph_id'], figure=fig)


def get_delta_action_tabs(df):
    tab_labels = [f"PART - {part}" for part in df.keys()]

    tabs_list = dmc.TabsList([dmc.Tab(label, value=str(index)) for index, label in enumerate(tab_labels)])
    tabs_panels = [
        dmc.TabsPanel(get_delta_action_plot(df[part], part), value=str(index))
        for index, part in enumerate(df.keys())
    ]

    tabs = dmc.Tabs(
        [tabs_list, *tabs_panels],
        color="orange",
        orientation="horizontal",
        value="0",
        id="delta-action-tabs-container"
    )

    return tabs


def get_historical_valuation_plot(df, product):
    df = df["product"][product]['valuation']

    params = {
        'x': 'Date',
        'mode': 'lines',
        'line_shape': 'linear',
        'graph_id': 'valuation-graph',
        'line_width': 3
    }

    color_dict = {}
    label_dict = {}

    if len(df.columns) > 1:
        color_dict[df.columns[1]] = '#F5A701'
        label_dict[df.columns[1]] = df.columns[1]

    if len(df.columns) > 2:
        color_dict[df.columns[2]] = "#00164F"
        label_dict[df.columns[2]] = df.columns[2]

    fig = create_figure(add_yaxis=True)

    add_traces_to_scatter_fig(fig, df,
                              params['x'],
                              params['mode'],
                              params['line_shape'],
                              color_dict,
                              label_dict,
                              params['line_width'],
                              add_yaxis=True)

    return dcc.Graph(id=params['graph_id'], figure=fig)


def get_var_plot(df):
    params = {
        'x': 'shock',
        'mode': 'lines+markers',
        'line_shape': 'linear',
        'graph_id': 'var-graph',
        'line_width': 3
    }

    color_dict = {
        'VaR': "#00164F"
    }

    label_dict = {
        'VaR': "Value-At-Risk 99% sur 20 jours"
    }

    fig = create_figure(add_yaxis=False)

    add_traces_to_scatter_fig(fig, df,
                              params['x'],
                              params['mode'],
                              params['line_shape'],
                              color_dict,
                              label_dict,
                              params['line_width'],
                              add_yaxis=False)

    fig.update_layout(showlegend=True,
                      yaxis=dict(autorange=False,
                                 range=[0 - (0.1 * df["VaR"].max()),
                                        (1.1 * df["VaR"].max())]),
                      xaxis=dict(autorange=True)
                      )

    return dcc.Graph(id=params['graph_id'], figure=fig)


def get_shock_plot(df):
    params = {
        'x': 'shock',
        'mode': 'lines+markers',
        'line_shape': 'linear',
        'graph_id': 'shock-graph',
        'line_width': 3
    }

    color_dict = {col: '#F5A701' if col == 'VEGA' else '#00164F' for col in df.columns if col != params['x']}

    label_dict = {}

    fig = create_figure(add_yaxis=False)

    add_traces_to_scatter_fig(fig, df,
                              params['x'],
                              params['mode'],
                              params['line_shape'],
                              color_dict,
                              label_dict,
                              params['line_width'],
                              add_yaxis=False)

    return dcc.Graph(id=params['graph_id'], figure=fig)


def get_schedule_bar_plot(df):
    params = {
        'x': 'Année',
        'graph_id': 'schedule-graph'
    }

    color_dict = {
        'Flux EUR': "#001640",
        'Flux EUR (wc)': '#B3B6B7',
        'Flux EUR (woc)': '#F5A701'
    }

    label_dict = {
        'Flux EUR': "FLUX ATTENDUS PAR AN (EUR)",
        'Flux EUR (wc)': 'MONTANTS RAPPELÉS PAR AN (EUR) (COUPONS INCLUS)',
        'Flux EUR (woc)': 'MONTANTS RAPPELÉS PAR AN (EUR) (HORS COUPONS)'
    }

    fig = create_figure(legend_pos="right")

    add_traces_to_bar_fig(fig, df, params['x'], color_dict, label_dict)

    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=df['Année'],
            ticktext=[f"{date}" for date in df['Année']]
        ))

    return dcc.Graph(id=params['graph_id'], figure=fig)


def get_pie_plot(df):
    if len(df) <= 6:
        colors = ["#324466", "#f7b833", "#66738c", "#624200", "#b2b9c5", "#fdedcc"]
    else:
        colors = plotly.colors.qualitative.Set2_r

    figure = go.Figure(data=[go.Pie(labels=df['labels'],
                                    values=df['values'],
                                    hole=.3,
                                    marker=dict(colors=colors),
                                    hoverinfo='label+percent+value',  # Customize hover info
                                    textinfo='label+percent' if len(df) <= 5 else 'percent',
                                    insidetextorientation='horizontal',
                                    textfont=dict(size=14, family="Arial, sans-serif"),

                                    )])

    figure.update_traces(hoverinfo='label+percent+value',
                         hoverlabel=dict(
                             bgcolor="white",
                             font=dict(
                                 family="Rockwell",
                                 size=16,
                             ),
                             bordercolor="#333"
                         ))

    figure.update_layout(transition_duration=500)

    return figure


# data.py

def get_sri_svg_path(base_path, risk_number):
    return f"{base_path}/SRI_{risk_number}.svg"


def load_product_data(folder_path):
    product_dict = {}
    for file in os.listdir(folder_path):
        if file.endswith('.xlsx'):
            file_path = os.path.join(folder_path, file)
            xls = pd.ExcelFile(file_path)
            sheet_dict = {}
            for sheet_name in xls.sheet_names:
                sheet_dict[sheet_name] = pd.read_excel(xls, sheet_name)
            product_dict[file.split(".")[0]] = sheet_dict
    return product_dict


def safe_read_excel(file_path, sheet_name, **kwargs):
    try:
        return pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
    except ValueError as e:
        return None


def load_data():
    base_dir = r"C:\Users\H6289\Documents\workspace\dsProjects\Dashboard"
    # base_dir = os.path.dirname(__file__)

    paths = dict(
        data=os.path.join(base_dir, 'data', 'data.xlsx'),
        product=os.path.join(base_dir, 'data', 'product'),
        SRI=r"/assets/images/SRI/"
    )

    return {
        "date": safe_read_excel(paths['data'], sheet_name='date', header=None),
        'index_global': safe_read_excel(paths['data'], sheet_name='index_global', index_col=0),
        'line_perf': safe_read_excel(paths['data'], sheet_name='perf_line'),
        'other_metric': safe_read_excel(paths['data'], sheet_name='other_metric', index_col=0),
        'perf_calendar': safe_read_excel(paths['data'], sheet_name='calendar'),
        'position': safe_read_excel(paths['data'], sheet_name='position'),
        'details': safe_read_excel(paths['data'], sheet_name='details', index_col=0),
        "worst_best": safe_read_excel(paths['data'], sheet_name='worst_best'),
        "flow": safe_read_excel(paths['data'], sheet_name='total_flows'),
        "upcoming_flows": safe_read_excel(paths['data'], sheet_name='upcoming_flows'),
        "exposures": safe_read_excel(paths['data'], sheet_name='exposures'),
        'delta_action': safe_read_excel(paths['data'], sheet_name='delta_action'),
        "shock": safe_read_excel(paths['data'], sheet_name='shock', header=None),
        "shock_greek": safe_read_excel(paths['data'], sheet_name='shock_greek'),
        "var": safe_read_excel(paths['data'], sheet_name='var'),
        "subscription": safe_read_excel(paths['data'], sheet_name='subscription'),
        "parts": safe_read_excel(paths['data'], sheet_name='parts', index_col=0),
        "specifications": safe_read_excel(paths['data'], sheet_name='specifications'),
        "name": safe_read_excel(paths['data'], sheet_name='names', header=None),
        "labels": safe_read_excel(paths['data'], sheet_name='labels'),
        "product": load_product_data(paths["product"]),
        "SRI": paths["SRI"]
    }


# figure.py
def create_other_metric_table(df):
    col_params = [{"name": "", "id": "index"}]
    col_params.extend([
        {'name': col, 'id': col}
        for col in df['other_metric'].columns])

    return get_table(df=df['other_metric'],
                     col_params=col_params,
                     id="metric-table")


def create_index_table(df):
    col_params = [{"name": "", "id": "index"}]
    col_params.extend([
        {"name": col, "id": "index" if col.strip() == "" else col, "presentation": "markdown"} if col in [
            "Performance YTD"]
        else {"name": col, "id": "index" if col.strip() == "" else col}
        for col in df["index_global"].columns if "_tmp" not in col
    ])
    return get_table(df=df['index_global'],
                     col_params=col_params,
                     id='index-table',
                     markdown=True)


def create_perf_calendar_table(df, part):
    col_params = [
        {'name': '' if col == 'Year' else col, 'id': col}
        for col in df['perf_calendar'][part].columns
    ]
    return get_calendar_table(df['perf_calendar'][part], col_params)


def get_perf_calendar_tabs(df):
    tab_labels = [f"PART - {part}" for part in df['perf_calendar'].keys()]

    tabs_list = dmc.TabsList([dmc.Tab(label, value=str(index)) for index, label in enumerate(tab_labels)])
    tabs_panels = [
        dmc.TabsPanel(create_perf_calendar_table(df, part), value=str(index))
        for index, part in enumerate(df['perf_calendar'].keys())
    ]

    tabs = dmc.Tabs(
        [tabs_list, *tabs_panels],
        color="orange",
        orientation="horizontal",
        value="0",
        id="calendar-tabs-container"
    )

    return tabs


def create_position_table(df):
    col_params = [{"name": col, "id": col, "presentation": "markdown"} if col in ["MtM", "Upside"]
                  else {"name": col, "id": col} for col in df["position"].columns if "_tmp" not in col]

    tooltip_header = {col: df["notes"].get(col, '') for col in df["position"].columns}
    return html.Div([
        get_table(df=df['position'],
                  col_params=col_params,
                  id="position-table",
                  tooltip_header=tooltip_header,
                  markdown=True),
        html.Div(id='popovers-container'),
        dcc.Store(id='popover-state', data={'open': None}),
        html.Div(id='icon-click-listener', style={'display': 'none'})
    ])


def create_worst_best_table(df):
    col_params = [{"name": col, "id": col, "presentation": "markdown"} if col in ["Performance MTD"]
                  else {"name": col, "id": col} for col in df["worst_best"].columns if "_tmp" not in col]
    return get_table(df=df["worst_best"],
                     col_params=col_params,
                     id="worst-best-table",
                     markdown=True)


def create_details_table(df):
    col_params = [{"name": "", "id": "index"}]

    col_params.extend([
        {"name": col, "id": "index" if col.strip() == "" else col, "presentation": "markdown"} if col in ["MtM"]
        else {"name": col, "id": "index" if col.strip() == "" else col}
        for col in df["details"].columns if "_tmp" not in col
    ])
    return get_table(df=df["details"],
                     col_params=col_params,
                     id="details-table",
                     markdown=True)


def create_upcoming_flows_table(df, shock):
    col_params = [{"name": col, "id": col} for col in df["upcoming_flows"][shock].columns if "_tmp" not in col]
    return get_table(df=df['upcoming_flows'][shock],
                     col_params=col_params,
                     id="flow-table")


def create_subscription_table(df):
    col_params = [{"name": col, "id": col} for col in df["subscription"].columns if "_tmp" not in col]
    return get_table(df=df['subscription'],
                     col_params=col_params,
                     id="subscription-table")


def create_shares_table(df):
    col_params = [{"name": "", "id": "index"}]
    col_params.extend([
        {"name": col, "id": "index" if col == " " else col}
        for col in df["parts"].columns if "_tmp" not in col])
    return get_table(df=df['parts'],
                     col_params=col_params,
                     id="shares-table")


# Not in figure_position

def create_cashflow_table(df, product):
    col_params = [{"name": col, "id": col} for col in df["product"][product]['cashflow'].columns if "_tmp" not in col]
    return get_table(df=df["product"][product]['cashflow'],
                     col_params=col_params,
                     id="cashflow-table")


def create_screening_table(df, product):
    col_params = [{"name": col, "id": col} for col in df["product"][product]['transparisation'].columns if
                  "_tmp" not in col]
    return get_table(df=df["product"][product]['transparisation'],
                     col_params=col_params,
                     id="transparisation-table")


def split_to_sentences(text):
    if text == "-":
        return text
    else:
        sentences = text.split(". ")
        sentences = [sentence + '.' if not sentence.endswith('.') else sentence for sentence in sentences[:-1]] + [
            sentences[-1]]
        return sentences


def get_overview_figure(df, product):
    pdt = df[product]["product"]
    description = split_to_sentences(pdt["DESCRIPTIF DU PRODUIT"][0])

    content = html.Div(className='fpd-body', children=[
        html.Div(className='fpd-container', children=[
            html.Div(className='fpd-header', children=[get_product_select_box(data['name'], product)]),
            html.Span(className='fpd-ticker', children=f'ISIN: {pdt["ISIN"][0]}'),
            html.H2(className='fpd-description-title', children="Caractéristiques du contrat"),
            html.Div(className='fpd-section', children=[
                html.Div(className='fpd-vertical-separator'),
                html.Div(className='fpd-column', children=[
                    html.Div(className='fpd-block', children=[
                        html.Div(className='fpd-pair', children=[
                            html.Span(className='fpd-label', children='Date de Trade'),
                            html.Span(className='fpd-value', children=pdt["DATE DE TRADE"][0]),
                        ]),
                        html.Div(className='fpd-pair', children=[
                            html.Span(className='fpd-label', children='Date de Maturité'),
                            html.Span(className='fpd-value', children=pdt["DATE DE MATURITÉ"][0]),
                        ]),
                    ]),
                    html.Div(className='fpd-block', children=[
                        html.Div(className='fpd-pair', children=[
                            html.Span(className='fpd-label', children='Date de Remboursement'),
                            html.Span(className='fpd-value', children=pdt["DATE DE REMBOURS."][0]),
                        ]),
                        html.Div(className='fpd-pair', children=[
                            html.Span(className='fpd-label', children='Nominal'),
                            html.Span(className='fpd-value', children=pdt["NOMINAL"][0]),
                        ]),
                    ]),
                ]),
                html.Div(className='fpd-vertical-separator'),
                html.Div(className='fpd-column', children=[
                    html.Div(className='fpd-block', children=[
                        html.Div(className='fpd-pair', children=[
                            html.Span(className='fpd-label', children='TRI'),
                            html.Span(className='fpd-value', children=pdt["TRI"][0]),
                        ]),
                    ]),
                    html.Div(className='fpd-block', children=[
                        html.Div(className='fpd-pair', children=[
                            html.Span(className='fpd-label', children='MTM'),
                            html.Span(className='fpd-value', children=pdt["MTM"][0]),
                        ]),
                    ]),
                ]),
            ]),
            html.Div(className='fpd-section', children=[
                html.Div(className='fpd-vertical-separator'),
                html.Div(className='fpd-column', children=[
                    html.Div(className='fpd-block', children=[
                        html.Div(className='fpd-pair', children=[
                            html.Span(className='fpd-label', children='Sous-jacent'),
                            html.Span(className='fpd-value', children=pdt["SOUS-JACENT"][0]),
                        ]),

                        html.Div(className='fpd-pair', children=[
                            html.Span(className='fpd-label', children='Barrière Autocall') if pdt["BARRIÈRE AUTOCALL"][
                                                                                                  0] != " - " else None,
                            html.Span(className='fpd-value', children=pdt["BARRIÈRE AUTOCALL"][0]) if
                            pdt["BARRIÈRE AUTOCALL"][0] != " - " else None,
                        ])
                    ]),
                    html.Div(className='fpd-block', children=[
                        html.Div(className='fpd-pair', children=[
                            html.Span(className='fpd-label', children='Spot'),
                            html.Span(className='fpd-value', children=pdt["SPOT"][0]),
                        ]),
                        html.Div(className='fpd-pair', children=[
                            html.Span(className='fpd-label', children='Strike'),
                            html.Span(className='fpd-value', children=pdt["STRIKE"][0]),
                        ]),
                    ]),
                ]),
                html.Div(className='fpd-vertical-separator'),
                html.Div(className='fpd-column', children=[
                    html.Div(className='fpd-block', children=[
                        html.Div(className='fpd-pair', children=[
                            html.Span(className='fpd-label', children='Date Prochaine Constatation'),
                            html.Span(className='fpd-value', children=pdt["DATE PROCH. CONSTAT. COUPON"][0]),
                        ]),
                        html.Div(className='fpd-pair', children=[
                            html.Span(className='fpd-label', children='Coupons Perçus'),
                            html.Span(className='fpd-value', children=pdt["COUPONS PERÇUS"][0]),
                        ]),
                    ]),
                    html.Div(className='fpd-block', children=[
                        html.Div(className='fpd-pair', children=[
                            html.Span(className='fpd-label', children='Prochain Coupon'),
                            html.Span(className='fpd-value', children=pdt["PROCHAIN COUPON"][0]),
                        ]),
                        html.Div(className='fpd-pair', children=[
                            html.Span(className='fpd-label', children='Coupons à Percevoir'),
                            html.Span(className='fpd-value', children=pdt["COUPONS À PERCEVOIR"][0]),
                        ]),
                    ]),
                ]),
            ]),
            html.Div(className='fpd-horizontal-separator'),
            html.Div(className='fpd-description-section', children=[
                html.H2(className='fpd-description-title', children='Descriptif du produit'),
                html.Ul(className='fpd-description-list', children=[html.Li(item) for item in description]),
            ]),
        ]),
    ])

    return content


def get_specification_figure(df):
    commentary = split_to_sentences(df['COMMENTAIRE'][0])
    target = split_to_sentences(df['OBJECTIF'][0])

    content = html.Div(className='fpd-body', children=[
        html.Div(className='fpd-container', children=[
            html.H2(className='fpd-description-title', children="Fournisseurs du service"),
            html.Div(className='fpd-section', children=[
                html.Div(className='fpd-vertical-separator'),
                html.Div(className='fpd-column', children=[
                    html.Div(className='fpd-block', children=[
                        html.Div(className='fpd-pair', children=[
                            html.Span(className='fpd-label', children='FONDS'),
                            html.Span(className='fpd-value', children=df["FONDS"][0]),
                        ]),
                        html.Div(className='fpd-pair', children=[
                            html.Span(className='fpd-label', children='FORME JURIDIQUE'),
                            html.Span(className='fpd-value', children=df["FORME JURIDIQUE"][0]),
                        ]),
                    ]),
                    html.Div(className='fpd-block', children=[
                        html.Div(className='fpd-pair', children=[
                            html.Span(className='fpd-label', children='DATE DE CRÉATION'),
                            html.Span(className='fpd-value', children=df["DATE DE CRÉATION"][0]),
                        ]),
                        html.Div(className='fpd-pair', children=[
                            html.Span(className='fpd-label', children='ACTIF NET DU FONDS'),
                            html.Span(className='fpd-value', children=df["ACTIF NET DU FONDS"][0]),
                        ]),
                    ]),
                ]),
                html.Div(className='fpd-vertical-separator'),
                html.Div(className='fpd-column', children=[
                    html.Div(className='fpd-block', children=[
                        html.Div(className='fpd-pair', children=[
                            html.Span(className='fpd-label', children='SOCIÉTÉ DE GESTION'),
                            html.Span(className='fpd-value', children=df["SOCIÉTÉ DE GESTION"][0]),
                        ]),
                        html.Div(className='fpd-pair', children=[
                            html.Span(className='fpd-label', children='SOCIÉTÉ DE CONSEIL'),
                            html.Span(className='fpd-value', children=df["SOCIÉTÉ DE CONSEIL"][0]),
                        ]),
                    ]),
                    html.Div(className='fpd-block', children=[
                        html.Div(className='fpd-pair', children=[
                            html.Span(className='fpd-label', children='GÉRANTS'),
                            html.Span(className='fpd-value', children=df["GÉRANTS"][0]),
                        ]),
                    ]),
                ]),
            ]),
            html.Div(className='fpd-section', children=[
                html.Div(className='fpd-vertical-separator'),
                html.Div(className='fpd-column', children=[
                    html.Div(className='fpd-block', children=[
                        html.Div(className='fpd-pair', children=[
                            html.Span(className='fpd-label', children='FRÉQ. DE VALORISATION'),
                            html.Span(className='fpd-value', children=df["FRÉQ. DE VALORISATION"][0]),
                        ]),
                        html.Div(className='fpd-pair', children=[
                            html.Span(className='fpd-label', children='TYPE DE PART'),
                            html.Span(className='fpd-value', children=df["TYPE DE PART"][0]),
                        ]),
                    ]),
                    html.Div(className='fpd-block', children=[
                        html.Div(className='fpd-pair', children=[
                            html.Span(className='fpd-label', children='SOUSCRIPTIONS/RACHATS'),
                            html.Span(className='fpd-value', children=df["SOUSCRIPTIONS/RACHATS"][0]),
                        ]),
                        html.Div(className='fpd-pair', children=[
                            html.Span(className='fpd-label', children='ÉLIGIBILITÉ'),
                            html.Span(className='fpd-value', children=df["ÉLIGIBILITÉ"][0]),
                        ]),
                    ]),
                ]),
                html.Div(className='fpd-vertical-separator'),
                html.Div(className='fpd-column', children=[
                    html.Div(className='fpd-block', children=[
                        html.Div(className='fpd-pair', children=[
                            html.Span(className='fpd-label', children='DÉLÉGATION COMPTABLE'),
                            html.Span(className='fpd-value', children=df["DÉLÉGATION COMPTABLE"][0]),
                        ]),
                        html.Div(className='fpd-pair', children=[
                            html.Span(className='fpd-label', children='DÉPOSITAIRE'),
                            html.Span(className='fpd-value', children=df["DÉPOSITAIRE"][0]),
                        ]),
                    ]),
                    html.Div(className='fpd-block', children=[
                        html.Div(className='fpd-pair', children=[
                            html.Span(className='fpd-label', children='CATÉGORIE QUANTALYS'),
                            html.Span(className='fpd-value', children=df["CATÉGORIE QUANTALYS"][0]),
                        ]),
                        html.Div(className='fpd-pair', children=[
                            html.Span(className='fpd-label', children='CATÉGORIE MORNINGSTAR'),
                            html.Span(className='fpd-value', children=df["CATÉGORIE MORNINGSTAR"][0]),
                        ]),
                    ]),
                ]),
            ]),
            html.Div(className='fpd-horizontal-separator'),
            html.Div(className='fpd-description-section', children=[
                html.H2(className='fpd-description-title', children="Niveau de risque SRI"),
                html.Div([
                    html.Img(src=get_sri_svg_path(data["SRI"], "4"),
                             style={'display': 'block', 'margin-left': '100px', 'margin-right': 'auto'})
                ], style={'text-align': 'center', 'display': 'flex', 'align-items': 'center',
                          'justify-content': 'center'})

            ]),
            html.Div(className='fpd-horizontal-separator'),
            html.Div(className='fpd-description-section', children=[
                html.H2(className='fpd-description-title', children="Objectif et stratégie d'investissement"),
                html.Ul(className='fpd-description-list', children=[html.Li(item) for item in target]),
            ]),
            html.Div(className='fpd-horizontal-separator'),
            html.Div(className='fpd-description-section', children=[
                html.H2(className='fpd-description-title', children="Commentaire de gestion"),
                html.Ul(className='fpd-description-list', children=[html.Li(item) for item in commentary]),
            ]),
        ]),
    ])

    return content


# In figure_position

def get_tables_dict(df):
    return {
        'other_metric': create_other_metric_table(df),
        'index': create_index_table(df),
        'position': create_position_table(df),
        "worst_best": create_worst_best_table(df),
        "details": create_details_table(df),
        "shares": create_shares_table(df)
    }


def get_element_dict(df, tables):
    return {
        "metric": tables['other_metric'],
        "index": tables['index'],
        "position": tables["position"],
        "worst_best": tables["worst_best"],
        "details": tables["details"],
        "exposures": dcc.Graph(id='exposures-graph'),
        "exposures_select_box_left": get_exposure_select_box(df["exposures"], side="left"),
        "exposures_select_box_right": get_exposure_select_box(df["exposures"], side="right"),
        "sensitivity": dcc.Graph(id="shock-graph"),
        "sensitivity_select_box": get_sensitivity_select_box(df["shock"]),
        "var": get_var_plot(df["var"]),
        "slider": get_slider(df["shock"]['spot']['shock'], 0.1, "slider-callback"),
        "flow-slider": get_slider(df["flow"], 0, "flow-slider-callback"),
        "shares": tables["shares"],
        "specifications": get_specification_figure(df["specifications"])
    }


def get_element_position(data):
    tables = get_tables_dict(data)
    return get_element_dict(data, tables)


# app.py

def set_card(component, title):
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                html.H4(title, className='card-title'),
                component
            ])
        ),
    ])


def get_performance_display(fig_position, df, part):
    content = html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("ÉVOLUTION DE LA PERFORMANCE", className='card-title'),
                            get_line_perf_tabs(df['line_perf']),
                        ], id='graph-card-body-2')
                    )
                ], width=12, style={"height": "100%"}, id='col-graph-2')
            ], id='row-graph-1', justify='center'),
            html.Br(),
            dbc.Row([
                dbc.Col(set_card(fig_position["metric"], "PERFORMANCE DU FOND"), width=5, id='col-metric'),
                dbc.Col(set_card(fig_position["index"], "INDICES GLOBAUX"), width=7, id='col-index-global'),
            ], id='row-metric-1', justify='around'),
            html.Br(),
            dbc.Row([
                dbc.Col(set_card(get_perf_calendar_tabs(data), "PERFORMANCES CALENDAIRES"), width=12,
                        id='col-calendar'),
            ], id='row-calendar', justify='around'),
        ], fluid=True)
    ], className='content-area')

    return content


def get_portfolio_display(fig_position):
    content = html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("POSITIONS EN PORTEFEUILLE", className='card-title'),
                            fig_position["position"]
                        ], id='position-card-body')
                    )
                ], width=12, id='col-position')
            ], id='row-graph-2', justify='center'),
            html.Div(id='download-url', style={'display': 'none'}),
            html.Div(id='dummy-div', style={'display': 'none'}),
            html.Br(),
            dbc.Row([
                dbc.Col(set_card(fig_position["details"], "DÉTAILS PAR POCHE"), width=5, id='col-details'),
                dbc.Col(set_card(fig_position["worst_best"], "MEILLEURES ET PIRES PERFORMANCES"), width=7, id='col-wb'),
            ], id='row-metric-2', justify='around')
        ], fluid=True)
    ], className='content-area')

    return content


def get_schedule_display(fig_position, df, shock="0"):
    content = html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("ÉCHÉANCIER & PROCHAINS ÉVÈNEMENTS", className='card-title'),
                            get_schedule_bar_plot(df["flow"][shock]),
                            html.Div([html.H4("Impact d'un choc sur les prix spot", className='slider-title'),
                                      get_slider(df["flow"], 0, "flow-slider-callback")],
                                     id="flow-slider-container")
                        ], id='schedule-card-body')
                    )
                ], width=12, id='col-schedule')
            ], id='row-graph-3', justify='center'),
            html.Br(),
            dbc.Row([
                dbc.Col(set_card(create_upcoming_flows_table(df, shock), "FLUX À VENIR"), width=12,
                        id='col-upcoming-flow')
            ], id='row-metric-3', justify='around')
        ], fluid=True)
    ], className='content-area')

    return content


def get_exposures_display(fig_position):
    content = html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("RÉPARTITIONS DES EXPOSITIONS (% ACTIF NET)", className='card-title'),
                            dbc.Row([
                                dbc.Col(fig_position["exposures_select_box_left"], width=6),
                                dbc.Col(fig_position["exposures_select_box_right"], width=6)
                            ]),
                            dbc.Row([
                                dbc.Col(dcc.Graph(id='exposures-graph-left'), width=6),
                                dbc.Col(dcc.Graph(id='exposures-graph-right'), width=6)
                            ])
                        ], id='exposure-card-body')
                    )
                ], width=12, id='col-exposure')
            ], id='row-graph-4', justify='center')
        ], fluid=True)
    ], className='content-area')

    return content


def get_sensitivity_display(fig_position, df, part):
    content = html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("DELTA 'ACTIONS'", className='card-title'),
                            get_delta_action_tabs(df["delta_action"]),
                        ], id='graph-card-body-1')
                    )
                ], width=12, id='col-graph-1')
            ], id='row-graph-5', justify='center'),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("SENSIBILITÉS DU PORTEFEUILLE", className='card-title'),
                            fig_position["sensitivity_select_box"],
                            fig_position["sensitivity"],
                            html.Div(fig_position["slider"], id='slider-container',
                                     style={'width': '90%', 'margin': '0 auto'}),
                        ], id='sensitivity-card-body')
                    )
                ], width=7),
                dbc.Col([set_card(fig_position["var"], "VALUE-AT-RISK")], width=5, id='col-var')
            ], id='row-sensibility', justify='around')
        ], fluid=True)
    ], className='content-area')

    return content


def get_focus_product_display(df, product=None):
    product = next(iter(df['product'])) if product is None else product

    content = html.Div([
        dcc.Store(id='selected-product'),
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("FOCUS PRODUIT", className='card-title'),
                            get_overview_figure(df["product"], product=product),
                        ]),
                        id="focus-card")
                ])
            ], justify='center'),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("VALORISATIONS HISTORIQUES", className='card-title'),
                            get_historical_valuation_plot(df, product=product)
                        ])
                    )
                ])
            ], justify='center'),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    set_card(create_cashflow_table(df, product=product), "ENSEMBLE DES CASHFLOWS")
                ], width=6),
                dbc.Col([
                    set_card(create_screening_table(df, product=product), "TRANSPARISATION DU SOUS-JACENT")
                ], width=6)
            ], justify='center'),
        ], fluid=True)
    ], className='content-area', id='content-product'
    )

    return content


def get_specification_display(fig_position):
    content = html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("CARACTÉRISTIQUES", className='card-title'),
                            fig_position["specifications"],
                        ])
                    )
                ])
            ], justify='center'),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    set_card(fig_position["shares"], "PARTS DISPONIBLES")
                ], width=6),
                dbc.Col([
                    set_card(create_subscription_table(data), "JOURNAL DES SOUSCRIPTIONS & RACHATS")
                ], width=6) if data.get('subscription') is not None else None
            ], justify='center')
        ], fluid=True),
        html.Br()
    ], className='content-area')

    return content


data = load_data()
data = process_data(data)
fig_position = get_element_position(data)
part = 'P'

external_stylesheets = [
    'https://unpkg.com/boxicons@2.1.4/css/boxicons.min.css',
    'https://fonts.googleapis.com/css2?family=Archivo:wght@500&display=swap',
    '/assets/calendar.css',
    '/assets/display.css',
    '/assets/slidebar.css',
    '/assets/table.css',
    '/assets/overview.css',
    dbc.themes.BOOTSTRAP]

server = Flask(__name__)
app = dash.Dash(__name__,
                server=server,
                external_stylesheets=external_stylesheets,
                suppress_callback_exceptions=True)

app.index_string = open('index.html').read()

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='loading-mask', children=[html.Div(className="loader")]),

    html.Div(id='page-content'),
    html.Div([
        html.Img(src='/assets/images/meeschaert_logo.png', className='top-bar-logo'),
        html.Div([
            html.H1([
                set_fund_name(data),
                html.Span(f'APERÇU DU PORTEFEUILLE AU {data["date"]}')  # TODO : VAR the DATE
            ], className='title')
        ], className='header')
    ], className='top-bar'),

    html.Nav([
        html.A('Performance', href='#performance', className='nav-link'),
        html.A('Portefeuille', href='#portefeuille', className='nav-link'),
        html.A('Sensibilités', href='#sensibilites', className='nav-link'),
        html.A('Exposition', href='#exposition', className='nav-link'),
        html.A('Échéancier', href='#echeancier', className='nav-link'),
        html.A('Produits', href='#produits', className='nav-link'),
        html.A('Caractéristique', href='#caracteristiques', className='nav-link')
    ], className='nav'),

    html.Section(id='performance', children=[
        get_performance_display(fig_position, data, part)

    ]),
    html.Section(id='portefeuille', children=[
        get_portfolio_display(fig_position)

    ]),
    html.Section(id='sensibilites', children=[
        get_sensitivity_display(fig_position, data, part)

    ]),
    html.Section(id='exposition', children=[
        get_exposures_display(fig_position)

    ]),
    html.Section(id='echeancier', children=[
        get_schedule_display(fig_position, data)

    ]),
    html.Section(id='produits', children=[
        get_focus_product_display(data)

    ]),
    html.Section(id='caracteristiques', children=[
        get_specification_display(fig_position)

    ])
])


@app.callback(
    Output({'type': 'performance-graph', 'index': MATCH}, 'figure'),
    [Input({'type': 'rebase-date-picker', 'index': MATCH}, 'date')]
)
def update_graph(rebase_date):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    else:
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        triggered_id_dict = eval(triggered_id)

        part = triggered_id_dict['index']

        return get_perf_plot(data['line_perf'][part], part, rebase_date).figure


def read_base64_from_txt(filename):
    path = os.path.join(os.getcwd(), 'data', 'ts', f"{filename}.txt")
    with open(path, 'r') as file:
        base64_content = file.read()
    return base64_content


@server.route('/download_pdf/<path:filename>')
def download_pdf(filename):
    base64_content = read_base64_from_txt(filename)
    pdf_bytes = base64.b64decode(base64_content)
    pdf_buffer = io.BytesIO(pdf_bytes)
    pdf_buffer.seek(0)

    response = make_response(send_file(pdf_buffer, as_attachment=False, mimetype='application/pdf'))
    response.headers['Content-Disposition'] = f'inline; filename="{filename}.pdf"'
    return response


@app.callback(
    Output('download-url', 'children'),
    [Input('position-table', 'active_cell')],
    [State('position-table', 'data')]
)
def update_download_url(active_cell, rows):
    if active_cell and active_cell['column_id'] == 'ISIN':
        isin = rows[active_cell['row']]['ISIN']
        if isin == "":
            return None
        else:
            encoded_filename = urllib.parse.quote(isin)
            return f'/download_pdf/{encoded_filename}'


app.clientside_callback(
    """
    function(url) {
        if (url) {
            window.open(url, '_blank');
            return '';
        }
    }
    """,
    Output('dummy-div', 'children'),
    Input('download-url', 'children'))


@app.callback(
    Output('product_select_box', 'value'),
    Output('content-product', 'children'),
    Input('product_select_box', 'value')
)
def update_product_display(selected_product):
    select_box = get_product_select_box(data['name'], selected_product)
    content = get_focus_product_display(data, selected_product)

    return select_box, content


@app.callback(
    Output("exposures-graph-right", "figure"),
    Input("exposure_select_box_right", "value")
)
def update_pie_chart(selected_input=None):
    selected_input = list(data['exposures'].keys())[0] if selected_input is None else selected_input
    df = data["exposures"][selected_input]
    return get_pie_plot(df)


@app.callback(
    Output("exposures-graph-left", "figure"),
    Input("exposure_select_box_left", "value")
)
def update_pie_chart(selected_input=None):
    selected_input = list(data['exposures'].keys())[1] if selected_input is None else selected_input
    df = data["exposures"][selected_input]
    return get_pie_plot(df)


@app.callback(
    [Output("shock-graph", "figure"),
     Output("slider-container", "children")],
    [Input("sensitivity_select_box", "value"),
     Input("slider-callback", "value")])
def update_shock_chart_and_slider(selected_input, slider_value):
    slider_value = slider_value if slider_value else 0.0
    if selected_input != "greek":

        df = data["shock"][selected_input][["shock", slider_value]]
        figure = get_shock_plot(df).figure

        y_max = data["shock"][selected_input].drop(columns=['shock']).max().max()
        y_min = data["shock"][selected_input].drop(columns=['shock']).min().min()

        figure.update_layout(
            yaxis=dict(range=[1.1 * y_min, 1.1 * y_max], autorange=False)
        )

        slider_range = [value for value in data["shock"][selected_input].columns
                        if isinstance(value, (int, float))]

        slider = get_slider(slider_range, slider_value, "slider-callback")
        return figure, slider

    else:
        df = data["shock"][selected_input]
        figure = get_shock_plot(df).figure
        return figure, None


@app.callback(
    Output('index-table', 'data'),
    [Input('index-table', 'sort_by')]
)
def update_index_table(sort_by):
    df = data["index_global"].copy()

    if sort_by:
        col_id = sort_by[0]['column_id']
        col_sort = f"{col_id}_tmp" if f"{col_id}_tmp" in df.columns else col_id

        df_sorted = df.sort_values(
            by=col_sort,
            ascending=(sort_by[0]['direction'] == 'asc'))

        return df_sorted.reset_index().to_dict('records')
    else:
        return df.reset_index().to_dict('records')


@app.callback(
    Output('worst-best-table', 'data'),
    [Input('worst-best-table', 'sort_by')]
)
def update_worst_best_table(sort_by):
    df = data["worst_best"].copy()

    if sort_by:
        col_id = sort_by[0]['column_id']
        col_sort = f"{col_id}_tmp" if f"{col_id}_tmp" in df.columns else col_id

        df_sorted = df.sort_values(
            by=col_sort,
            ascending=(sort_by[0]['direction'] == 'asc'))

        return df_sorted.to_dict('records')
    else:
        return df.to_dict('records')


@app.callback(
    Output("details-table", 'data'),
    [Input("details-table", 'sort_by')]
)
def update_details_table(sort_by):
    df = data["details"].copy()

    if sort_by:
        col_id = sort_by[0]['column_id']
        col_sort = f"{col_id}_tmp" if f"{col_id}_tmp" in df.columns else col_id

        df_sorted = df.sort_values(
            by=col_sort,
            ascending=(sort_by[0]['direction'] == 'asc'))

        return df_sorted.reset_index().to_dict('records')
    else:
        return df.reset_index().to_dict('records')


@app.callback(
    Output('transparisation-table', 'data'),
    [Input('transparisation-table', 'sort_by'),
     Input('product_select_box', 'value')]
)
def update_transparisation_table(sort_by, selected_product):
    df = data["product"][selected_product]["transparisation"].copy()

    if sort_by:
        col_id = sort_by[0]['column_id']
        col_sort = f"{col_id}_tmp" if f"{col_id}_tmp" in df.columns else col_id

        df_sorted = df.sort_values(
            by=col_sort,
            ascending=(sort_by[0]['direction'] == 'asc'))

        return df_sorted.to_dict('records')
    else:
        return df.to_dict('records')


def create_popover(index, col):
    unique_value = [val for val in list(data["position"][col].unique()) if val != '']
    return dmc.Popover(
        id=f"popover-{index}",
        width=300,
        position="bottom",
        withArrow=True,
        zIndex=999999,
        shadow="md",
        children=[
            dmc.PopoverTarget(html.Button(
                id=f'popover-trigger-button-{index}',
                style={'display': 'none'}
            )),
            dmc.PopoverDropdown(
                dmc.MultiSelect(
                    placeholder="Pick values",
                    data=unique_value,
                    id=f"multi-select-{index}"
                )
            ),
        ],
    )


@app.callback(
    Output('popovers-container', 'children'),
    [Input('popover-state', 'data')]
)
def render_popovers(popover_state):
    return [create_popover(i, col) for i, col in enumerate(['Format', 'Poche', 'Contrepartie', 'Sous-jacent'])]


@app.callback(
    Output('popover-state', 'data'),
    [Input({'type': 'icon-click', 'index': ALL}, 'n_clicks')],
    [State('popover-state', 'data')]
)
def toggle_popover(n_clicks, popover_state):
    ctx = dash.callback_context
    if not ctx.triggered or not n_clicks:
        return popover_state

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    index = eval(triggered_id)['index']
    if popover_state['open'] == index:
        popover_state['open'] = None
    else:
        popover_state['open'] = index
    return popover_state


app.clientside_callback(
    """
    function(data) {
        const columnsWithPopovers = ['Format', 'Poche', 'Contrepartie', 'Sous-jacent'];

        setTimeout(function() {
            console.log("Running setTimeout function");
            const th_elements = document.querySelectorAll('.dash-header');

            let popoverIndex = 0;
            th_elements.forEach((el) => {
                const colName = el.innerText.trim();

                if (columnsWithPopovers.includes(colName) && !el.querySelector('.bx')) {

                    const iconContainer = document.createElement('a');
                    iconContainer.dataset.index = popoverIndex;
                    iconContainer.onclick = (function(index) {
                        return function() {
                            console.log("Icon clicked at index:", index);
                            const event = new CustomEvent('icon-click', { detail: { index: index } });
                            window.dispatchEvent(event);
                        };
                    })(popoverIndex);

                    const icon = document.createElement('i');
                    icon.className = `bx bx-dots-vertical-rounded custom-icon-${popoverIndex}`;

                    iconContainer.appendChild(icon);
                    el.appendChild(iconContainer);

                    popoverIndex++;
                }
            });

            window.addEventListener('icon-click', function(event) {
                const index = event.detail.index;
                const triggerButton = document.getElementById(`popover-trigger-button-${index}`);
                if (triggerButton) {
                    triggerButton.click();
                } else {
                    console.log("Trigger button not found for index:", index);
                }
            });

            // Insert popovers into th elements
            const popoversContainer = document.getElementById('popovers-container');

            popoversContainer.childNodes.forEach(popover => {
                const index = parseInt(popover.id.split('-')[1]);
                const targetTh = document.querySelector(`.dash-header.column-${index}`);
                if (targetTh && targetTh.children.length === 0) {
                    targetTh.appendChild(popover);
                }
            });
        }, 500);
        return '';
    }
    """,
    Output('icon-click-listener', 'children'),
    [Input('position-table', 'data')]
)


@app.callback(
    Output('position-table', 'data'),
    [Input('position-table', 'sort_by')]
)
def update_position_table(sort_by):
    df = data["position"].copy()

    if sort_by:
        col_id = sort_by[0]['column_id']
        col_sort = f"{col_id}_tmp" if f"{col_id}_tmp" in df.columns else col_id

        isin_blank = df[df['ISIN'] == '']
        df = df[df['ISIN'] != '']

        df_sorted = df.sort_values(
            by=col_sort,
            ascending=(sort_by[0]['direction'] == 'asc')
        )

        df_sorted = pd.concat([df_sorted, isin_blank])

        return df_sorted.to_dict('records')
    else:
        return df.to_dict('records')


# @app.callback(
#     Output('position-table', 'data'),
#     [Input('multi-select-1', 'value')]
# )
# def update_position_table_1(selected_values):
#     print(selected_values)
#     df = data["position"].copy()
#     if selected_values:
#         df = df[df['Format'].isin(selected_values)]
#     return df.to_dict('records')


@app.callback(
    [Output('flow-table', 'data'),
     Output("schedule-graph", "figure"),
     Output("flow-slider-container", "children")],
    [Input("flow-slider-callback", "value"),
     Input('flow-table', 'sort_by')]
)
def update_schedule_chart_slider_flow_table(slider_value, sort_by):
    slider_value = slider_value if slider_value else "0"

    df_slider = data["flow"][str(slider_value)].copy()
    figure = get_schedule_bar_plot(df_slider).figure

    slider = get_slider(data["flow"], float(slider_value), "flow-slider-callback")
    slider_div = [html.H4("Impact d'un choc sur les prix spot", className='slider-title'), slider]

    df_table = data["upcoming_flows"][str(slider_value)].copy()

    if sort_by:
        col_id = sort_by[0]['column_id']
        col_sort = f"{col_id}_tmp" if f"{col_id}_tmp" in df_table.columns else col_id

        df_sorted = df_table.sort_values(
            by=col_sort,
            ascending=(sort_by[0]['direction'] == 'asc'))

        df_table_data = df_sorted.to_dict('records')
    else:
        df_table_data = df_table.to_dict('records')

    return df_table_data, figure, slider_div


if __name__ == '__main__':
    app.run_server(debug=True, port=8020)

# %%
