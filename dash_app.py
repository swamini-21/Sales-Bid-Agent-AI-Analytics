#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# In[ ]:


# Read the Excel file
df = pd.read_excel('binned_new.xlsx')
# df.info()


# In[ ]:


features = ['IR_range','Complete_range','LOI_range']


# In[6]:


client_segments = df['Client Segment Type'].dropna().unique().tolist()
client_segments_options = [{'label': seg, 'value': seg} for seg in client_segments]
client_segments_options.insert(0, {'label': 'All', 'value': 'All'})


# In[7]:


# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Predictive Pricing Model"


# In[8]:


def data_filter(selected_segment):
    if selected_segment != 'All':
        filtered_df = df[df['Client Segment Type'] == selected_segment]
    else:
        filtered_df = df.copy()

    win_df = filtered_df[filtered_df['Status'] == 'Invoiced']
    lost_df = filtered_df[filtered_df['Status'] == 'Lost']
    return filtered_df, win_df, lost_df


# In[9]:


filtered_df, win_df, lost_df = data_filter('All')
total_revenue = win_df['Actual Project Revenue'].sum()
win_cpi = win_df['CPI'].mean()
lost_revenue = lost_df['Actual Project Revenue'].sum()
lost_cpi = lost_df['CPI'].mean()


# ### Dashboard Layout

# In[ ]:


app.layout = html.Div([
    # ─── Fixed Sidebar ──────────────────────────────────────────────────────────
    html.Div([
        html.H3("Predictive Pricing Model", style={'textAlign':'center'}),

        # KPI cards
        html.Div([
             html.Div([
                html.H4("Annual Revenue", style={"margin": 0, "color": "gray", "text-align": "center", "font-weight": "bold"}),
                html.H5(f"${total_revenue:,.2f}", 
                    style={"border": "1px solid #ccc", "margin":0,"border-radius": "15px", "padding": "10px", "text-align": "center", "color": "#6CA36C", 
                    "font-size": "24px", "box-shadow": "0px 2px 10px rgba(0, 0, 0, 0.1)"}),
                ], className="kpi-card"),

            html.Div([
                html.H4("Lost Revenue", style={"margin": 0, "color": "gray", "text-align": "center", "font-weight": "bold"}),
                html.H5(f"${lost_revenue:,.2f}", 
                    style={"border": "1px solid #ccc", "margin":0,"border-radius": "15px", "padding": "10px", "text-align": "center", "color": "#C44E52", 
                    "font-size": "24px", "box-shadow": "0px 2px 10px rgba(0, 0, 0, 0.1)"})
            ], className="kpi-card"),

            html.Div([
                html.H4("Average Win CPI", style={"margin": 0, "color": "gray", "text-align": "center", "font-weight": "bold"}),
                html.H5(f"{win_cpi:,.2f}", 
                    style={"border": "1px solid #ccc", "margin":0,"border-radius": "15px", "padding": "10px", "text-align": "center", "color": "#6CA36C", 
                    "font-size": "24px", "box-shadow": "0px 2px 10px rgba(0, 0, 0, 0.1)"})
            ], className="kpi-card"),

            html.Div([
                html.H4("Average Lost CPI", style={"margin": 0, "color": "gray", "text-align": "center", "font-weight": "bold"}),
                html.H5(f"{lost_cpi:,.2f}", 
                    style={"border": "1px solid #ccc", "margin":0,"border-radius": "15px", "padding": "10px", "text-align": "center", "color": "#C44E52", 
                    "font-size": "24px", "box-shadow": "0px 2px 10px rgba(0, 0, 0, 0.1)"})
            ], className="kpi-card"),
        ]),

        # Filters
        html.Div([
            html.Label("Select Feature:"),
            dcc.Dropdown(
                id='feature-dropdown',
                options=[{'label': f, 'value': f} for f in features],
                value=features[0],
                clearable=False,
                style={'marginBottom':'20px'}
            ),

            html.Label("Select Segment Type:"),
            dcc.Dropdown(
                id='segment-dropdown',
                options=client_segments_options,
                value='All',
                clearable=False
            ),
        ], style={'marginTop':'30px'})
    ], style={
        'position': 'fixed',
        'top': 0,
        'left': 0,
        'bottom': 0,
        'width': '12%',
        'padding': '0.25%',
        'overflowY': 'auto',
        'backgroundColor': '#f8f9fa',
        'boxShadow': '2px 0 5px rgba(0,0,0,0.1)'
    }),

    # ─── Main Content ────────────────────────────────────────────────────────────
    html.Div([
        dcc.Graph(id='scatter-plot'),
        html.Div([
            html.Div([dcc.Graph(id='sunburst')],className = 'accntRev', style={'display':'inline-block', 'width':'45%','margin':'0px'}),
            html.Div([dcc.Graph(id='bar-chart-lost')],className = 'accntRev', style={'display':'inline-block', 'width':'45%','margin':'0px'})

        ]),
    ], style={
        'marginLeft': '15%',  
        'padding': '0px'
    })
])


# ### Scatter Plots

# In[13]:


def feature_sel(sf):
    if sf == 'IR_range':
        return 'IR'
    elif sf == 'Complete_range':
        return 'Complete'
    elif sf == 'LOI_range':
        return 'LOI'


# In[14]:


@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('feature-dropdown', 'value'), Input('segment-dropdown', 'value')]
)
def update_line_graph(selected_feature, selected_segment):
    # Assume data_filter returns three DataFrames: filtered_df, df_invoiced, and df_lost.
    filtered_df, df_invoiced, df_lost = data_filter(selected_segment)

    # Sort each subset by the selected feature
    df_invoiced = df_invoiced.sort_values(by=selected_feature)
    df_lost = df_lost.sort_values(by=selected_feature)

    # Define a mapping dictionary for your CPI range categories.
    # Adjust the keys (categories) and color values as needed.
    cpi_color_map = {
        "Very Low": "#6B8EAE",
        "Low": "#6CA36C",
        "Moderate": "#E1C16E",
        "High": "#D18F5F",
        "Very High": "#C44E52"
    }

    # Map the colors based on the CPI range column for each subset
    df_invoiced['color'] = df_invoiced['CPI_range'].map(cpi_color_map)
    df_lost['color'] = df_lost['CPI_range'].map(cpi_color_map)

    # Create a subplot figure with two columns (one for Invoiced and one for Lost deals)
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Invoiced Deals", "Lost Deals")
    )


    # Add trace for Invoiced deals (using the mapped colors for each marker)
    feature = feature_sel(selected_feature)
    customdata_in = df_invoiced[['CPI', 'CPI_range']]
    customdata_in[selected_feature] = df_invoiced[selected_feature] 
    customdata_in[feature] = df_invoiced[feature]

    fig.add_trace(
        go.Scatter(
            x=df_invoiced[selected_feature],
            y=df_invoiced['CPI'],  
            mode='markers',
            marker={'symbol': 'circle-open', 'color': df_invoiced['color']},
            name="Invoiced",
            showlegend=False,
            customdata=customdata_in.values,
            hovertemplate=(
                f'<b>{selected_feature}:</b> %{{customdata[2]}}<br>' +
                f'<b>{feature}:</b> %{{customdata[3]}}<br>' +
                '<b>CPI:</b> %{customdata[0]}<br>' +
                '<b>CPI Range:</b> %{customdata[1]}<br>' 
            )
        ),
        row=1, col=1
    )

    # Add trace for Lost deals (using the mapped colors for each marker)
    customdata_l = df_lost[['CPI', 'CPI_range']]
    customdata_l[selected_feature] = df_lost[selected_feature] 
    customdata_l[feature] = df_lost[feature]

    fig.add_trace(
        go.Scatter(
            x=df_lost[selected_feature],
            y=df_lost['CPI'],
            mode='markers',
            marker={'symbol': 'circle-open', 'color': df_lost['color']},
            name="Lost",
            showlegend=False,
            customdata=customdata_l.values,
            hovertemplate=(
                f'<b>{selected_feature}:</b> %{{customdata[2]}}<br>' +
                f'<b>{feature}:</b> %{{customdata[3]}}<br>' +
                '<b>CPI:</b> %{customdata[0]}<br>' +
                '<b>CPI Range:</b> %{customdata[1]}<br>' 
            )
        ),
        row=1, col=2
    )
    for cpi_category, color in cpi_color_map.items():
        fig.add_trace(
            go.Scatter(
                x=[None],  # Empty value to make it invisible
                y=[None],  # Empty value to make it invisible
                mode='markers',
                marker={'symbol': 'circle-open', 'color': color},
                name=cpi_category,  # Set the name for the CPI range in the legend
                showlegend=True,  # Ensure it shows in the legend
                legendgroup=cpi_category  # Group the legend by the CPI category
            ),
            row=1, col=1  # Add it to both subplots
        )


    # y_min = min(df_invoiced['CPI'].min()-5, df_lost['CPI'].min()-5)
    # y_max = max(df_invoiced['CPI'].max()+5, df_lost['CPI'].max()+5)
    # Update layout and axis titles.
    fig.update_layout(
        xaxis=dict(categoryorder='array', categoryarray=['Very Low', 'Low', 'Moderate', 'High', 'Very High']),
        xaxis2=dict(categoryorder='array',  categoryarray=['Very Low', 'Low', 'Moderate', 'High', 'Very High']),
        # yaxis=dict(range=[y_min, y_max]),  # Set y-axis range for first subplot
        # yaxis2=dict(range=[y_min, y_max]),  # Set y-axis range for second subplot
        title_text=f'CPI vs {selected_feature} for Invoiced and Lost Deals',title_x=0.5,title_y=0.98,  title_xanchor='center',title_yanchor='top',
        margin=dict(t=50, b=0,l=0,r=0), height=350
    )

    fig.update_xaxes(title_text=selected_feature, row=1, col=1)
    fig.update_xaxes(title_text=selected_feature, row=1, col=2)
    fig.update_yaxes(title_text="CPI", row=1, col=1)
    fig.update_yaxes(title_text="CPI", row=1, col=2)

    return fig


# In[15]:


@app.callback(
    Output('sunburst', 'figure'),
    Input('segment-dropdown', 'value')
)
def create_radar_chart(selected_segment):
    # Filter data for the selected segment
    filtered_df, df_invoiced, df_lost = data_filter(selected_segment)

    # Define the custom order for 'CPI_range'
    cpi_order = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']

    # Create a dictionary to convert CPI_range into an ordered category
    cpi_order_dict = {cpi_range: index for index, cpi_range in enumerate(cpi_order)}

    # Map the 'CPI_range' to numerical values based on the custom order
    filtered_df['CPI_range_order'] = filtered_df['CPI_range'].map(cpi_order_dict)

    # Sort the DataFrame based on the custom 'CPI_range_order'
    filtered_df = filtered_df.sort_values(by='CPI_range_order')

    # Create the sunburst chart
    fig = px.sunburst(
        filtered_df,
        path=['CPI_range', 'Status', 'Category'],  # Ensure hierarchy is respected
        title='CPI Range Breakdown by Client and Status',
        color='CPI_range',
        color_discrete_map={
        "Very Low": "#6B8EAE",
        "Low": "#6CA36C",
        "Moderate": "#E1C16E",
        "High": "#D18F5F",
        "Very High": "#C44E52"
        }
    )

    # Update layout to ensure the chart title and other properties are set properly
    fig.update_layout(
        title='CPI Range Breakdown by Client and Status',title_x=0.5,title_y=0.98,margin=dict(t=60, b=0,l=0,r=0), height=350
    )

    # Optional: Update hover information and make adjustments as needed
    fig.update_traces(
        hoverinfo="label+percent entry+value"
    )

    return fig


# In[16]:


@app.callback(
    Output('bar-chart-lost', 'figure'),
   Input('segment-dropdown', 'value')
)

def update_bar_lost(selected_segment):
    filtered_df, df_invoiced, df_lost = data_filter(selected_segment)

    # Step 1: Get top 10 clients by lost deals
    lost_deals_by_client = df_lost.groupby('Client Name').size().reset_index(name='Lost Deal Count')
    top_clients_df = lost_deals_by_client.sort_values(by='Lost Deal Count', ascending=False).head(10)
    top_clients = top_clients_df['Client Name']

    # Step 2: Filter combined data
    df_combined = pd.concat([df_invoiced, df_lost])
    df_top_clients = df_combined[df_combined['Client Name'].isin(top_clients)]

    # Optional: join to get Lost Deal Count for ordering
    df_top_clients = df_top_clients.merge(top_clients_df, on='Client Name')

    # Step 3: Group by Client and Status to get deal counts and avg CPI
    summary = (
        df_top_clients.groupby(['Client Name', 'Status'])
        .agg(
            Deal_Count=('CPI', 'count'),
            Avg_CPI=('CPI', 'mean')
        )
        .reset_index()
    )
    summary['Avg_CPI'] = summary['Avg_CPI'].round(2)

    # Step 4: Apply custom sort order to Client Name
    client_order = top_clients_df['Client Name']
    summary['Client Name'] = pd.Categorical(summary['Client Name'], categories=client_order, ordered=True)
    summary = summary.sort_values('Client Name')

    # Step 5: Create grouped bar chart
    fig = px.bar(
        summary,
        x='Client Name',
        y='Deal_Count',
        color='Status',
        hover_data=['Avg_CPI'],
        color_discrete_map={
            'Lost': '#C44E52',     # Previously used for Invoiced
            'Invoiced': '#6CA36C'          # Previously used for Lost
        },
        title='Top 10 Accounts by Lost Deals — Grouped by Deal Status',
        labels={'Deal Count': 'Number of Deals'},
        template='plotly_white'
    )
    fig.update_layout(xaxis_tickangle=-45, barmode='group',title_x=0.5,title_y=0.98,margin=dict(t=60, b=0,l=0,r=0), height=350)

    return fig


# In[17]:


if __name__ == '__main__':
    app.run(debug=True, port=8050, use_reloader=False)  # Set use_reloader=False to avoid double callbacks in debug mode


# In[ ]:




