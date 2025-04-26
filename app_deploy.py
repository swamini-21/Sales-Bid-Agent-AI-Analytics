import streamlit as st
import subprocess, threading
import streamlit.components.v1 as components
import pickle
import numpy as np
import pandas as pd
import requests
import google.generativeai as genai 
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Page configuration
st.set_page_config(page_title="Predictive Pricing Model", layout="wide")
st.title("🏆 Bid Analyzer")

# Custom CSS to make iframe full screen
st.markdown(
    """
    <style>
    .full-screen-iframe {
        width: 100%;
        height: 100vh;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def start_dash():
    subprocess.run(["python", "dash_app.py"])

threading.Thread(target=start_dash, daemon=True).start()

tab_dash, tab_ml  = st.tabs([ "📊 Dashboard", "🧠 ML Model"])

with tab_dash:
    st.header("Interactive Dashboard")

    # Load your data
    @st.cache_data
    def load_dashboard_data():
        return pd.read_excel('binned_new.xlsx')
    df = load_dashboard_data()

    # Features and segment options
    features = ['IR_range', 'Complete_range', 'LOI_range']
    client_segments = df['Client Segment Type'].dropna().unique().tolist()
    client_segments_options = ['All'] + client_segments

    # Helper for feature selection
    def feature_sel(sf):
        if sf == 'IR_range':
            return 'IR'
        elif sf == 'Complete_range':
            return 'Complete'
        elif sf == 'LOI_range':
            return 'LOI'

    # Data filter
    def data_filter(selected_segment):
        if selected_segment != 'All':
            filtered_df = df[df['Client Segment Type'] == selected_segment]
        else:
            filtered_df = df.copy()
        win_df = filtered_df[filtered_df['Status'] == 'Invoiced']
        lost_df = filtered_df[filtered_df['Status'] == 'Lost']
        return filtered_df, win_df, lost_df

    # KPI Cards
    filtered_df, win_df, lost_df = data_filter('All')
    total_revenue = win_df['Actual Project Revenue'].sum()
    win_cpi = win_df['CPI'].mean()
    lost_revenue = lost_df['Actual Project Revenue'].sum()
    lost_cpi = lost_df['CPI'].mean()

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Annual Revenue", f"${total_revenue:,.2f}")
    kpi2.metric("Lost Revenue", f"${lost_revenue:,.2f}")
    kpi3.metric("Average Win CPI", f"{win_cpi:,.2f}")
    kpi4.metric("Average Lost CPI", f"{lost_cpi:,.2f}")

    # Filters
    st.markdown("### Filters")
    col1, col2 = st.columns(2)
    with col1:
        selected_feature = st.selectbox("Select Feature", features)
    with col2:
        selected_segment = st.selectbox("Select Segment Type", client_segments_options)

    # --- Scatter Plot (Subplots) ---
    filtered_df, df_invoiced, df_lost = data_filter(selected_segment)
    df_invoiced = df_invoiced.sort_values(by=selected_feature)
    df_lost = df_lost.sort_values(by=selected_feature)

    cpi_color_map = {
        "Very Low": "#6B8EAE",
        "Low": "#6CA36C",
        "Moderate": "#E1C16E",
        "High": "#D18F5F",
        "Very High": "#C44E52"
    }

    df_invoiced['color'] = df_invoiced['CPI_range'].map(cpi_color_map)
    df_lost['color'] = df_lost['CPI_range'].map(cpi_color_map)

    feature = feature_sel(selected_feature)

    def get_customdata(df):
        customdata = df[['CPI', 'CPI_range']].copy()
        customdata[selected_feature] = df[selected_feature]
        customdata[feature] = df[feature]
        return customdata.values

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Invoiced Deals", "Lost Deals")
    )
    fig.add_trace(
        go.Scatter(
            x=df_invoiced[selected_feature],
            y=df_invoiced['CPI'],
            mode='markers',
            marker={'symbol': 'circle-open', 'color': df_invoiced['color']},
            name="Invoiced",
            showlegend=False,
            customdata=get_customdata(df_invoiced),
            hovertemplate=(
                f'<b>{selected_feature}:</b> %{{customdata[2]}}<br>' +
                f'<b>{feature}:</b> %{{customdata[3]}}<br>' +
                '<b>CPI:</b> %{customdata[0]}<br>' +
                '<b>CPI Range:</b> %{customdata[1]}<br>'
            )
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df_lost[selected_feature],
            y=df_lost['CPI'],
            mode='markers',
            marker={'symbol': 'circle-open', 'color': df_lost['color']},
            name="Lost",
            showlegend=False,
            customdata=get_customdata(df_lost),
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
                x=[None], y=[None], mode='markers',
                marker={'symbol': 'circle-open', 'color': color},
                name=cpi_category, showlegend=True, legendgroup=cpi_category
            ),
row=1, col=2
    )
    for cpi_category, color in cpi_color_map.items():
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode='markers',
                marker={'symbol': 'circle-open', 'color': color},
                name=cpi_category, showlegend=True, legendgroup=cpi_category
            ),
            row=1, col=1
        )
    fig.update_layout(
        xaxis=dict(categoryorder='array', categoryarray=list(cpi_color_map.keys())),
        xaxis2=dict(categoryorder='array', categoryarray=list(cpi_color_map.keys())),
        title_text=f'CPI vs {selected_feature} for Invoiced and Lost Deals', # Fixed the quote here
        title_x=0.5, 
        height=350,
        margin=dict(t=50, b=0, l=0, r=0)
    )
    fig.update_xaxes(title_text=selected_feature, row=1, col=1)
    fig.update_xaxes(title_text=selected_feature, row=1, col=2)
    fig.update_yaxes(title_text="CPI", row=1, col=1)
    fig.update_yaxes(title_text="CPI", row=1, col=2)
    st.plotly_chart(fig, use_container_width=True)

    # --- Sunburst Chart ---
    cpi_order = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
    cpi_order_dict = {cpi_range: idx for idx, cpi_range in enumerate(cpi_order)}
    filtered_df['CPI_range_order'] = filtered_df['CPI_range'].map(cpi_order_dict)
    filtered_df_sorted = filtered_df.sort_values(by='CPI_range_order')
    fig_sun = px.sunburst(
        filtered_df_sorted,
        path=['CPI_range', 'Status', 'Category'],
        title='CPI Range Breakdown by Client and Status',
        color='CPI_range',
        color_discrete_map=cpi_color_map
    )
    fig_sun.update_layout(
        title_x=0.5, margin=dict(t=60, b=0, l=0, r=0), height=350
    )
    st.plotly_chart(fig_sun, use_container_width=True)

    # --- Bar Chart: Top 10 Accounts by Lost Deals ---
    lost_deals_by_client = df_lost.groupby('Client Name').size().reset_index(name='Lost Deal Count')
    top_clients_df = lost_deals_by_client.sort_values(by='Lost Deal Count', ascending=False).head(10)
    top_clients = top_clients_df['Client Name']
    df_combined = pd.concat([df_invoiced, df_lost])
    df_top_clients = df_combined[df_combined['Client Name'].isin(top_clients)]
    df_top_clients = df_top_clients.merge(top_clients_df, on='Client Name')
    summary = (
        df_top_clients.groupby(['Client Name', 'Status'])
        .agg(
            Deal_Count=('CPI', 'count'),
            Avg_CPI=('CPI', 'mean')
        )
        .reset_index()
    )
    summary['Avg_CPI'] = summary['Avg_CPI'].round(2)
    client_order = top_clients_df['Client Name']
    summary['Client Name'] = pd.Categorical(summary['Client Name'], categories=client_order, ordered=True)
    summary = summary.sort_values('Client Name')
    fig_bar = px.bar(
        summary,
        x='Client Name',
        y='Deal_Count',
        color='Status',
        hover_data=['Avg_CPI'],
        color_discrete_map={'Lost': '#C44E52', 'Invoiced': '#6CA36C'},
        title='Top 10 Accounts by Lost Deals - Grouped by Deal Status',
        labels={'Deal Count': 'Number of Deals'},
        template='plotly_white'
    )
    fig_bar.update_layout(
        xaxis_tickangle=-45, barmode='group', title_x=0.5, margin=dict(t=60, b=0, l=0, r=0), height=350
    )
    st.plotly_chart(fig_bar, use_container_width=True)


with tab_ml:
    st.header("Machine Learning Model")
    # Load model
    @st.cache_resource  # Cache the model loading
    def load_model():
        try:
            with open('rf_model_probab.pkl', 'rb') as file:
                model = pickle.load(file)
            return model
        except FileNotFoundError:
            st.error("Model file not found. Please ensure 'rf_model_probab.pkl' exists in the project directory.")
            return None

        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None
    model = load_model()


    @st.cache_data  # Cache the data loading
    def load_excel_data():
        try:
            # Read the Excel file
            df = pd.read_excel('merged_data.xlsx')
            return df
        except FileNotFoundError:
            st.error("Excel file not found. Please ensure 'merged_data.xlsx' exists in the project directory.")
            return None
        except Exception as e:
            st.error(f"Error loading Excel file: {str(e)}")
            return None
        
    df = load_excel_data()  
    analyzer = (df[['LOI', 'Complete', 'IR', 'CPI', 'Status']]).to_json(orient='records')
    # Input parameters with float validation
    with st.form("bid_inputs"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            loi = st.number_input("LOI (Minutes)", 
                                min_value=0.0, max_value=40.0, 
                                value=15.0, step=0.1,
                                format="%.1f")
        with col2:
            complete = st.number_input("Complete", 
                                    min_value=0.0, max_value=3000.0,
                                    value=480.0, step=0.1,
                                    format="%.1f")
        with col3:
            ir = st.number_input("Incident Rate", 
                            min_value=0.0, max_value=100.0,
                            value=36.0, step=0.1,
                            format="%.1f")
        with col4:
            cpi = st.number_input("Market CPI", 
                                min_value=0.0, max_value=92.0,
                                value=17.0, step=0.1,
                                format="%.1f")
        
        submitted = st.form_submit_button("Calculate Win Probability")

    if submitted:
        # Create input array with validated feature order 
        input_data = np.array([[
        float(complete),
        float(cpi),
        float(ir),
        float(loi)]]).astype(np.float64) 
        
        try:
            # Validate model compatibility
            if not hasattr(model, 'predict_proba'):
                raise AttributeError("Loaded model doesn't support probability predictions")
            
            # Get probability prediction
            win_probability = model.predict_proba(input_data)[0][1]
            
            # Visualization
            st.subheader("Bid Prediction Analysis")
            
            # Create three-column layout
            left, center, right = st.columns([1,2,1])
            
            with left:
                st.metric("Win Probability", f"{win_probability*100:.1f}%")
                st.caption(f"LOI: {loi:.1f}")
                st.caption(f"Complete: {complete:.1f}")
                st.caption(f"Incident Rate: {ir:.1f}")
                st.caption(f"CPI: {cpi:.1f}")
            
            with center:
                # Probability gauge
                gauge_html = f"""
                <div style="text-align:center;">
                    <svg width="250" height="150">
                        <circle cx="125" cy="125" r="90" fill="none" stroke="#eee" stroke-width="12"/>
                        <path d="M35,125 A90,90 0 0,1 215,125" 
                            stroke="url(#gradient)" 
                            stroke-width="12" 
                            fill="none"
                            stroke-dasharray="{win_probability*565} 565"/>
                        <text x="50%" y="60%" font-size="30" text-anchor="middle">
                            {win_probability*100:.1f}%
                        </text>
                        <defs>
                            <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                                <stop offset="0%" style="stop-color:#ff4444"/>
                                <stop offset="100%" style="stop-color:#44ff44"/>
                            </linearGradient>
                        </defs>
                    </svg>
                </div>
                """
                st.markdown(gauge_html, unsafe_allow_html=True)
                
                # Recommendation
                if win_probability >= 0.7:
                    st.success("## 🚀 Strong Bid Recommendation: Proceed!")
                elif win_probability >= 0.5:
                    st.warning("## ⚖️ Competitive Bid: Consider Optimizations")
                else:
                    st.error("## 🛑 High Risk: Re-evaluate Bid Strategy")
            

            # LLM Strategy Insights
            st.subheader("AI Recommendations")
            with right:
                if "GEMINI_API_KEY" not in st.secrets:
                    st.error("GEMINI_API_KEY not found in secrets")
                    st.stop()
            
            try:
                # Configure the client
                genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                model = genai.GenerativeModel('gemini-2.0-flash')

                prompt = f""" You are a bid optimization consultant.
                    Utilize and thoroughly analyze historical bid data from {analyzer} to provide data-driven recommendations.

                    Data Reference
                    Project #: Unique project/bid identifier (Column B)

                    Project Name: Name of the project (Column C)

                    Invoiced Date: Official project completion date (Column G)

                    Actual Project Revenue: Total value of the project (Column H)

                    Client Name: Name of the client (Column J)

                    Complete: Number of completed units delivered (Column L)

                    CPI: Cost per Interview (Column M)

                    CPI = Price per complete; Actual Project Revenue = CPI × Completes

                    Actual IR: Incidence Rate (Column N)

                    Major driver of CPI

                    Actual LOI: Length of Interview (Column O)

                    Secondary driver of CPI

                    Countries: Project geography (Column Q)

                    Audience: Target sub-population (Column R)

                    Ignore all other columns.

                    Your Task
                    Statistical Analysis

                    For both won and lost bids, calculate minimum, maximum, mean, and standard deviation for:

                    LOI (Length of Interview), Complete, IR (Incidence Rate), CPI (Cost per Interview).

                    Bid Comparison

                    For a new bid (with values for LOI {loi}, Complete {complete}, IR {ir}, CPI {cpi}, and predicted win probability {win_probability}), compare each input to historical ranges.

                    Outlier Detection

                    Identify if any input is an outlier (outside 5th–95th percentile or >1.5 standard deviations from the mean of won bids).

                    Critical Factor Identification

                    Highlight the 3 features most limiting win probability, based on deviation from successful bids.

                    Actionable Recommendations

                    Suggest 3 prioritized, actionable steps to improve win probability, with statistically-derived target ranges.

                    High Probability Handling

                    If win probability >80%, provide positive reinforcement and flag any metrics near risk thresholds, with monitoring/mitigation suggestions.

                    Output Format
                    ❗ Critical Factors

                    Briefly list the 3 most significant limiting features, with context (e.g., “LOI is 23, above 95th percentile for wins [max: 18]”).

                    🛠️ Recommended Actions

                    3 prioritized, actionable steps, each with a target range or value.

                    📈 Expected Outcome

                    Quantify the potential improvement in win probability, based on historical data.

                    💡 Business Summary

                    Concise summary of the bid’s position and key next steps to maximize win rate.

                    Example Output
                    ❗ Critical Factors

                    Interview Length (LOI) is 23, above the 95th percentile for won bids (max recommended: 18).

                    Completes are 40, below the mean for won bids (recommended: >65).

                    CPI is 28, exceeding the optimal range for wins (recommended: 8–18).

                    🛠️ Recommended Actions

                    Reduce LOI to under 18 minutes.

                    Increase project completes to at least 65 before submitting the bid.

                    Adjust pricing to bring CPI into the 8–18 range.

                    📈 Expected Outcome

                    Implementing these changes is associated with a 30–45% increase in win probability.

                    💡 Business Summary

                    Your bid currently falls outside optimal ranges on key factors. Aligning with successful bid metrics can significantly boost your win rate. Monitor CPI and completes closely, as they are strong predictors of success.

                    Instructions:

                    Do not include metrics in column headers—treat all columns as numeric values.

                    Base all recommendations and analyses strictly on the data provided. (Do not provide code)"""
                # Generate content
                response = model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.3,
                        # "max_output_tokens": 800
                    }
                )
                
                if response:
                    st.markdown(response.text)
                else:
                    st.error("No response generated from the API")
                    
            except Exception as e:
                st.error(f"API Error: {str(e)}")
                    
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

# st.markdown("### Full Dash dashboard below:")
# components.iframe("http://localhost:8050", height=700, scrolling=True)