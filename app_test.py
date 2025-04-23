import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Main app
st.set_page_config(page_title="Bid Analyzer Pro", layout="wide")
st.title("🏆 Bid Success & Market Analysis")
# Load models
@st.cache_resource
def load_models():
    return {
        'Win Probability': pickle.load(open('rf_model_probab.pkl', 'rb')),
        'CPI Random Forest': pickle.load(open('rf_pred.pkl', 'rb')),
        'CPI Linear Regression': pickle.load(open('lr_pred.pkl', 'rb')),
        'CPI Decision Tree': pickle.load(open('dt_pred.pkl', 'rb'))
    }

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_excel('merged_data.xlsx')
    won_df = df[df['Status'] == 'Invoiced']
    return won_df[['LOI', 'Complete', 'IR', 'CPI']]

models = load_models()
won_data = load_data()


# Input Section
with st.form("bid_inputs"):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        loi = st.number_input("LOI (Minutes)", 0.0, 40.0, 15.0, 0.1)
    with col2:
        complete = st.number_input("Completion", 0.0, 3000.0, 480.0, 0.1)
    with col3:
        ir = st.number_input("Incident Rate", 0.0, 100.0, 36.0, 0.1)
    with col4:
        cpi = st.number_input("Market CPI", 0.0, 92.0, 17.0, 0.1)
    
    submitted = st.form_submit_button("Analyze Bid")

if submitted:
    # Win Probability Prediction
    win_prob = models['Win Probability'].predict_proba(
        np.array([[complete, cpi, ir, loi]])
    )[0][1]
    
    # CPI Predictions
    cpi_input = np.array([[loi, complete, ir]])
    cpi_preds = {
        'Random Forest': models['CPI Random Forest'].predict(cpi_input)[0],
        'Linear Regression': models['CPI Linear Regression'].predict(cpi_input)[0],
        'Decision Tree': models['CPI Decision Tree'].predict(cpi_input)[0]
    }
    
    # Historical Analysis
    historical_avg = won_data['CPI'].mean()
    similar_bids = won_data[
        (won_data['LOI'].between(loi-5, loi+5)) &
        (won_data['Complete'].between(complete-100, complete+100)) &
        (won_data['IR'].between(ir-10, ir+10))
    ]
    
    # Visualization
    st.subheader("Analysis Results")
    
    # Create columns layout
    col_left, col_center, col_right = st.columns([1,2,1])
    
    with col_left:
        st.metric("Win Probability", f"{win_prob*100:.1f}%")
        st.caption(f"LOI: {loi:.1f} mins")
        st.caption(f"Completion: {complete:.1f}%")
        st.caption(f"Interviews: {ir:.1f}/hr")
        st.caption(f"Current CPI: {cpi:.1f}")
        
    with col_center:
        # CPI Comparison Chart
        fig, ax = plt.subplots(figsize=(8,4))
        
        # Model predictions
        models = list(cpi_preds.keys())
        values = list(cpi_preds.values())
        
        bars = ax.bar(models, values, color=['#4C72B0', '#DD8452', '#55A868'])
        ax.axhline(historical_avg, color='#C44E52', linestyle='--', 
                  label=f'Historical Avg: {historical_avg:.1f}')
        
        # Annotations
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height+0.5,
                    f'{height:.1f}',
                    ha='center', va='bottom')
        
        ax.set_ylabel('Predicted CPI')
        ax.set_title('Market CPI Predictions vs Historical Average')
        ax.legend()
        st.pyplot(fig)
        
    with col_right:
        # Historical Context
        st.subheader("Historical Context")
        if not similar_bids.empty:
            st.write(f"Found {len(similar_bids)} similar historical bids:")
            st.dataframe(similar_bids.head(3), hide_index=True)
        else:
            st.warning("No similar historical bids found")
            
        st.metric("Historical Average CPI", 
                 f"{historical_avg:.1f}",
                 delta="From won bids")
