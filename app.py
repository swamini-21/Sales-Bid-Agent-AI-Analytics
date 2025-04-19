# # import streamlit as st
# # import pickle
# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import requests

# # # Load ML models
# # models = {
# #     'Model 1': pickle.load(open('best_model.pkl', 'rb')),
# #     'Model 2': pickle.load(open('best_model.pkl', 'rb')),
# #     'Model 3': pickle.load(open('best_model.pkl', 'rb'))
# # }

# # # Page configuration
# # st.set_page_config(page_title="CPI Predictor", layout="wide")
# # st.title("AI-Powered CPI Prediction System")

# # # Input section
# # with st.form("input_form"):
# #     col1, col2, col3 = st.columns(3)
# #     with col1:
# #         loi = st.number_input("LOI", min_value=0.0, max_value=100.0, step=0.1)
# #     with col2:
# #         complete = st.number_input("COMPLETE Index", min_value=0.0, step=0.1)
# #     with col3:
# #         ir = st.number_input("IR Rate", min_value=0.0, step=0.1)
    
# #     submitted = st.form_submit_button("Predict CPI")

# # if submitted:
# #     # Create input array
# #     input_data = np.array([[loi, complete, ir]])
    
# #     # Get predictions
# #     predictions = {}
# #     for name, model in models.items():
# #         predictions[name] = model.predict(input_data)[0]
    
# #     # Calculate average
# #     avg_cpi = np.mean(list(predictions.values()))
    
# #     # Visualization
# #     st.subheader("Prediction Results")
# #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
    
# #     # Model comparisons
# #     ax1.bar(predictions.keys(), predictions.values())
# #     ax1.set_title("Model Predictions Comparison")
# #     ax1.set_ylabel("CPI")
    
# #     # Average vs predictions
# #     ax2.bar(['Average CPI'] + list(predictions.keys()), 
# #             [avg_cpi] + list(predictions.values()))
# #     ax2.set_title("System Average vs Model Predictions")
    
# #     st.pyplot(fig)
    
# #     # LLM Decision Summary
# #     st.subheader("AI Decision Summary")
# #     llm_prompt = f"""
# #     Given input values:
# #     - LOI: {loi}%
# #     - COMPLETE: {complete}
# #     - IR: {ir}
    
# #     Model predictions:
# #     {chr(10).join([f'{k}: {v:.2f}' for k,v in predictions.items()])}
    
# #     Average CPI: {avg_cpi:.2f}
    
# #     Provide a concise executive summary of the economic implications 
# #     and recommended actions based on this data.
# #     """
    
# #     # LLM API Call
# #     headers = {
# #         "Authorization": f"Bearer {st.secrets['API_KEY']}",
# #         "Content-Type": "application/json"
# #     }
    
# #     response = requests.post(
# #         "https://api.llm-provider.com/v1/complete",
# #         headers=headers,
# #         json={
# #             "prompt": llm_prompt,
# #             "max_tokens": 500,
# #             "temperature": 0.3
# #         }
# #     )
    
# #     if response.status_code == 200:
# #         st.info(response.json()['choices'][0]['text'])
# #     else:
# #         st.error("Error generating summary")

########################################################

# import streamlit as st
# import pickle
# import numpy as np
# import requests

# # Load bid prediction model
# model = pickle.load(open('best_model.pkl', 'rb'))

# # Page configuration
# st.set_page_config(page_title="Bid Success Predictor", layout="wide")
# st.title("🏆 AI Bid Win Probability Calculator")

# # Input sliders with bid-specific parameters
# with st.form("bid_inputs"):
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         loi = st.slider("LOI", 1, 40, 15,
#                       help="Length of Interview with client")
#     with col2:
#         complete = st.slider("Completion Rate", 1.0, 3000.0, 480.0,
#                            help="Current project completion percentage")
#     with col3:
#         ir = st.slider("Interview Rate (/hr)", 0.0, 100.0, 36.0,
#                      help="Client meetings per hour")
#     with col4:
#         cpi = st.slider("Market CPI", 1, 92, 17,
#                       help="Current Customer Price Index")
    
#     submitted = st.form_submit_button("Calculate Win Probability")

# if submitted:
#     # Create input array in correct feature order
#     input_data = np.array([[complete, cpi, ir, loi]])
    
#     try:
#         # Get probability prediction
#         win_probability = model.predict_proba(input_data)[0][1]
        
#         # Visualization
#         st.subheader("Bid Prediction Analysis")
        
#         # Create three-column layout
#         left, center, right = st.columns([1,2,1])
        
#         with left:
#             st.metric("Win Probability", f"{win_probability*100:.1f}%")
#             st.write(f"LOI: {loi} mins")
#             st.write(f"Completion: {complete}%")
#             st.write(f"Interviews: {ir}/hr")
#             st.write(f"CPI: {cpi}")
        
#         with center:
#             # Probability gauge
#             gauge_html = f"""
#             <div style="text-align:center;">
#                 <svg width="250" height="150">
#                     <circle cx="125" cy="125" r="90" fill="none" stroke="#eee" stroke-width="12"/>
#                     <path d="M35,125 A90,90 0 0,1 215,125" 
#                           stroke="url(#gradient)" 
#                           stroke-width="12" 
#                           fill="none"
#                           stroke-dasharray="{win_probability*565} 565"/>
#                     <text x="50%" y="45%" font-size="32" text-anchor="middle">
#                         {win_probability*100:.1f}%
#                     </text>
#                     <defs>
#                         <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
#                             <stop offset="0%" style="stop-color:#ff4444"/>
#                             <stop offset="100%" style="stop-color:#44ff44"/>
#                         </linearGradient>
#                     </defs>
#                 </svg>
#             </div>
#             """
#             st.markdown(gauge_html, unsafe_allow_html=True)
            
#             # Recommendation
#             if win_probability >= 0.7:
#                 st.success("## 🚀 Strong Bid Recommendation: Proceed!")
#             elif win_probability >= 0.5:
#                 st.warning("## ⚖️ Competitive Bid: Consider Optimizations")
#             else:
#                 st.error("## 🛑 High Risk: Re-evaluate Bid Strategy")
        
#         with right:
#             # LLM Strategy Insights
#             st.subheader("AI Recommendations")
#             prompt = f"""As a bid strategy expert, analyze:
#             - Interview Length: {loi} mins
#             - Project Completion: {complete}%
#             - Client Meetings: {ir}/hr
#             - Market CPI: {cpi}
#             - Win Probability: {win_probability*100:.1f}%
            
#             Provide 3-5 bullet points for bid optimization"""
            
#             try:
#                 response = requests.post(
#                     "https://api.llm-provider.com/v1/complete",
#                     headers={"Authorization": f"Bearer {st.secrets['API_KEY']}"},
#                     json={
#                         "prompt": prompt,
#                         "max_tokens": 300,
#                         "temperature": 0.3
#                     }
#                 )
#                 st.markdown(f"``````")
#             except Exception as e:
#                 st.error("LLM service unavailable")

#     except Exception as e:
#         st.error(f"Prediction error: {str(e)}")

# # # Sidebar instructions
# # with st.sidebar:
# #     st.header("How to Use")
# #     st.markdown("""
# #     1. Adjust sliders for current bid parameters
# #     2. Click 'Calculate Win Probability'
# #     3. Review probability gauge and AI recommendations
# #     4. Use market CPI to contextualize pricing strategy
# #     """)
# #     st.image("bid_strategy_graph.png", caption="Bid Success Factors")


###############################################
import streamlit as st
import pickle
import numpy as np
import requests
import google.generativeai as genai 
# Page configuration
st.set_page_config(page_title="Bid Success Predictor", layout="wide")
st.title("🏆 Bid Analyzer")

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



# Input parameters with float validation
with st.form("bid_inputs"):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        loi = st.number_input("LOI (Minutes)", 
                            min_value=0.0, max_value=40.0, 
                            value=15.0, step=0.1,
                            format="%.1f")
    with col2:
        complete = st.number_input("Completion Rate", 
                                 min_value=0.0, max_value=3000.0,
                                 value=480.0, step=0.1,
                                 format="%.1f")
    with col3:
        ir = st.number_input("Interview Rate", 
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
            st.caption(f"LOI: {loi:.1f} mins")
            st.caption(f"Completion: {complete:.1f}%")
            st.caption(f"Interviews: {ir:.1f}/hr")
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
                    <text x="50%" y="45%" font-size="32" text-anchor="middle">
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
        
            # Create the prompt
            prompt = f"""As a bid strategy expert, analyze:
             - Interview Length: {loi:.1f} mins
            - Project Completion: {complete:.1f}%
            - Client Meetings: {ir:.1f}/hr
            - Market CPI: {cpi:.1f}
            - Win Probability: {win_probability*100:.1f}%
            
            IF win probability is higher provide positive encouragement to proceed by ensuring that
            this metric is 80% confident. If win probability is lower
            consider all the data and Provide 3-5 bullet points for bid optimization to win"""
            
            # Generate content
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 800
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
