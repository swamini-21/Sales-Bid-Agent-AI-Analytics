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
import pandas as pd
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
        
            # # Create the prompt
            # # prompt = f"""As a bid strategy expert, analyze:
            # #  - Interview Length: {loi:.1f} mins
            # # - Project Completion: {complete:.1f}%
            # # - Client Meetings: {ir:.1f}/hr
            # # - Market CPI: {cpi:.1f}
            # # - Win Probability: {win_probability*100:.1f}%
            
            # # IF win probability is higher provide positive encouragement to proceed by ensuring that
            # # this metric is 80% confident. If win probability is lower
            # # consider all the data and Provide 3-5 bullet points for bid optimization to win"""

            # prompt = f"""
            # You are a bid optimization consultant.
            # refer and analyze data from  {analyzer} throughly when responding
            # "You are a bid optimization consultant analyzing these metrics:
            # Incident Rate: {loi:.1f}  ( range - )
            # Complete : {complete:.1f}% (Typical competitive threshold - )
            # CPI: {cpi:.1f} (Current sector benchmark)
            # Calculated Win Probability: {win_probability*100:.1f}%
            # Interview
            # You have access to a data {analyzer} containing historical bid data with features such as LOI (Interview Length), Complete , IR (Incident Rate), and CPI (Cost per Interview), along with the outcome of each bid (Status: 'Invoiced'=Won, 'Lost'=Lost).

            # Metadata - 
            # Column - A	Project Code Child	ignore
            # Column - B	Project Code Parent	project # (same as Bid # in the bid data)
            # Column - C	Project Name	this is what we call the project
            # Column - G	Invoiced Date	Date when we offically called the project complete
            # Column - H	Actual Project Revenue 	total project value
            # Column - J	Client Name	self explanatory
            # Column - L	Complete	This is the units of things we sell.  So, a project will have 100 completes, for example. 
            # Column - M	CPI 	Price metric.  This is multiplied by "complete" to get the "Actual Project Revenue"
            # Column - N	Actual Ir	"Incidence Rate"  This is a MAJOR driver of CPI.
            # Column - O	Actual Loi	Length of Interview.  Another CPI driver, but less so than CPI
            # Column - P	Project Tags	you can ignore
            # Column - Q	Countries	self explanatory
            # Column - R	Audience	subcategory of the population being studied.  This may play a role in CPI

            # Your task:

            # Statistically analyze the entire df to determine the minimum, maximum, mean, and standard deviation for each feature (LOI, Complete, IR, CPI) for both won and lost bids.

            # For a given new bid (with provided values for LOI, Complete, IR, CPI and predicted win probability), compare each input value to the statistical range of historical data.

            # Identify if any input is an outlier (outside the 5th–95th percentile or more than 1.5 standard deviations from the mean of won bids).

            # Highlight the 3 most significant factors (features) that are limiting win probability, based on deviation from successful (won) bids.

            # Suggest 3 prioritized, actionable recommendations to improve win probability, specifying statistically-derived target ranges for each feature.

            # If win probability is already high (e.g., >80%), provide positive reinforcement but also highlight any metrics that are near historical risk thresholds, with suggestions for monitoring or mitigation.

            # Output format:
            # ❗ Critical Factors
            # 🛠️ Recommended Actions
            # 📈 Expected Outcome
            # 💡 Business Summary

            # Example Output:
            # ❗ Critical Factors

            # Interview Length (LOI) is 23 mins, which is above the 95th percentile of won bids (max recommended: 18 mins).

            # Projects completed so far is 40, below the mean for won bids (recommended: >65).

            # CPI is 28, which is higher than the optimal range for win (recommended: 8–18).

            # 🛠️ Recommended Actions

            # Shorten interview length to under 18 mins to align with successful bids.

            # DO MORe PORJECTS OR INCREASE PROJECTS completion number to at least 65% before submitting bid.

            # Re-evaluate pricing strategy to bring CPI into the 8–18 range.

            # 📈 Expected Outcome
            # Implementing these changes is statistically associated with a 30–45% lift in win probability based on historical patterns.

            # 💡 Business Summary
            # Your bid is currently outside the optimal range on several key factors. By aligning your metrics with those of previously successful bids, you can significantly increase your chances of winning. Monitor CPI and PROJECTS COMPLETED closely, as these are the strongest predictors of bid success in your historical data.
            # """

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

                Base all recommendations and analyses strictly on the data provided."""
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
