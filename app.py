import streamlit as st
import pandas as pd
import pickle

# Page Config for a better browser tab title and layout
st.set_page_config(page_title="IPL Win Predictor", layout="centered")

# Custom CSS to improve the look and feel
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Declaring the teams
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore', 
         'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings', 
         'Rajasthan Royals', 'Delhi Capitals']

# declaring the venues
cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

# Load Model
pipe = pickle.load(open('pipe.pkl', 'rb'))

st.title('🏏 IPL Win Predictor')
st.markdown("---")

# Team Selection
col1, col2 = st.columns(2)
with col1:
    battingteam = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowlingteam = st.selectbox('Select the bowling team', sorted(teams))

# City Selection
city = st.selectbox('Select the city where the match is being played', sorted(cities))

# Target Input
target = st.number_input('Target Score', min_value=0, step=1)

# Match Progress Inputs
st.markdown("### Match Progress")
col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('Current Score', min_value=0, step=1)
with col4:
    overs = st.number_input('Overs Completed', min_value=0.0, max_value=20.0, step=0.1)
with col5:
    wickets = st.number_input('Wickets Fallen', min_value=0, max_value=10, step=1)

st.markdown("<br>", unsafe_allow_html=True)

if st.button('Predict Probability'):
    # Backend Logic (No names or values changed)
    runs_left = target-score
    balls_left = 120-(overs*6)
    wickets = 10-wickets
    currentrunrate = score/overs if overs > 0 else 0
    requiredrunrate = (runs_left*6)/balls_left if balls_left > 0 else 0

    input_df = pd.DataFrame({
        'batting_team': [battingteam], 
        'bowling_team': [bowlingteam], 
        'city': [city], 
        'runs_left': [runs_left], 
        'balls_left': [balls_left], 
        'wickets': [wickets], 
        'total_runs_x': [target], 
        'cur_run_rate': [currentrunrate], 
        'req_run_rate': [requiredrunrate]
    })

    result = pipe.predict_proba(input_df)
    lossprob = result[0][0]
    winprob = result[0][1]

    # UI Improvement for Results
    st.markdown("---")
    st.subheader("Win Probability")
    
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.markdown(f"### {battingteam}")
        st.title(f"{round(winprob*100)}%")
        st.progress(winprob)
        
    with res_col2:
        st.markdown(f"### {bowlingteam}")
        st.title(f"{round(lossprob*100)}%")
        st.progress(lossprob)

    # Added a simple Rate Comparison Bar Chart for better UX
    st.markdown("---")
    st.markdown("### Rate Analysis")
    rates = pd.DataFrame({
        'Metric': ['Current Run Rate', 'Required Run Rate'],
        'Value': [round(currentrunrate, 2), round(requiredrunrate, 2)]
    })
    st.bar_chart(rates, x='Metric', y='Value', color="#ff4b4b")