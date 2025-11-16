import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os 

st.title("Dementia Risk Prediction")
st.write("Enter patient details to estimate dementia risk.")
st.write("Assist a co-participant to help the process!!.")

import streamlit as st
st.write("Hello, Streamlit is running!")

# --- Model download and loading ---
MODEL_URL = "https://huggingface.co/ThisaraAdhikari04/dementia-risk-model/resolve/main/Dementia_model.pkl"
MODEL_PATH = "Dementia_model.pkl"

def download_file(url, output_path):
    if not os.path.exists(output_path):
        with st.spinner("Downloading model..."):
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

download_file(MODEL_URL, MODEL_PATH)
model = joblib.load(MODEL_PATH)


def predict(input_df):
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_df)[0]
        pred_class = int(np.argmax(prob))
    else:
        pred_class = int(model.predict(input_df)[0])
        prob = [1 - pred_class, pred_class]
    return pred_class, prob


def user_input_features():
    SEX = st.selectbox("Gender", ["Male", "Female"])
    SEX_val = 1 if SEX=="Male" else 2 if SEX=="Female" else np.nan #ok

    HISPANIC = st.selectbox("Hispanic/Latino Ethnicity", ["No", "Yes", "Unknown"])
    HISPANIC_val = 0 if HISPANIC=="No" else 1 if HISPANIC=="Yes" else np.nan  #ok

    HISPOR = st.selectbox(
    "Hispanic Origin (HISPOR)",
    ["Mexican", "Puerto Rican", "Cuban", "Dominican", "Central American", "South American", "Other", "Unknown"])
    HISPOR_val = {
    "Mexican": 1,
    "Puerto Rican": 2,
    "Cuban": 3,
    "Dominican": 4,
    "Central American": 5,
    "South American": 7,
    "Other": 8,
    "Unknown": np.nan}[HISPOR]  #ok 

    RACE = st.selectbox("Race", ["White", "Black/African American", "American Indian","Native Hawaiian", "Asian", "Other", "Unknown"])
    RACE_val = 1 if RACE=="White" else 2 if RACE=="Black/African American" else 3 if RACE=="American Indian" else 4 if RACE=="Native Hawaiian" else 5 if RACE=="Asian" else 6 if RACE=="Other" else np.nan
    #ok
    
    PRIMLANG = st.selectbox(
    "Primary Language",
    ["English", "Spanish", "Mandarin", "Cantonese", "Russian", "Japanese", "Other", "Unknown"])
    PRIMLANG_val = {
    "English": 1,
    "Spanish": 2,
    "Mandarin": 3,
    "Cantonese": 4,
    "Russian": 5,
    "Japanese": 6,
    "Other": 7,
    "Unknown": np.nan}[PRIMLANG] #ok

    educ_options = ["Unknown"]+[str(i) for i in range(0, 37)] 
    EDUC = st.selectbox("Years of Education", educ_options)
    EDUC_val = np.nan if EDUC == "Unknown" else int(EDUC) #ok


    MARISTAT = st.selectbox(
    "Marital Status",
    ["Married", "Widowed", "Divorced", "Separated", "Never married/Annulled", "Domestic partner", "Unknown"])
    MARISTAT_val = (
    1 if MARISTAT == "Married" else
    2 if MARISTAT == "Widowed" else
    3 if MARISTAT == "Divorced" else
    4 if MARISTAT == "Separated" else
    5 if MARISTAT == "Never married/Annulled" else
    6 if MARISTAT == "Domestic partner" else
    np.nan) #ok

    NACCLIVS = st.selectbox(
    "Living situation",
    ["Alone", "With spouce/partner", "With Relative/Friend", "With a Group", "Other","Unknown"])
    NACCLIVS_val = (
    1 if NACCLIVS == "Alone" else
    2 if NACCLIVS == "With spouce/partner" else
    3 if NACCLIVS == "With Relative/Friend" else
    4 if NACCLIVS == "With a Group" else
    5 if NACCLIVS == "Other" else
    np.nan) #ok

    INDEPEND = st.selectbox(
    "Level of Independance",
    ["able to live independantly", "Require assistance with complex activities", "Require assistance with basic activities", "Completely dependent","Unknown"])
    INDEPEND_val = (
    1 if INDEPEND == "able to live independantly" else
    2 if INDEPEND == "Require assistance with complex activities" else
    3 if INDEPEND == "Require assistance with basic activities" else
    4 if INDEPEND == "Completely dependent" else
    np.nan) #ok

    RESIDENC = st.selectbox(
    "Residence Type",
    [    "Private residence",
        "Retirement community",
        "Assisted living",
        "Nursing facility",
        "Unknown"])
    RESIDENC_val = (
    1 if RESIDENC == "Private residence" else
    2 if RESIDENC == "Retirement community" else
    3 if RESIDENC == "Assisted living" else
    4 if RESIDENC == "Nursing facility" else
    np.nan)  #ok

    HANDED = st.selectbox("Handedness", ["Left", "Right","Ambidextrous", "Unknown"])
    HANDED_val = 1 if HANDED=="Left" else 2 if HANDED=="Right" else 3 if HANDED=="Ambidextrous" else np.nan  #ok

    NACCAGE = st.number_input("Person's Age", min_value=18, max_value=120, value=70)#ok
    NACCAGEB = st.number_input("Confirm Person's age", min_value=18, max_value=120, value=70)#ok
    
    birth_options = ["Unknown"]+[str(i) for i in range(1875, 2026)]
    INBIRYR = st.selectbox("Co-participant Birth Year", birth_options)
    INBIRYR_val = np.nan if INBIRYR == "Unknown" else int(INBIRYR) #ok

    NEWINF = st.selectbox("Familiar with data entering process?", ["No", "Yes","Unknown"])
    NEWINF_val = 0 if NEWINF=="No" else 1 if NEWINF=="Yes" else np.nan  #ok

    ineduc_options = ["Unknown"] + [str(i) for i in range(0, 37)]
    INEDUC = st.selectbox("Co-participant Years of Education", ineduc_options)
    INEDUC_val = np.nan if INEDUC == "Unknown" else int(INEDUC) #ok

    INRELTO_options = [
    "Spouse / Partner / Companion",
    "Child",
    "Sibling",
    "Other Relative",
    "Friend / Neighbor / Known",
    "Paid Caregiver / Clinician",
    "Other",
    "Unknown"]
    INRELTO = st.selectbox("Co-participant's Relationship to Subject", INRELTO_options)
    INRELTO_val = (
    1 if INRELTO == "Spouse / Partner / Companion" else
    2 if INRELTO == "Child" else
    3 if INRELTO == "Sibling" else
    4 if INRELTO == "Other Relative" else
    5 if INRELTO == "Friend / Neighbor / Known" else
    6 if INRELTO == "Paid Caregiver / Clinician" else
    7 if INRELTO == "Other" else
    np.nan )
    
    INLIVWTH_options = [
    "No",
    "Yes",
    "Unknown"]
    INLIVWTH = st.selectbox("Does the co-participant live with person?", INLIVWTH_options)
    INLIVWTH_val = (
    0 if INLIVWTH == "No" else
    1 if INLIVWTH == "Yes" else
    np.nan) #ok
    
    INRELY_options = [
    "No",
    "Yes",
    "Unknown"]
    INRELY = st.selectbox("How sure can you be about co-participant's answers?", INRELY_options)
    INRELY_val = (
    0 if INRELY == "No" else
    1 if INRELY == "Yes" else
    np.nan) #ok
    
    NACCFAM_options = [
    "No report of family member with cognitive impairment",
    "Family member reported to have cognitive impairment",
    "Unknown / Not available"]
    NACCFAM = st.selectbox("Family member Cognitive Impairment", NACCFAM_options)
    NACCFAM_val = (
    0 if NACCFAM == "No report of family member with cognitive impairment" else
    1 if NACCFAM == "Family member reported to have cognitive impairment" else
    np.nan) #ok
    
    NACCMOM_options = [
    "No report of mother with cognitive impairment",
    "Mother reported to have cognitive impairment",
    "Unknown / Not available"]
    NACCMOM = st.selectbox("Mother Cognitive Impairment", NACCMOM_options)
    NACCMOM_val = (
    0 if NACCMOM == "No report of mother with cognitive impairment" else
    1 if NACCMOM == "Mother reported to have cognitive impairment" else
    np.nan) #ok
    
    NACCDAD_options = [
    "No report of father with cognitive impairment",
    "Father reported to have cognitive impairment",
    "Unknown / Not available"]
    NACCDAD = st.selectbox("Father Cognitive Impairment", NACCDAD_options)
    NACCDAD_val = (
    0 if NACCDAD == "No report of father with cognitive impairment" else
    1 if NACCDAD == "Father reported to have cognitive impairment" else
    np.nan) #ok
        
    ANYMEDS = st.selectbox("Taking any medications?", ["No", "Yes", "Unknown"])
    ANYMEDS_val = 0 if ANYMEDS=="No" else 1 if ANYMEDS=="Yes" else np.nan #ok

    NACCAMD_options = ["Unknown"] + [str(i) for i in range(0, 41)]
    NACCAMD = st.selectbox(" Total number of medications take?",  NACCAMD_options)
    NACCAMD_val = np.nan if NACCAMD == "Unknown" else int(NACCAMD) #ok

    TOBAC100 = st.selectbox("Smoked 100+ cigarettes in lifetime?", ["No", "Yes", "Unknown"])
    TOBAC100_val = 0 if TOBAC100=="No" else 1 if TOBAC100=="Yes" else np.nan #ok

    SMOKYRS_options = ["Unknown"] + [str(i) for i in range(0, 88)]
    SMOKYRS = st.selectbox("Total years smoked cigarettes", SMOKYRS_options)
    SMOKYRS_val = np.nan if SMOKYRS == "Unknown" else int(SMOKYRS) #ok
    
    pack_options = [
    "No reported cigarette use",
    "1 cigarette to less than 1/2 pack",
    "½ pack to less than 1 pack",
    "1 pack to 1½ packs",
    "1½ packs to 2 packs",
    "More than two packs",
    "Not applicable",
    "Unknown",
    "Not available"]
    PACKSPER = st.selectbox("Cigarettes per day", pack_options)
    PACKSPER_val = (
    0 if PACKSPER == "No reported cigarette use" else
    1 if PACKSPER == "1 cigarette to less than 1/2 pack" else
    2 if PACKSPER == "½ pack to less than 1 pack" else
    3 if PACKSPER == "1 pack to 1½ packs" else
    4 if PACKSPER == "1½ packs to 2 packs" else
    5 if PACKSPER == "More than two packs" else
    np.nan) #ok

    CVHATT = st.selectbox(
    "Any Heart attack/cardiac arrest? ",
    ["No", "Recent","Remote", "Unknown"])
    CVHATT_val = (
    0 if CVHATT == "No" else
    1 if CVHATT == "Recent" else
    2 if CVHATT == "Remote" else
    np.nan) #ok

    CVBYPASS = st.selectbox(
    "Had a Cardiac bypass procedure? ",
    ["No", "Recent","Remote", "Unknown"])
    CVBYPASS_val = (
    0 if CVBYPASS == "No" else
    1 if CVBYPASS == "Recent" else
    2 if CVBYPASS == "Remote" else
    np.nan) #ok

    CVPACE = st.selectbox(
    "Use a pacemaker? ",
    ["No", "Recent","Remote", "Unknown"])
    CVPACE_val = (
    0 if CVPACE == "No" else
    1 if CVPACE == "Recent" else
    2 if CVPACE == "Remote" else
    np.nan) #ok

    CVHVALVE = st.selectbox(
    "Heart valve replacement or repair? ",
    ["No", "Recent","Remote", "Unknown"])
    CVHVALVE_val = (
    0 if CVHVALVE == "No" else
    1 if CVHVALVE == "Recent" else
    2 if CVHVALVE == "Remote" else
    np.nan) #ok

    CBSTROKE = st.selectbox(
    "Any Stroke? ",
    ["No", "Recent","Remote", "Unknown"])
    CBSTROKE_val = (
    0 if CBSTROKE == "No" else
    1 if CBSTROKE == "Recent" else
    2 if CBSTROKE == "Remote" else
    np.nan) #ok

    TBIBRIEF = st.selectbox(
    "Traumatic brain injury (TbI) with brief loss of consciousness? ",
    ["No", "Single","Multiple", "Unknown"])
    TBIBRIEF_val = (
    0 if TBIBRIEF == "No" else
    1 if TBIBRIEF == "Single" else
    2 if TBIBRIEF == "Multiple" else
    np.nan) #ok

    TBIEXTEN = st.selectbox(
    "TbI with extended loss of consciousness(5 minutes or longer)? ",
    ["No", "Single","Multiple", "Unknown"])
    TBIEXTEN_val = (
    0 if TBIEXTEN == "No" else
    1 if TBIEXTEN == "Single" else
    2 if TBIEXTEN == "Multiple" else
    np.nan) #ok

    DEP2YRS = st.selectbox(
    "Active depression in the last two years?",
    ["No", "Yes", "Unknown"])
    DEP2YRS_val = (
    0 if DEP2YRS == "No" else
    1 if DEP2YRS == "Yes" else
    np.nan) #ok
    
    DEPOTHR = st.selectbox(
    "Depression episodes more than two years ago?",
    ["No", "Yes", "Unknown"])
    DEPOTHR_val = (
    0 if DEPOTHR == "No" else
    1 if DEPOTHR == "Yes" else
    np.nan) #ok

    NACCTBI = st.selectbox(
    "History of Traumatic Brain Injury (TBI)",
    ["No", "Yes", "Unknown"])
    NACCTBI_val = (
    0 if NACCTBI == "No" else
    1 if NACCTBI == "Yes" else
    np.nan) #ok

    HEIGHT_input = st.number_input("Height (m)", min_value=0.5, max_value=2.5, value=1.7, step=0.01)
    HEIGHT = HEIGHT_input * 39.3701 #ok
    
    WEIGHT_input = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=70.0, step=0.1)
    WEIGHT = WEIGHT_input * 2.20462 #ok
    
    NACCBMI = (WEIGHT * 703) / (HEIGHT ** 2) #ok

    VISION = st.selectbox("Without corrective lenses, is vision functionally normal?", ["No", "Yes", "Unknown"])
    VISION_val = 0 if VISION=="No" else 1 if VISION=="Yes" else np.nan #ok

    VISCORR = st.selectbox("usually wear corrective lenses?", ["No", "Yes", "Unknown"])
    VISCORR_val = 0 if VISCORR=="No" else 1 if VISCORR=="Yes" else np.nan #ok

    VISWCORR = st.selectbox(" vision functionally normal with corrective lenses?", ["No", "Yes", "Unknown/Not wear lenses"])
    VISWCORR_val = 0 if VISWCORR=="No" else 1 if VISWCORR=="Yes" else np.nan #ok

    HEARING = st.selectbox("Hearing functionally normal without a hearing aid(s)?", ["No", "Yes", "Unknown"])
    HEARING_val = 0 if HEARING=="No" else 1 if HEARING=="Yes" else np.nan #ok

    HEARAID = st.selectbox("usually wear a hearing aid(s)?", ["No", "Yes", "Unknown"])
    HEARAID_val = 0 if HEARAID=="No" else 1 if HEARAID=="Yes" else np.nan #ok

    HEARWAID = st.selectbox("hearing functionally normal with a hearing aid(s)?", ["No", "Yes", "Unknown?Do not wear any"])
    HEARWAID_val = 0 if HEARWAID=="No" else 1 if HEARWAID=="Yes" else np.nan #ok

    HXSTROKE = st.selectbox("Any history of stroke?", ["No", "Yes", "Unknown"])
    HXSTROKE_val = 0 if HXSTROKE=="No" else 2 if HXSTROKE=="Yes" else np.nan #ok

    HALL = st.selectbox(" Hallucinations in the last month?", ["No", "Yes", "Unknown"])
    HALL_val = 0 if HALL=="No" else 1 if HALL=="Yes" else np.nan #ok

    APP = st.selectbox("Appetite and eating problems in the last month", ["No", "Yes", "Unknown"])
    APP_val = 0 if APP=="No" else 1 if APP=="Yes" else np.nan #ok

    BILLS = st.selectbox(
    "Can pay bills/checks recently?",
    ["Normal", 
     "Has difficulty but does by self", 
     "Requires assistance",
     "Dependent",
     "Never did",
     "Unknown"])
    BILLS_val = (
    0 if BILLS == "Normal" else
    1 if BILLS == "Has difficulty but does by self" else
    2 if BILLS == "Requires assistance" else
    3 if BILLS == "Dependent" else
    np.nan) #ok

    TAXES = st.selectbox(
    "Can record accounts and money recently?",
    ["Normal", 
     "Has difficulty but does by self", 
     "Requires assistance",
     "Dependent",
     "Never did",
     "Unknown"])
    TAXES_val = (
    0 if TAXES == "Normal" else
    1 if TAXES == "Has difficulty but does by self" else
    2 if TAXES == "Requires assistance" else
    3 if TAXES == "Dependent" else
    np.nan) #ok

    SHOPPING = st.selectbox(
    "Can shop by self recently?",
    ["Normal", 
     "Has difficulty but does by self", 
     "Requires assistance",
     "Dependent",
     "Never did",
     "Unknown"])
    SHOPPING_val = (
    0 if SHOPPING == "Normal" else
    1 if SHOPPING == "Has difficulty but does by self" else
    2 if SHOPPING == "Requires assistance" else
    3 if SHOPPING == "Dependent" else
    np.nan) #ok
    
    GAMES = st.selectbox(
    "Can play games(chess/bridge) recently?",
    ["Normal", 
     "Has difficulty but does by self", 
     "Requires assistance",
     "Dependent",
     "Never did",
     "Unknown"])
    GAMES_val = (
    0 if GAMES == "Normal" else
    1 if GAMES == "Has difficulty but does by self" else
    2 if GAMES == "Requires assistance" else
    3 if GAMES == "Dependent" else
    np.nan) #ok

    STOVE = st.selectbox(
    "Can make tea/off the stove by self recently?",
    ["Normal", 
     "Has difficulty but does by self", 
     "Requires assistance",
     "Dependent",
     "Never did",
     "Unknown"])
    STOVE_val = (
    0 if STOVE == "Normal" else
    1 if STOVE == "Has difficulty but does by self" else
    2 if STOVE == "Requires assistance" else
    3 if STOVE == "Dependent" else
    np.nan) #ok

    MEALPREP = st.selectbox(
    "Can prepare a meal recently?",
    ["Normal", 
     "Has difficulty but does by self", 
     "Requires assistance",
     "Dependent",
     "Never did",
     "Unknown"])
    MEALPREP_val = (
    0 if MEALPREP == "Normal" else
    1 if MEALPREP == "Has difficulty but does by self" else
    2 if MEALPREP == "Requires assistance" else
    3 if MEALPREP == "Dependent" else
    np.nan) #ok

    EVENTS = st.selectbox(
    "Can keep track of events recently?",
    ["Normal", 
     "Has difficulty but does by self", 
     "Requires assistance",
     "Dependent",
     "Never did",
     "Unknown"])
    EVENTS_val = (
    0 if EVENTS == "Normal" else
    1 if EVENTS == "Has difficulty but does by self" else
    2 if EVENTS == "Requires assistance" else
    3 if EVENTS == "Dependent" else
    np.nan) #ok

    PAYATTN = st.selectbox(
    "Can pay attention recently?",
    ["Normal", 
     "Has difficulty but does by self", 
     "Requires assistance",
     "Dependent",
     "Never did",
     "Unknown"])
    PAYATTN_val = (
    0 if PAYATTN == "Normal" else
    1 if PAYATTN == "Has difficulty but does by self" else
    2 if PAYATTN == "Requires assistance" else
    3 if PAYATTN == "Dependent" else
    np.nan) #ok

    REMDATES = st.selectbox(
    "Can Remember dates and occasions recently?",
    ["Normal", 
     "Has difficulty but does by self", 
     "Requires assistance",
     "Dependent",
     "Never did",
     "Unknown"])
    REMDATES_val = (
    0 if REMDATES == "Normal" else
    1 if REMDATES == "Has difficulty but does by self" else
    2 if REMDATES == "Requires assistance" else
    3 if REMDATES == "Dependent" else
    np.nan) #ok

    TRAVEL = st.selectbox(
    "Can Travel recently?",
    ["Normal", 
     "Has difficulty but does by self", 
     "Requires assistance",
     "Dependent",
     "Never did",
     "Unknown"])
    TRAVEL_val = (
    0 if TRAVEL == "Normal" else
    1 if TRAVEL == "Has difficulty but does by self" else
    2 if TRAVEL == "Requires assistance" else
    3 if TRAVEL == "Dependent" else
    np.nan) #ok

    data = {
    "SEX": SEX_val, "HISPANIC": HISPANIC_val, "HISPOR": HISPOR_val, "RACE": RACE_val,
    "PRIMLANG": PRIMLANG_val, "EDUC": EDUC_val, "MARISTAT": MARISTAT_val, "NACCLIVS": NACCLIVS_val,
    "INDEPEND": INDEPEND_val, "RESIDENC": RESIDENC_val, "HANDED": HANDED_val,
    "NACCAGE": NACCAGE, "NACCAGEB": NACCAGEB, "INBIRYR": INBIRYR_val, "NEWINF": NEWINF_val,
    "INRELTO": INRELTO_val, "INLIVWTH": INLIVWTH_val, "INRELY": INRELY_val, "NACCFAM": NACCFAM_val,
    "NACCMOM": NACCMOM_val, "NACCDAD": NACCDAD_val, "ANYMEDS": ANYMEDS_val, "NACCAMD": NACCAMD_val,
    "TOBAC100": TOBAC100_val, "SMOKYRS": SMOKYRS_val, "PACKSPER": PACKSPER_val,
    "CVHATT": CVHATT_val, "CVBYPASS": CVBYPASS_val, "CVPACE": CVPACE_val, "CVHVALVE": CVHVALVE_val,
    "CBSTROKE": CBSTROKE_val, "TBIBRIEF": TBIBRIEF_val, "TBIEXTEN": TBIEXTEN_val,
    "DEP2YRS": DEP2YRS_val, "DEPOTHR": DEPOTHR_val, "NACCTBI": NACCTBI_val,
    "HEIGHT": HEIGHT, "WEIGHT": WEIGHT, "NACCBMI": NACCBMI, "VISION": VISION_val,
    "VISCORR": VISCORR_val, "VISWCORR": VISWCORR_val, "HEARING": HEARING_val,
    "HEARAID": HEARAID_val, "HEARWAID": HEARWAID_val, "HXSTROKE": HXSTROKE_val,
    "HALL": HALL_val, "APP": APP_val,
    "BILLS": BILLS_val, "TAXES": TAXES_val, "SHOPPING": SHOPPING_val, "GAMES": GAMES_val,
    "STOVE": STOVE_val, "MEALPREP": MEALPREP_val, "EVENTS": EVENTS_val, "PAYATTN": PAYATTN_val,
    "REMDATES": REMDATES_val, "TRAVEL": TRAVEL_val}

    return pd.DataFrame([data])

input_df = user_input_features()

if st.button("Predict"):
    prediction, prediction_prob = dummy_predict(input_df)
    prediction_label = {0: "Non-Dementia", 1: "Risk of Dementia"}

    st.subheader("Prediction Result")
    st.write(f"Predicted Dementia State: **{prediction_label[prediction]}**")
    st.write(f"Probability of Non-Dementia: {prediction_prob[0]*100:.2f}%")
    st.write(f"Probability of Risk of Dementia: {prediction_prob[1]*100:.2f}%")
