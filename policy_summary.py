import streamlit as st
import numpy as np
import pandas as pd
import pickle
import shap
import random
import requests
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# ------------------------- Config -------------------------

CHAT_API_KEY = "your-key"  # API key from DeepSeek
model_path = "global_model(2).h5"

# ---------------------- Load Data -------------------------
with open("global_shap_values.pkl", "rb") as f:
    shap_data = pickle.load(f)

shap_values = shap_data["shap_values"]
X_sample = shap_data["X_sample"]
feature_names = shap_data["feature_names"]

with open("fl_data.pkl", "rb") as f:
    fl_data = pickle.load(f)

data_by_country = fl_data["data_by_country"]
X_scaled = fl_data["X_scaled"]
feature_names_c = fl_data["feature_names"]

with open("scaler_final.pkl", "rb") as f:
    scaler_final = pickle.load(f)


lc = ['Afghanistan', 'Angola', 'Albania', 'United Arab Emirates',
       'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan',
       'Burundi', 'Belgium', 'Benin', 'Burkina Faso', 'Bangladesh',
       'Bulgaria', 'Bahrain', 'Bosnia and Herzegovina', 'Belarus',
       'Bolivia', 'Brazil', 'Barbados', 'Botswana',
       'Central African Republic', 'Canada', 'Switzerland', 'Chile',
       'China', "Cote d'Ivoire", 'Cameroon',
       'Democratic Republic of Congo', 'Congo', 'Colombia', 'Comoros',
       'Cape Verde', 'Costa Rica', 'Cuba', 'Cyprus', 'Czechia', 'Germany',
       'Djibouti', 'Dominica', 'Denmark', 'Dominican Republic', 'Algeria',
       'Ecuador', 'Egypt', 'Spain', 'Estonia', 'Ethiopia', 'Finland',
       'France', 'Gabon', 'United Kingdom', 'Georgia', 'Ghana', 'Guinea',
       'Gambia', 'Guinea-Bissau', 'Equatorial Guinea', 'Greece',
       'Guatemala', 'Hong Kong', 'Honduras', 'Croatia', 'Haiti',
       'Hungary', 'Indonesia', 'India', 'Ireland', 'Iran', 'Iraq',
       'Iceland', 'Israel', 'Italy', 'Jamaica', 'Jordan', 'Japan',
       'Kazakhstan', 'Kenya', 'Kyrgyzstan', 'Cambodia', 'South Korea',
       'Kuwait', 'Laos', 'Lebanon', 'Liberia', 'Libya', 'Saint Lucia',
       'Sri Lanka', 'Lesotho', 'Lithuania', 'Luxembourg', 'Latvia',
       'Morocco', 'Moldova', 'Madagascar', 'Mexico', 'North Macedonia',
       'Mali', 'Malta', 'Myanmar', 'Montenegro', 'Mongolia', 'Mozambique',
       'Mauritania', 'Mauritius', 'Malawi', 'Malaysia', 'Namibia',
       'Niger', 'Nigeria', 'Nicaragua', 'Netherlands', 'Norway', 'Nepal',
       'New Zealand', 'Oman', 'Pakistan', 'Panama', 'Peru', 'Philippines',
       'Poland', 'North Korea', 'Portugal', 'Paraguay', 'Palestine',
       'Qatar', 'Romania', 'Russia', 'Rwanda', 'Saudi Arabia', 'Senegal',
       'Singapore', 'Sierra Leone', 'El Salvador', 'Serbia',
       'Sao Tome and Principe', 'Slovakia', 'Slovenia', 'Sweden',
       'Eswatini', 'Seychelles', 'Syria', 'Chad', 'Togo', 'Thailand',
       'Tajikistan', 'Turkmenistan', 'Trinidad and Tobago', 'Tunisia',
       'Turkey', 'Taiwan', 'Tanzania', 'Uganda', 'Ukraine', 'Uruguay',
       'United States', 'Uzbekistan', 'Venezuela', 'Vietnam', 'Yemen',
       'South Africa', 'Zambia', 'Zimbabwe']

ln=[  0,   3,   1, 154,   4,   5,   6,   7,   8,  21,  13,  14,  20,
        10,  19,   9,  16,  12,  15,  18,  11,  17,  26,  24, 141,  28,
        29,  34,  23,  39,  32,  30,  31,  25,  33,  36,  37,  38,  56,
        41,  42,  40,  43,   2,  44,  45, 138,  48,  50,  51,  52,  53,
       155,  55,  57,  60,  54,  61,  47,  58,  59,  64,  63,  35,  62,
        65,  68,  67,  71,  69,  70,  66,  72,  73,  74,  76,  75,  77,
        78,  80,  22, 137,  79,  81,  83,  85,  86, 126, 139,  84,  87,
        88,  82, 100,  97,  89,  96, 111,  92,  93, 102,  99,  98, 101,
        94,  95,  90,  91, 103, 108, 109, 107, 105, 112, 104, 106, 113,
       114, 116, 118, 119, 120, 110, 121, 117, 115, 122, 123, 124, 125,
       128, 129, 133, 132,  46, 130, 127, 134, 135, 140,  49, 131, 142,
        27, 147, 146, 144, 151, 148, 149, 150, 143, 145, 152, 153, 157,
       156, 158, 159, 160, 161, 136, 162, 163]

country_dict = dict(zip(lc,ln))
valid_countries=[]
for i in country_dict:
    if country_dict[i] in data_by_country.keys():
        valid_countries.append(i)

# ---------------------- Functions ------------------
def summarize_with_openrouter(text):
    try:
        headers = {
                    'Authorization': f'Bearer {CHAT_API_KEY}',
                    'Content-Type': 'application/json'
                  }

        # Define the request payload (data)
        data = {
              "model": "deepseek/deepseek-chat",
              "messages": [{
                "role": "system",
                "content": "You are a policy analyst helping governments understand SHAP feature impact on emissions. Based on the input, generate a short policy suggestion in plain language."
                },
                {
                "role": "user",
                "content": text
                }]
              }

        # Send the POST request to the DeepSeek API
        response = requests.post('https://openrouter.ai/api/v1/chat/completions', json=data, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
           result = response.json()
           summary = result['choices'][0]['message']['content']
           return summary
        else:
           print("Failed to fetch data from API. Status Code:", response.status_code)

    except Exception as e:
        return f"Summarization failed: {e}"

def convert_shap_to_text(shap_value, feature_names, max_features=5):
    try:
        if isinstance(shap_value, shap.Explanation):
            shap_array = shap_value.values
        else:
            shap_array = shap_value

        if len(shap_array.shape) == 3:
            shap_array = shap_array.mean(axis=2)

        mean_shap = np.abs(shap_array).mean(axis=0)  # Avoid sign cancellation

        if len(mean_shap) != len(feature_names):
            raise ValueError("Mismatch between SHAP values and feature names.")

        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": mean_shap
        }).sort_values(by="SHAP Value", ascending=False).head(max_features)

        text_summary = "Top contributing features and their impact:\n"
        for _, row in shap_df.iterrows():
            text_summary += f"- {row['Feature']} contributes significantly.\n"

        return text_summary, shap_df
    except Exception as e:
        return f"Error summarizing SHAP values: {str(e)}"


def plot_top_shap_bar(shap_df, title="Top Feature Importances", max_features=10):
    fig, ax = plt.subplots()
    shap_df.head(max_features).plot(kind="barh", x="Feature", y="SHAP Value", ax=ax, color="skyblue", legend=False)
    ax.set_title(title)
    plt.gca().invert_yaxis()
    st.pyplot(fig)


# ---------------------- Streamlit App --------------------
st.title("SHAP Policy Insights via DeepSeek")
st.write("Analyze global and local emission contributions with personalized suggestions.")

model = load_model(model_path)

# Tab Layout
tabs = st.tabs(["Global Policy Suggestions", "Local (Country-wise) Suggestions"])

# ------------------ Global Tab ---------------------------
with tabs[0]:
    st.subheader("Global Model Analysis")
    try:
        shap_text, shap_df = convert_shap_to_text(shap_values, feature_names)
        summary = summarize_with_openrouter(shap_text)
        plot_top_shap_bar(shap_df, title="Top Feature Importances")

        st.write("### Policy Suggestion")
        st.info(summary)
    except Exception as e:
        st.error(f"Failed to analyze global model: {e}")

# ------------------ Local Tab ----------------------------
with tabs[1]:
    st.subheader("Choose a Country")

    selected_country_name = st.selectbox("Select a country:", valid_countries)

    run = st.button("Enter")

    if run:
        try:
            # Map selected country name back to number
            selected_country_id = country_dict[selected_country_name]

            # Use the id to get the data
            X_localr, _ = data_by_country[selected_country_id]
            X_local = scaler_final.transform(X_localr) #Scale the data to ensure consistency with model inputs during training
            X_local_df = pd.DataFrame(X_local, columns=feature_names_c)
            st.write(X_local)

            explainer_local = shap.Explainer(model, X_sample)
            shap_values_local = explainer_local(X_local_df)
            st.write(shap_values_local)

            shap_text_local, shap_df_local = convert_shap_to_text(shap_values_local, feature_names)
            summary_local = summarize_with_openrouter(shap_text_local)
            st.write("this is shap df",shap_df_local)

            plot_top_shap_bar(shap_df_local, title=f"Top Feature Importances for {selected_country_name}")

            st.write("### Policy Suggestion")
            st.info(summary_local)

        except Exception as e:
            st.error(f"Failed to analyze {selected_country_name}: {e}")

