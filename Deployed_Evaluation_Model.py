import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

st.set_page_config(page_title="Car Evaluation Explorer", layout="wide")
st.title("🚗 Car Evaluation Classification Explorer")

# Load dataset directly from UCI repository
@st.cache_data
def load_data():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'
    cols = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    df = pd.read_csv(url, header=None, names=cols)
    return df

# Preprocessing function
@st.cache_data
def preprocess(df):
    X = df.drop('class', axis=1)
    y = df['class']
    le = LabelEncoder()
    y_num = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_num, test_size=0.2, random_state=42, stratify=y_num
    )
    ct = ColumnTransformer(
        transformers=[
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), X.columns.tolist())
        ]
    )
    return X_train, X_test, y_train, y_test, ct, le

# Training function (ignore caching hash for _preprocessor)
@st.cache_resource
def train_models(X_train, X_test, y_train, y_test, _preprocessor):
    """
    Trains multiple models and returns accuracy results and pipelines.
    The _preprocessor argument is prefixed with underscore to avoid Streamlit hashing error.
    """
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Linear Regression': LinearRegression()
    }
    results = {}
    pipelines = {}
    for name, model in models.items():
        pipe = Pipeline([('preproc', _preprocessor), ('model', model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        if name == 'Linear Regression':
            preds = np.rint(preds).astype(int)
            preds = np.clip(preds, y_train.min(), y_train.max())
        acc = accuracy_score(y_test, preds)
        results[name] = acc
        pipelines[name] = pipe
    return results, pipelines

# Main app logic

def main():
    # Load data
    df = load_data()

    # Show dataset preview
    st.subheader("Dataset Preview (first 10 rows)")
    st.dataframe(df.head(10))

    # EDA: class distribution
    st.subheader("Class Distribution")
    dist = df['class'].value_counts()
    fig, ax = plt.subplots()
    dist.plot(kind='bar', ax=ax)
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title('Car Acceptability Class Distribution')
    st.pyplot(fig)

    # Trigger training
    if st.button("🔍 Train & Evaluate Models"):
        with st.spinner("Training models..."):
            X_train, X_test, y_train, y_test, preprocessor, label_encoder = preprocess(df)
            results, pipelines = train_models(X_train, X_test, y_train, y_test, preprocessor)

        # Display results
        st.subheader("Model Accuracies")
        res_df = pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy'])
        res_df = res_df.sort_values(by='Accuracy', ascending=False)
        st.table(res_df)

        best_name = res_df.index[0]
        st.write(f"**Best Model:** {best_name} with accuracy {res_df.iloc[0,0]:.4f}")

        # Store in session state
        st.session_state['pipelines'] = pipelines
        st.session_state['label_encoder'] = label_encoder
        st.session_state['best_name'] = best_name

        # Download best model
        best_pipe = pipelines[best_name]
        buffer = BytesIO()
        joblib.dump({'model_pipeline': best_pipe, 'label_encoder': label_encoder}, buffer)
        buffer.seek(0)
        st.download_button(
            label="📥 Download Best Model",
            data=buffer,
            file_name='car_eval_model.joblib',
            mime='application/octet-stream'
        )

    # Manual prediction form
    if 'best_name' in st.session_state:
        st.subheader("Manual Prediction")
        with st.form("prediction_form"):
            buying = st.selectbox("Buying price", ['vhigh', 'high', 'med', 'low'])
            maint = st.selectbox("Maintenance cost", ['vhigh', 'high', 'med', 'low'])
            doors = st.selectbox("Number of doors", ['2', '3', '4', '5more'])
            persons = st.selectbox("Person capacity", ['2', '4', 'more'])
            lug_boot = st.selectbox("Luggage boot size", ['small', 'med', 'big'])
            safety = st.selectbox("Safety rating", ['low', 'med', 'high'])
            submitted = st.form_submit_button("Predict Acceptability")
        if submitted:
            input_df = pd.DataFrame([{
                'buying': buying,
                'maint': maint,
                'doors': doors,
                'persons': persons,
                'lug_boot': lug_boot,
                'safety': safety
            }])
            best_pipe = st.session_state['pipelines'][st.session_state['best_name']]
            pred_num = best_pipe.predict(input_df)[0]
            pred_label = st.session_state['label_encoder'].inverse_transform([pred_num])[0]
            st.write(f"### Predicted Acceptability: **{pred_label}**")

if __name__ == '__main__':
    main()