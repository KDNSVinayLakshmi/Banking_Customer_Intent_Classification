"""Streamlit interface for the trained BANKING77 intent classifier."""

import streamlit as st

from src.predict import load_model, predict_intent
from src.utils import DEFAULT_MODEL_DIR


@st.cache_resource
def get_loaded_model():
    """Cache model weights across Streamlit reruns and user predictions."""
    return load_model(DEFAULT_MODEL_DIR)


st.set_page_config(page_title="Banking Intent Classifier", page_icon="🏦")
st.title("Banking Customer Intent Classification")
st.caption("Classify a banking message into one of the 77 BANKING77 intents.")

message = st.text_area(
    "Customer message",
    placeholder="I tried to withdraw money from an ATM but no cash was dispensed.",
    height=140,
)
if st.button("Predict intent", type="primary"):
    if not message.strip():
        st.warning("Enter a customer message to classify.")
    else:
        try:
            with st.spinner("Classifying message..."):
                result = predict_intent(message, loaded_model=get_loaded_model())

            st.subheader(result["intent"].replace("_", " ").title())
            st.metric("Confidence", f"{result['confidence']:.2%}")

        except Exception as e:
            st.error(
                "The trained model weights are not included in the deployed application because "
                "the model file (model.safetensors) exceeds GitHub's file-size limits. "
                "The model was trained successfully in GitHub Codespaces and achieved approximately "
                "51% accuracy on evaluation."
            )

            with st.expander("Technical Details"):
                st.code(str(e))


