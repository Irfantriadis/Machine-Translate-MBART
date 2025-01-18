import streamlit as st
from transformers import MBartForConditionalGeneration, MBartTokenizer
import time

# Set the page config as the first Streamlit command
st.set_page_config(
    page_title="Indo-Ngapak Translator",
    layout="centered"
)

# Custom CSS for dark theme
st.markdown("""
    <style>
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #333333;
        background-color: #2D2D2D;
        color: #FFFFFF;
    }
    .stButton>button {
        border-radius: 20px;
        padding: 10px 24px;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        border-color: #ff3333;
    }
    .success-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #2D2D2D;
        margin: 10px 0;
        border: 1px solid #333333;
    }
    .info-box {
        background-color: #2D2D2D;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #333333;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the fine-tuned model
@st.cache_resource
def load_model():
    model = MBartForConditionalGeneration.from_pretrained("model/indojawa2")
    tokenizer = MBartTokenizer.from_pretrained("model/indojawa2")
    return model, tokenizer

def translate_text(text, model, tokenizer):
    """Function to handle the translation process"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Simulate translation progress
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)
        status_text.text(f"Menerjemahkan... {i+1}%")

    # Actual translation
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    output_ids = model.generate(inputs["input_ids"])
    translation = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Remove progress bar and status text
    progress_bar.empty()
    status_text.empty()
    
    return translation

def main():
    st.title("Indonesia ke Ngapak Translator")
    st.markdown("""
    <div class='info-box'>
    Masukkan teks berbahasa Indonesia di bawah ini dan klik tombol <b>Terjemahkan</b> untuk 
    mendapatkan hasil terjemahan ke Bahasa Ngapak.
    </div>
    """, unsafe_allow_html=True)

    # Initialize model
    model, tokenizer = load_model()

    # Initialize session states
    if 'translation' not in st.session_state:
        st.session_state.translation = ""
    if 'button_clicked' not in st.session_state:
        st.session_state.button_clicked = False

    # Input text area with character counter
    source_text = st.text_area(
        "Masukkan teks berbahasa Indonesia",
        height=150,
        max_chars=512,
        help="Maksimal 512 karakter"
    )
    
    # Character counter
    remaining_chars = 512 - len(source_text)
    st.caption(f"Sisa karakter: {remaining_chars}")

    def handle_translation():
        """Callback function for the translate button"""
        if source_text.strip():
            st.session_state.translation = translate_text(source_text, model, tokenizer)
        else:
            st.warning("Harap masukkan teks untuk diterjemahkan.")

    # Center the translate button with persistent text
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        translate_button = st.empty()
        translate_button.button(
            "Terjemahkan",
            on_click=handle_translation,
            use_container_width=True,
            key="translate_button"
        )

    # Always display translation if it exists in session state
    if st.session_state.translation:
        st.markdown("""
            <div class='success-box'>
                <h3 style='color: #ffffff;'>Hasil Terjemahan:</h3>
                <p style='font-size: 1em;'>{}</p>
            </div>
            """.format(st.session_state.translation), unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666;'>
        <p>Dikembangkan oleh Teknik Informatika PHB</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
