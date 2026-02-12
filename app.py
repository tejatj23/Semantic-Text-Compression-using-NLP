import streamlit as st
import zlib
from huffman import huffman_compress, huffman_decompress
from compressor import semantic_compress
import pandas as pd

st.set_page_config(page_title="Semantic Text Compression", layout="wide")
st.title(" Semantic Text Compression Using NLP")

# --- Input
uploaded = st.file_uploader("Upload a .txt file", type=["txt"])
text_input = st.text_area("Enter text:")

threshold = st.slider("Similarity Threshold", min_value=0.5, max_value=0.95, value=0.75, step=0.01)

# To store semantic compressed text
if "sem_text" not in st.session_state:
    st.session_state["sem_text"] = ""

# --- Run SBERT Compression
if st.button("SBERT Compression"):
    if uploaded:
        text = uploaded.read().decode("utf-8")
    else:
        text = text_input

    if not text.strip():
        st.warning("Please enter or upload some text.")
    else:
        sem_text = semantic_compress(text, threshold=threshold)
        st.session_state["sem_text"] = sem_text
        st.session_state["original_text"] = text
        st.success(" SBERT Semantic Compression Completed.")
        st.subheader("Compressed Semantic Text")
        st.text(sem_text)

# --- Run Huffman Compression
if st.button("Huffman Compression"):
    sem_text = st.session_state.get("sem_text", "")
    if not sem_text:
        st.warning("Please run SBERT compression first.")
    else:
        sem_bytes = sem_text.encode("utf-8")
        original_size = len(sem_bytes)

        h_bytes, code_map, padding, total_bits = huffman_compress(sem_text)
        h_size = len(h_bytes)
        compression_rate = (1 - h_size / original_size) * 100

        st.subheader(" Huffman Compression Results")
        st.write(f"Original (Semantic Text) Size: {original_size} bytes")
        st.write(f"Huffman Compressed Size: {h_size} bytes")
        st.write(f"Compression Rate: {compression_rate:.2f}%")

        with st.expander("Huffman Decompressed Text"):
            st.text(huffman_decompress(h_bytes, code_map, padding, total_bits))

# --- Run zlib Compression
if st.button("zlib Compression"):
    sem_text = st.session_state.get("sem_text", "")
    if not sem_text:
        st.warning("Please run SBERT compression first.")
    else:
        sem_bytes = sem_text.encode("utf-8")
        original_size = len(sem_bytes)

        z_bytes = zlib.compress(sem_bytes)
        z_size = len(z_bytes)
        compression_rate = (1 - z_size / original_size) * 100

        st.subheader(" zlib Compression Results")
        st.write(f"Original (Semantic Text) Size: {original_size} bytes")
        st.write(f"zlib Compressed Size: {z_size} bytes")
        st.write(f"Compression Rate: {compression_rate:.2f}%")

        with st.expander("zlib Decompressed Text"):
            st.text(zlib.decompress(z_bytes).decode("utf-8"))

# --- Show Summary Table
if st.button("Show Compression Summary Table"):
    sem_text = st.session_state.get("sem_text", "")
    text = st.session_state.get("original_text", "")
    if not sem_text or not text:
        st.warning("Please run SBERT compression first.")
    else:
        original_size = len(text.encode("utf-8"))
        sem_size = len(sem_text.encode("utf-8"))
        sbert_rate = (1 - sem_size / original_size) * 100

        # Huffman
        h_bytes, code_map, padding, total_bits = huffman_compress(sem_text)
        h_size = len(h_bytes)
        h_rate = (1 - h_size / sem_size) * 100

        # zlib
        z_bytes = zlib.compress(sem_text.encode("utf-8"))
        z_size = len(z_bytes)
        z_rate = (1 - z_size / sem_size) * 100

        summary = pd.DataFrame({
            "Method": ["SBERT Semantic", "Huffman", "zlib"],
            "Original Size (bytes)": [original_size, sem_size, sem_size],
            "Compressed Size (bytes)": [sem_size, h_size, z_size],
            "Compression Rate (%)": [f"{sbert_rate:.2f}", f"{h_rate:.2f}", f"{z_rate:.2f}"]
        })

        st.subheader("Compression Summary Table")
        st.table(summary)
