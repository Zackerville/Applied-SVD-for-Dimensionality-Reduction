import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pdfplumber
import docx

def read_file(file, file_type):
    if file_type == "CSV":
        df = pd.read_csv(file)
        df_numeric = df.select_dtypes(include=[np.number])
        return df_numeric.values, df
    elif file_type == "EXCEL":
        df = pd.read_excel(file)
        df_numeric = df.select_dtypes(include=[np.number])
        return df_numeric.values, df
    elif file_type == "NUMPY":
        arr = np.load(file)
        if isinstance(arr, np.lib.npyio.NpzFile):
            arr = arr[arr.files[0]]
        return arr, arr
    elif file_type == "IMAGE":
        img = Image.open(file).convert('L')
        arr = np.array(img)
        return arr, img
    elif file_type == "TXT":
        text = file.read().decode("utf-8")
        docs = [line.strip() for line in text.split('\n') if line.strip()]
        return docs, text
    elif file_type == "PDF":
        with pdfplumber.open(file) as pdf:
            docs = [page.extract_text() for page in pdf.pages if page.extract_text()]
        return docs, "\n\n".join(docs)
    elif file_type == "WORD":
        doc = docx.Document(file)
        docs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
        return docs, "\n\n".join(docs)
    else:
        st.error("Unsupported file type!")
        return None, None

def perform_svd(data):
    data = np.array(data, dtype=float)
    U, S, VT = np.linalg.svd(data, full_matrices=False)
    return U, S, VT

def reconstruct(U, S, VT, k):
    S_k = np.diag(S[:k])
    U_k = U[:, :k]
    VT_k = VT[:k, :]
    return np.dot(U_k, np.dot(S_k, VT_k))

def plot_singular_values(S):
    fig, ax = plt.subplots()
    ax.plot(S, 'o-')
    ax.set_title("Singular Values")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    st.pyplot(fig)

def compute_rmse(original, reconstructed):
    return np.sqrt(np.mean((original - reconstructed) ** 2))

def get_file_type(file):
    ext = file.name.split('.')[-1].lower()
    if ext in ['csv']:
        return "CSV"
    if ext in ['xlsx', 'xls']:
        return "EXCEL"
    if ext in ['npy', 'npz']:
        return "NUMPY"
    if ext in ['png', 'jpg', 'jpeg', 'bmp', 'tiff']:
        return "IMAGE"
    if ext in ['txt']:
        return "TXT"
    if ext in ['pdf']:
        return "PDF"
    if ext in ['docx']:
        return "WORD"
    return "UNKNOWN"

def text_to_matrix(docs):
    # docs: list các documents (VD: mỗi trang, mỗi đoạn)
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(docs)
    return X

st.set_page_config(page_title="SVD Dimensionality Reduction App", layout="wide")
st.title("SVD Dimensionality Reduction App")

st.markdown("""
### 1. Upload your file
""")

file = st.file_uploader("Choose a file...", type=['csv', 'xlsx', 'xls', 'npy', 'npz', 'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'txt', 'pdf', 'docx'])

if file:
    file_type = get_file_type(file)
    data, raw = read_file(file, file_type)
    if data is not None:
        st.write("**Preview original data:**")
        if file_type in ["CSV", "EXCEL", "NUMPY"]:
            st.dataframe(pd.DataFrame(data))
        elif file_type == "IMAGE":
            st.image(raw, caption="Original Image", use_column_width=True)
        elif file_type in ["TXT", "PDF", "WORD"]:
            st.text_area("File Text Content", value=raw[:5000] + ("..." if len(raw) > 5000 else ""), height=200)
            # Chuyển text thành ma trận số (TF-IDF sparse matrix)
            X = text_to_matrix(data)
            st.write("TF-IDF matrix shape:", X.shape)
        else:
            st.warning("Cannot preview this file type.")

        st.markdown("---")
        st.subheader("2. SVD Analysis")
        # Kiểm tra để dùng TruncatedSVD hoặc np.linalg.svd
        if file_type in ["TXT", "PDF", "WORD"]:
            min_dim = min(X.shape)
            if min_dim > 1:
                k = st.slider(
                    "Select number of dimensions/components to keep",
                    min_value=1, max_value=min(100, min_dim), value=min(10, min_dim)
                )
                svd = TruncatedSVD(n_components=k)
                data_k = svd.fit_transform(X)
                S = svd.singular_values_
                st.write(f"Reduced matrix shape: {data_k.shape}")
                plot_singular_values(S)
            else:
                st.warning("Không đủ documents hoặc số chiều để giảm chiều.")
                st.stop()
        else:
            U, S, VT = perform_svd(data)
            st.write(f"Shape: U {U.shape}, S {S.shape}, VT {VT.shape}")
            plot_singular_values(S)
            min_dim = min(data.shape)
            if min_dim > 1:
                k = st.slider(
                    "Select number of dimensions/components to keep",
                    min_value=1, max_value=min_dim, value=min(10, min_dim)
                )
                data_k = reconstruct(U, S, VT, k)
            else:
                st.warning("Dữ liệu chỉ có 1 chiều (1 hàng hoặc 1 cột). Không thể giảm chiều với SVD.")
                st.stop()

        st.markdown("---")
        st.subheader("3. Result after dimensionality reduction")
        if file_type in ["CSV", "EXCEL", "NUMPY"]:
            st.write("Reduced Data (reconstructed):")
            st.dataframe(np.round(data_k, 3))
        elif file_type == "IMAGE":
            st.image(np.clip(data_k, 0, 255).astype(np.uint8), caption=f"Reconstructed Image (k={k})", use_column_width=True)
        elif file_type in ["TXT", "PDF", "WORD"]:
            st.write("Reduced TF-IDF matrix (after SVD):")
            st.dataframe(np.round(data_k, 3))
        else:
            st.write("Cannot display this file type after reduction.")

        st.markdown("---")
        st.subheader("4. Evaluation Metrics")
        # RMSE/Compression ratio chỉ có ý nghĩa khi dữ liệu là dense
        if file_type in ["CSV", "EXCEL", "NUMPY", "IMAGE"]:
            rmse = compute_rmse(data, data_k)
            st.write(f"**RMSE (Root Mean Squared Error):** {rmse:.4f}")
            compression_ratio = (k * (data.shape[0] + data.shape[1] + 1)) / data.size
            st.write(f"**Compression Ratio:** {compression_ratio:.4f} (lower = better)")
        elif file_type in ["TXT", "PDF", "WORD"]:
            st.write(f"Reduced TF-IDF shape: {data_k.shape}")

        st.markdown("---")
        st.subheader("5. Download reconstructed data")
        csv_buffer = BytesIO()
        pd.DataFrame(data_k).to_csv(csv_buffer, index=False)
        st.download_button("Download CSV", csv_buffer.getvalue(), file_name="reconstructed.csv")
