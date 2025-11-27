
import streamlit as st
import numpy as np, cv2, pickle, tensorflow as tf
from PIL import Image
from mtcnn import MTCNN

@st.cache_resource
def load_all():
    lugares = tf.keras.models.load_model("modelo_clasificador.h5")
    faces = tf.keras.models.load_model("modelo_faces.keras")
    with open("embeddings.pkl", "rb") as f:
        personas = pickle.load(f)
    detector = MTCNN()
    clases = ["Biblioteca", "Baños", "Administrativo", "Aula", "Cafetería"]  # ← cámbialos por tus clases reales si quieres
    return lugares, faces, personas, detector, clases

modelo_lugares, modelo_faces, personas, detector, clases = load_all()

st.set_page_config(page_title="UDENAR ID", page_icon="🏛️", layout="centered")
st.title("UDENAR ID")
st.markdown("<h3 style='color:#006633;text-align:center;'>Universidad de Nariño</h3>", True)

if "user" not in st.session_state:
    st.session_state.user = None

if not st.session_state.user:
    st.markdown("### Inicia sesión con tu rostro")
    file = st.file_uploader("Sube una selfie", ["jpg","jpeg","png"])
    if file:
        img = cv2.imdecode(np.frombuffer(file.getvalue(), np.uint8), 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb)
        if faces:
            x,y,w,h = faces[0]['box']
            rostro = cv2.resize(rgb[max(0,y):y+h, max(0,x):x+w], (100,100))[None,...]/255.0
            pred = modelo_faces.predict(rostro, verbose=0)[0]
            if pred.max() > 0.7:
                st.session_state.user = personas[pred.argmax()]
                st.success(f"¡Bienvenid@ {st.session_state.user}!")
                st.balloons()
            else:
                st.error("Confianza baja, intenta otra selfie")
        else:
            st.error("No detecté rostro")
else:
    st.success(f"¡Hola {st.session_state.user}!")
    if st.button("Cerrar sesión"):
        st.session_state.user = None
        st.rerun()

    st.markdown("### ¿En qué lugar estás?")
    file2 = st.file_uploader("Sube foto del lugar", ["jpg","jpeg","png"], key=2)
    if file2:
        img = Image.open(file2).resize((224,224))
        st.image(img, width=350)
        arr = np.array(img)[None,...]/255.0
        pred = modelo_lugares.predict(arr, verbose=0)[0]
        lugar = clases[pred.argmax()]
        st.markdown(f"### ¡Estás en **{lugar}**!")
        st.progress(float(pred.max()))

st.caption("© 2025 UDENAR ID")
