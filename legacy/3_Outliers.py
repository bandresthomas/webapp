import os
import streamlit as st
from PIL import Image

# -----------------------------
# Constants and configuration
# -----------------------------
IMAGES_DIR_NAME: str = "images"
OUTLIERS_IMAGES = {
    "QB": "qb_outliers.png",
    "RB Tiers 1-5": "rb_outliers_tiers_1_5.png", 
    "RB Tiers 6-10": "rb_outliers_tiers_6_10.png",
    "WR Tiers 1-5": "wr_outliers_tiers_1_5.png",
    "WR Tiers 6-10": "wr_outliers_tiers_6_10.png", 
    "TE": "te_outliers.png"
}


def _repo_root_dir() -> str:
    """Get the repository root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _images_dir() -> str:
    """Get the images directory path."""
    return os.path.join(_repo_root_dir(), IMAGES_DIR_NAME)


def _load_outliers_image(image_name: str) -> Image.Image:
    """Load an outliers image from the images directory."""
    image_path = os.path.join(_images_dir(), image_name)
    if os.path.exists(image_path):
        return Image.open(image_path)
    else:
        st.error(f"Imagem não encontrada: {image_path}")
        return None


def main() -> None:
    st.set_page_config(
        page_title="Outliers - Fantasy Football 2025",
        page_icon="📊",
        layout="wide"
    )
    
    # Header
    st.title("📊 Análise de Outliers - Fantasy Football 2025")
    
    # Create tabs for different positions
    tab1, tab2, tab3, tab4 = st.tabs(["🏈 Quarterbacks", "🏃‍♂️ Running Backs", "🤲 Wide Receivers", "🤏 Tight Ends"])
    
    with tab1:
        qb_image = _load_outliers_image(OUTLIERS_IMAGES["QB"])
        if qb_image:
            st.image(qb_image, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            rb_tiers_1_5_image = _load_outliers_image(OUTLIERS_IMAGES["RB Tiers 1-5"])
            if rb_tiers_1_5_image:
                st.image(rb_tiers_1_5_image, use_container_width=True)
        
        with col2:
            rb_tiers_6_10_image = _load_outliers_image(OUTLIERS_IMAGES["RB Tiers 6-10"])
            if rb_tiers_6_10_image:
                st.image(rb_tiers_6_10_image, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            wr_tiers_1_5_image = _load_outliers_image(OUTLIERS_IMAGES["WR Tiers 1-5"])
            if wr_tiers_1_5_image:
                st.image(wr_tiers_1_5_image, use_container_width=True)
        
        with col2:
            wr_tiers_6_10_image = _load_outliers_image(OUTLIERS_IMAGES["WR Tiers 6-10"])
            if wr_tiers_6_10_image:
                st.image(wr_tiers_6_10_image, use_container_width=True)
    
    with tab4:
        te_image = _load_outliers_image(OUTLIERS_IMAGES["TE"])
        if te_image:
            st.image(te_image, use_container_width=True)
    



if __name__ == "__main__":
    main()
