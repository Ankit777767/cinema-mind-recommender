# 🧠 Cinema Mind: Multimodal Emotion-Aware Recommender

Traditional movie recommendation systems rely on collaborative filtering (matching what similar users clicked). Cinema Mind is a modern, multimodal AI search engine that recommends movies based on **emotional resonance and visual tone**.

Instead of searching by genre, users search by *feeling* (e.g., "A visually stunning sci-fi that feels melancholic but ends with hope").

## 🔗 Live Application Demo
Explore the live, cloud-deployed AI model here: **[https://cinema-mind-recommender-by-ankit.streamlit.app/]**

## 🚀 Architecture & Tech Stack
* **Natural Language Processing:** `HuggingFace (all-MiniLM-L6-v2)` translates movie plot summaries and user prompts into 384-dimensional dense vectors.
* **Computer Vision:** A customized `ResNet-50` (with the classification head removed) extracts 2048-dimensional visual feature vectors from movie posters to understand color palettes and visual mood.
* **Vector Database:** `Qdrant` stores these multimodal Named Vectors and executes blazing-fast Cosine Similarity searches.
* **Frontend:** `Streamlit` provides a clean, responsive web interface.
* **Data Ingestion:** Custom Python pipelines gracefully handle rate limits, checkpointing, and pagination from the TMDB API.

## 🛠️ Local Setup Instructions
1. Clone this repository.
2. Install the required dependencies: `pip install -r requirements.txt`
3. Create a `.env` file in the root directory and add your TMDB API key: `TMDB_API_KEY=your_key_here`
4. Run the data fetcher: `python src/data_pipeline/data_fetcher.py`
5. Run the poster downloader: `python src/data_pipeline/download_posters.py`
6. Generate Text Math: `python src/embedding_engine/text_embedder.py`
7. Generate Visual Math: `python src/vision_module/image_embedder.py`
8. Build the Qdrant DB: `python src/vector_db/build_index.py`
9. Launch the App: `streamlit run src/frontend/app.py`