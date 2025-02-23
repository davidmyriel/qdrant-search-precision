import streamlit as st
from datasets import load_dataset
from qdrant_client import QdrantClient, models
import time
from typing import List, Dict, Any

# Constants
VECTOR_SIZE = 768
BATCH_SIZE = 250  # Increased batch size for faster uploads
COLLECTION_NAME = "arxiv-titles-instructorxl-embeddings"

# Page config
st.set_page_config(
    page_title="Vector Search Quality Demo | Qdrant",
    page_icon="🎯",
    layout="wide"
)

# Header with hero image
st.image("hero.jpg", use_container_width=True)
st.subheader("Measure and optimize your search precision with Qdrant")

# Initialize Qdrant client
@st.cache_resource
def get_qdrant_client() -> QdrantClient:
    """Initialize and return Qdrant client."""
    return QdrantClient("localhost", port=6333)

@st.cache_data
def load_data(num_train: int, num_test: int) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load and cache dataset splits."""
    dataset = load_dataset(
        "Qdrant/arxiv-titles-instructorxl-embeddings", 
        split="train", 
        streaming=True
    )
    dataset_iterator = iter(dataset)
    
    # Use list comprehension for better performance
    train_dataset = [next(dataset_iterator) for _ in range(num_train)]
    test_dataset = [next(dataset_iterator) for _ in range(num_test)]
    
    return train_dataset, test_dataset

def init_collection(client: QdrantClient, hnsw_m: int, hnsw_ef: int) -> None:
    """Initialize Qdrant collection with given parameters."""
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE,
        ),
        hnsw_config=models.HnswConfigDiff(
            m=hnsw_m,
            ef_construct=hnsw_ef,
        )
    )

def calculate_precision(client: QdrantClient, test_item: Dict[str, Any], k: int) -> float:
    """Calculate precision@k for a single test item."""
    # Get both approximate and exact search results
    ann_ids = {
        point.id for point in client.query_points(
            collection_name=COLLECTION_NAME,
            query=test_item["vector"],
            limit=k
        ).points
    }
    
    knn_ids = {
        point.id for point in client.query_points(
            collection_name=COLLECTION_NAME,
            query=test_item["vector"],
            limit=k,
            search_params=models.SearchParams(exact=True)
        ).points
    }
    
    return len(ann_ids.intersection(knn_ids)) / k

def main():    
    # Sidebar settings
    with st.sidebar:
        st.header("First, configure your HNSW parameters:")
        k_value = st.slider(
            "K value for Precision@K",
            1, 20, 5,
            help="The number of top results to compare between approximate and exact search. "
            "Precision@K measures how many of the approximate top-K results match the exact top-K results."
        )
        
        num_train = st.number_input(
            "Number of training samples", 
            min_value=1000,
            max_value=100_000,
            value=10_000,
            step=1000,
            help="The number of vectors to index in the collection. "
            "These vectors will be used to build the HNSW graph for approximate search."
        )
        
        num_test = st.number_input(
            "Number of test samples",
            min_value=100,
            max_value=10_000,
            value=500,
            step=100,
            help="The number of test queries to evaluate search quality. "
            "Each test vector will be used to compare approximate vs exact search results."
        )
        
        hnsw_m = st.number_input(
            "HNSW M parameter", 
            8, 64, 16,
            help="The number of edges per node in the HNSW graph. "
            "Larger values increase search precision but require more memory and indexing time. "
            "Default is 16."
        )
        
        hnsw_ef = st.number_input(
            "HNSW EF Construct",
            50, 400, 100,
            help="The size of the dynamic candidate list during HNSW graph construction. "
            "Larger values increase search precision but require longer indexing time. "
            "Default is 100."
        )
    
    client = get_qdrant_client()
    
    if st.button("Run Evaluation", type="primary"):
        try:
            # Load dataset
            with st.spinner("Loading dataset..."):
                train_dataset, test_dataset = load_data(num_train, num_test)
                st.success("Dataset loaded!")

            # Initialize collection
            with st.spinner("Initializing collection..."):
                init_collection(client, hnsw_m, hnsw_ef)
                st.success("Collection initialized!")

            # Upload points with progress bar
            progress_bar = st.progress(0)
            points_uploaded = 0
            
            with st.spinner("Uploading points..."):
                for i in range(0, len(train_dataset), BATCH_SIZE):
                    batch = train_dataset[i:i + BATCH_SIZE]
                    points = [
                        models.PointStruct(
                            id=item["id"],
                            vector=item["vector"],
                            payload=item
                        )
                        for item in batch
                    ]
                    
                    client.upload_points(
                        collection_name=COLLECTION_NAME,
                        points=points
                    )
                    
                    points_uploaded += len(batch)
                    progress_bar.progress(min(1.0, points_uploaded / len(train_dataset)))

            # Wait for indexing
            with st.spinner("Waiting for indexing to complete..."):
                while True:
                    collection_info = client.get_collection(COLLECTION_NAME)
                    if collection_info.status == models.CollectionStatus.GREEN:
                        break
                    time.sleep(1)
                st.success("Indexing complete!")

            # Calculate precision
            with st.spinner(f"Calculating Precision@{k_value}..."):
                test_progress = st.progress(0)
                precisions = []
                
                for i, item in enumerate(test_dataset):
                    precision = calculate_precision(client, item, k_value)
                    precisions.append(precision)
                    test_progress.progress(min(1.0, (i + 1) / len(test_dataset)))

                avg_precision = sum(precisions) / len(precisions)

            # Display results
            st.header("Results")
            st.metric(
                label=f"Average Precision@{k_value}", 
                value=f"{avg_precision:.4f}"
            )
            
            # Add recommendations section in a box
            st.header("Recommendations")
            with st.container():
                st.markdown("---")  # Horizontal line for visual separation
                
                # Create columns for better layout
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Current Configuration")
                    st.write(f"""
                    - HNSW M parameter: {hnsw_m} (default: 16)
                    - HNSW EF Construct: {hnsw_ef} (default: 100)
                    - Number of vectors: {num_train}
                    """)
                
                with col2:
                    st.subheader("Optimization Tips")
                    if avg_precision < 0.95:
                        st.write("""
                        To improve precision:
                        - Increase M parameter (requires more memory)
                        - Increase EF Construct (requires longer indexing time)
                        """)
                    elif avg_precision > 0.99:
                        st.write("""
                        Current precision is excellent! Consider:
                        - Reducing M or EF if faster indexing is needed
                        - Keep current settings if memory and indexing time are acceptable
                        """)
                    else:
                        st.write("""
                        Good precision! Possible optimizations:
                        - Slightly increase M or EF for better precision
                        - Current settings might be optimal for speed/accuracy trade-off
                        """)
                        
                st.info("""
                    Remember: The quality of embeddings is the most important factor in search quality. 
                    HNSW parameters help optimize the speed-accuracy trade-off, but cannot improve the underlying embedding quality.
                """)
                st.markdown("---")  # Horizontal line for visual separation
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 