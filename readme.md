# Vector Search Quality Assessment

![Precision visualization](precision.gif)

## What Makes for Quality Vector Search?
Vector search quality depends on two key components: **embedding quality** and **ANN algorithm performance**. Embedding quality is typically evaluated through benchmarks like MTEB using ground truth datasets and exact kNN search, isolating the embedding performance from retrieval approximations. This foundational quality determines how well vectors represent semantic relationships in the underlying space.

## This Evaluation is About ANN Performance
The practical performance of vector search systems relies heavily on Approximate Nearest Neighbors (ANN) algorithms, which trade perfect accuracy for speed. In Qdrant, this quality-speed tradeoff is primarily controlled through HNSW parameters - `m` (edges per node, default 16) and `ef_construct` (neighbors during index building, default 100). Higher values increase precision but require more memory and indexing time. The effectiveness of these approximations is measured using Precision@k, calculated as `|ANN_results ∩ exact_kNN_results| / k`.

## How This Evaluation Works: Methodology
The evaluation process involves creating a test collection, indexing training data, and comparing ANN search results against exact search results enabled via `search_params=models.SearchParams(exact=True)`. This allows direct measurement of how well the ANN algorithm approximates exact search results. While HNSW typically achieves high precision (often >99%), specific use cases may require tuning these parameters to optimize the trade-off between search quality and performance.

[Source: Qdrant Documentation - Retrieval Quality](https://qdrant.tech/documentation/beginner-tutorials/retrieval-quality/)

## Prerequisites
- Python 3.8+
- Virtual environment running
- Qdrant running locally (Docker or binary)

## Running the Demo
1. Start Qdrant:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the demo:
```bash
streamlit run app.py
```