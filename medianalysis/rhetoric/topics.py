import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
from umap import UMAP
import json


def fit_topics(
    cuerpo_csv: str,
    output_docs_csv: str,    # topic asignado por documento
    output_topics_csv: str,  # keywords por topic
    nr_topics: int = "auto",
):
    df = pd.read_csv(cuerpo_csv)
    docs = df["body"].tolist()

    # Mismos modelos que el resto del pipeline
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # UMAP y HDBSCAN explícitos para poder controlar parámetros
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=10,   # mínimo 10 docs para formar un topic
        min_samples=1,
        metric="euclidean",    # UMAP ya redujo dimensionalidad
        prediction_data=True,  # necesario para asignar topics a docs nuevos
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics=nr_topics,
        language="multilingual",
        calculate_probabilities=False,
        verbose=True,
    )

    topics, _ = topic_model.fit_transform(docs)

    # ── Output 1: topic por documento ────────────────────────
    docs_df = pd.DataFrame({
        "id":       df["id"],
        "topic_id": topics,   # -1 = outlier
    })
    docs_df.to_csv(output_docs_csv, index=False)

    # ── Output 2: keywords por topic ─────────────────────────
    topic_info = topic_model.get_topic_info()
    topic_rows = []
    for _, row in topic_info.iterrows():
        if row["Topic"] == -1:
            continue
        keywords = [word for word, _ in topic_model.get_topic(row["Topic"])]
        topic_rows.append({
            "topic_id":    row["Topic"],
            "count":       row["Count"],
            "keywords":    json.dumps(keywords, ensure_ascii=False),
            "label":       None,   # se llena manualmente después
        })

    pd.DataFrame(topic_rows).to_csv(output_topics_csv, index=False)

    print(f"{len(topic_rows)} topics encontrados")
    print(f"Outliers: {sum(1 for t in topics if t == -1)}/{len(topics)}")

    return topic_model   # devuelve el modelo para poder reusar


def assign_topics(
    topic_model: BERTopic,
    cuerpo_csv: str,
    output_csv: str,
):
    """
    Asigna topics a documentos nuevos sin re-entrenar.
    Útil cuando el corpus crece.
    """
    df = pd.read_csv(cuerpo_csv)
    topics, _ = topic_model.transform(df["body"].tolist())
    pd.DataFrame({
        "id":       df["id"],
        "topic_id": topics,
    }).to_csv(output_csv, index=False)