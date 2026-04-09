import json
import numpy as np
import pandas as pd
import hdbscan


def _load_valid_embeddings(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Filtra embeddings fallidos y devuelve df + array numpy."""
    df = df[df["embedding"].apply(lambda e: json.loads(e) != [None])].copy()
    embeddings = np.array([json.loads(e) for e in df["embedding"]])
    return df.reset_index(drop=True), embeddings


def _hdbscan_labels(embeddings: np.ndarray, prefix: str, min_cluster_size: int) -> list:
    labels = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric="cosine"
    ).fit_predict(embeddings)
    return [f"{prefix}__{l}" if l != -1 else None for l in labels]


def _print_stats(rows: list, label: str):
    total      = len(rows)
    n_noise    = sum(1 for r in rows if list(r.values())[-1] is None)
    n_clusters = len(set(list(r.values())[-1] for r in rows if list(r.values())[-1]))
    print(f"{n_clusters} {label} | {n_noise}/{total} como ruido")


def cluster_generic(
    embeddings_csv: str,
    id_col: str,
    output_csv: str,
    prefix: str,
    min_cluster_size: int = 2,
    group_by_col: str | None = None,  # si se especifica, clustea por grupos
    group_source_df: pd.DataFrame | None = None,
):
    """
    Función genérica de clustering.
    
    - Sin group_by_col: clustea todos los embeddings juntos (relaciones, eventos)
    - Con group_by_col: clustea por subgrupos (menciones por tipo de entidad)
    """
    df, embeddings = _load_valid_embeddings(pd.read_csv(embeddings_csv))
    rows = []

    if group_by_col is None:
        # Clustering global
        labels = _hdbscan_labels(embeddings, prefix, min_cluster_size)
        for idx, (_, row) in enumerate(df.iterrows()):
            rows.append({id_col: row[id_col], "cluster_id": labels[idx]})

    else:
        # Clustering por grupo — necesita columna extra desde source_df
        df = df.merge(group_source_df[[id_col, group_by_col]], on=id_col, how="left")

        for group_val in df[group_by_col].unique():
            mask    = df[group_by_col] == group_val
            sub_df  = df[mask].reset_index(drop=True)
            sub_emb = embeddings[mask]

            if len(sub_df) < 2:
                for _, row in sub_df.iterrows():
                    rows.append({id_col: row[id_col], "cluster_id": None})
                continue

            labels = _hdbscan_labels(sub_emb, f"{prefix}_{group_val}", min_cluster_size)
            for idx, (_, row) in enumerate(sub_df.iterrows()):
                rows.append({id_col: row[id_col], "cluster_id": labels[idx]})

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    _print_stats(rows, prefix)