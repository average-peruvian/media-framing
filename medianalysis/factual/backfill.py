import json
import pandas as pd


def backfill(
    extracciones_csv: str,
    menciones_raw_csv: str,
    clusters_csv: str,
    entidades_csv: str,
    relaciones_raw_csv: str,
    relaciones_clusters_csv: str,
    tipos_relacion_csv: str,
    eventos_raw_csv: str,
    eventos_clusters_csv: str,
    tipos_evento_csv: str,
    output_grafo_csv: str,
    output_eventos_csv: str,
):
    # ── Cargar tablas de resolución ───────────────────────────

    # mention_id → entity_id canónico
    mention_to_entity = _build_mention_to_entity(
        menciones_raw_csv, clusters_csv, entidades_csv
    )

    # relation_id → type_id canónico
    relation_to_type = _build_id_to_type(
        relaciones_raw_csv, relaciones_clusters_csv, tipos_relacion_csv,
        id_col="relation_id"
    )

    # event_id → type_id canónico
    event_to_type = _build_id_to_type(
        eventos_raw_csv, eventos_clusters_csv, tipos_evento_csv,
        id_col="event_id"
    )

    # ── Construir grafo de relaciones ─────────────────────────

    rel_rows = []
    evt_rows = []

    for _, doc in pd.read_csv(extracciones_csv).iterrows():
        doc_id = doc["id"]

        # Relaciones
        for rel in json.loads(doc["relations"]):
            subj_mention = f"{doc_id}__{rel['subject']}"
            obj_mention  = f"{doc_id}__{rel['object']}"
            rel_id       = f"{doc_id}__{rel['id']}"

            subj      = mention_to_entity.get(subj_mention)
            obj       = mention_to_entity.get(obj_mention)
            type_id   = relation_to_type.get(rel_id)

            # Solo entra al grafo si todo está resuelto
            if not all([subj, obj, type_id]):
                continue

            rel_rows.append({
                "subject":          subj,
                "relation_type_id": type_id,
                "object":           obj,
                "doc_id":           doc_id,
                "confidence":       rel["confidence"],
                "evidence":         rel["evidence"],
            })

        # Eventos
        for evt in json.loads(doc["events"]):
            evt_id  = f"{doc_id}__{evt['id']}"
            type_id = event_to_type.get(evt_id)

            if not type_id:
                continue

            # Resolver entity_ids en arguments
            resolved_args = {}
            for role, local_id in evt["arguments"].items():
                mention_id = f"{doc_id}__{local_id}"
                entity_id  = mention_to_entity.get(mention_id)
                if entity_id:
                    resolved_args[role] = entity_id

            evt_rows.append({
                "event_type_id": type_id,
                "doc_id":        doc_id,
                "trigger":       evt["trigger"],
                "arguments":     json.dumps(resolved_args, ensure_ascii=False),
                "confidence":    evt["confidence"],
            })

    # ── Guardar ───────────────────────────────────────────────

    grafo_df   = pd.DataFrame(rel_rows)
    eventos_df = pd.DataFrame(evt_rows)

    grafo_df.to_csv(output_grafo_csv,   index=False)
    eventos_df.to_csv(output_eventos_csv, index=False)

    print(f"{len(grafo_df)} tripletas en el grafo")
    print(f"{len(eventos_df)} eventos resueltos")

    # Stats de cobertura
    total_rel = sum(len(json.loads(doc["relations"])) for _, doc in pd.read_csv(extracciones_csv).iterrows())
    total_evt = sum(len(json.loads(doc["events"]))    for _, doc in pd.read_csv(extracciones_csv).iterrows())
    print(f"Cobertura relaciones: {len(grafo_df)}/{total_rel} ({100*len(grafo_df)//total_rel}%)")
    print(f"Cobertura eventos:    {len(eventos_df)}/{total_evt} ({100*len(eventos_df)//total_evt}%)")


# ── Helpers ───────────────────────────────────────────────────

def _build_mention_to_entity(
    menciones_raw_csv: str,
    clusters_csv: str,
    entidades_csv: str,
) -> dict:
    """mention_id → entity_id canónico."""
    menciones_df  = pd.read_csv(menciones_raw_csv)
    clusters_df   = pd.read_csv(clusters_csv)
    entidades_df  = pd.read_csv(entidades_csv)

    # cluster_id → entity_id
    cluster_to_entity = dict(zip(entidades_df["cluster_id"], entidades_df["entity_id"]))

    # mention_id → cluster_id → entity_id
    df = menciones_df.merge(clusters_df, on="mention_id", how="left")
    df["entity_id"] = df["cluster_id"].map(cluster_to_entity)

    return dict(zip(df["mention_id"], df["entity_id"]))


def _build_id_to_type(
    raw_csv: str,
    clusters_csv: str,
    tipos_csv: str,
    id_col: str,
) -> dict:
    """relation_id / event_id → type_id canónico."""
    raw_df     = pd.read_csv(raw_csv)[[id_col]]
    clusters_df = pd.read_csv(clusters_csv)
    tipos_df   = pd.read_csv(tipos_csv)

    # cluster_id → type_id
    cluster_to_type = dict(zip(tipos_df["cluster_id"], tipos_df["type_id"]))

    df = raw_df.merge(clusters_df, on=id_col, how="left")
    df["type_id"] = df["cluster_id"].map(cluster_to_type)

    return dict(zip(df[id_col], df["type_id"]))