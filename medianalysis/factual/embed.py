import json
import pandas as pd
from sentence_transformers import SentenceTransformer

from ..distrib import BaseWorker

def build_embedding_input(
    menciones_csv: str,
    extracciones_csv: str,
    cuerpo_csv: str,
    output_csv: str
):
    menciones_df     = pd.read_csv(menciones_csv)
    extracciones_df  = pd.read_csv(extracciones_csv)
    cuerpo_df        = pd.read_csv(cuerpo_csv)[["id", "body"]]

    # Explota entities para recuperar name por mention_id
    entity_rows = []
    for _, doc in extracciones_df.iterrows():
        for ent in json.loads(doc["entities"]):
            entity_rows.append({
                "id":       doc["id"],
                "local_id": ent["id"],
                "name":     ent["name"],
                "type":     ent["type"],
            })
    entities_df = pd.DataFrame(entity_rows)

    df = menciones_df \
        .merge(entities_df, on=["id", "local_id"], how="left") \
        .merge(cuerpo_df,   on="id",               how="left")

    df.to_csv(output_csv, index=False)

class Embedder(BaseWorker):
    """
    Worker genérico de embeddings.
    
    fields: lista de columnas a concatenar con [SEP]
    id_col: columna usada como clave única (mention_id, relation_id, event_id)
    out_id_col: nombre del id en el output — por defecto igual a id_col
    """
    def __init__(
        self,
        fields: list[str],
        out_id_col: str | None = None,
        model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.fields     = fields
        self.out_id_col = out_id_col or kwargs.get("id_col")
        self.model      = SentenceTransformer(model)

    def process_row(self, row) -> dict | None:
        text = " [SEP] ".join(str(row[f]) for f in self.fields)
        embedding = self.model.encode(text, show_progress_bar=False)
        return {
            self.out_id_col: row[self.out_id_col],
            "embedding":     json.dumps(embedding.tolist()),
        }

    def on_error(self, row, exc: Exception) -> dict | None:
        print(f"✗ {row[self.out_id_col]}: {exc}")
        return {
            self.out_id_col: row[self.out_id_col],
            "embedding":     json.dumps([None]),
        }
    
def MentionEmbedder(**kwargs):
    return Embedder(fields=["name", "body"],          **kwargs)

def RelationEmbedder(**kwargs):
    return Embedder(fields=["relation", "evidence"],  **kwargs)

def EventEmbedder(**kwargs):
    return Embedder(fields=["event_type", "trigger"], **kwargs)
