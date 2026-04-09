import json
import uuid
import requests
import pandas as pd
import ollama
from pydantic import BaseModel
from typing import Optional
from distrib import BaseWorker

def build_el_input(
    clusters_csv: str,
    extracciones_csv: str,
    cuerpo_csv: str,
    output_csv: str
):
    clusters_df = pd.read_csv(clusters_csv)
    cuerpo_df   = pd.read_csv(cuerpo_csv)[["id", "body"]]

    entity_rows = []
    for _, doc in pd.read_csv(extracciones_csv).iterrows():
        for ent in json.loads(doc["entities"]):
            entity_rows.append({
                "mention_id": f"{doc['id']}__{ent['id']}",
                "id":         doc["id"],
                "name":       ent["name"],
                "type":       ent["type"],
            })
    entities_df = pd.DataFrame(entity_rows)

    df = clusters_df \
        .merge(entities_df, on="mention_id", how="left") \
        .merge(cuerpo_df,   on="id",         how="left")

    rows = []
    for cluster_id, group in df[df["cluster_id"].notna()].groupby("cluster_id"):
        rows.append({
            "cluster_id": cluster_id,
            "type":       group["type"].iloc[0],
            "mentions":   json.dumps(group["name"].unique().tolist(), ensure_ascii=False),
            "contextos":  json.dumps(group["body"].tolist()[:3],      ensure_ascii=False),
        })
    for _, row in df[df["cluster_id"].isna()].iterrows():
        rows.append({
            "cluster_id": f"noise__{row['mention_id']}",
            "type":       row["type"],
            "mentions":   json.dumps([row["name"]], ensure_ascii=False),
            "contextos":  json.dumps([row["body"]], ensure_ascii=False),
        })

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"{len(rows)} clusters listos para EL")

def build_canon_input(
    clusters_csv: str,
    raw_csv: str,
    id_col: str,           # relation_id o event_id
    text_cols: list[str],  # columnas a agrupar — ["relation", "evidence"] o ["event_type", "trigger"]
    output_csv: str
):
    """Genérico para relaciones y eventos."""
    df = pd.read_csv(clusters_csv).merge(
        pd.read_csv(raw_csv), on=id_col, how="left"
    )

    rows = []
    for cluster_id, group in df[df["cluster_id"].notna()].groupby("cluster_id"):
        row = {"cluster_id": cluster_id}
        for col in text_cols:
            row[col] = json.dumps(group[col].unique().tolist()[:5], ensure_ascii=False)
        rows.append(row)

    for _, row in df[df["cluster_id"].isna()].iterrows():
        r = {"cluster_id": f"noise__{row[id_col]}"}
        for col in text_cols:
            r[col] = json.dumps([row[col]], ensure_ascii=False)
        rows.append(r)

    pd.DataFrame(rows).to_csv(output_csv, index=False)

def wikidata_lookup(canonical: str) -> str | None:
    try:
        r = requests.get(
            "https://www.wikidata.org/w/api.php",
            params={
                "action":   "wbsearchentities",
                "search":   canonical,
                "language": "es",
                "format":   "json",
                "limit":    1,
            },
            timeout=10
        )
        results = r.json().get("search", [])
        return results[0]["id"] if results else None
    except Exception:
        return None
    
class CanonWorker(BaseWorker):
    """
    Worker genérico de canonicalización.
    Subclasear para definir schema, system prompt y prompt builder.
    """

    schema      = None   # Pydantic BaseModel
    system      = ""
    id_col      = "cluster_id"

    def build_prompt(self, row) -> str:
        raise NotImplementedError

    def build_result(self, row, res) -> dict:
        raise NotImplementedError

    def process_row(self, row) -> dict | None:
        stream = ollama.chat(
            model="qwen2.5:32b",
            format=self.schema.model_json_schema(),
            messages=[
                {"role": "system", "content": self.system},
                {"role": "user",   "content": self.build_prompt(row)}
            ]
        )
        res = self.schema.model_validate_json(stream.message.content)
        return self.build_result(row, res)

    def on_error(self, row, exc: Exception) -> dict | None:
        print(f"✗ {row['cluster_id']}: {exc}")
        return None


# ── EL Worker ─────────────────────────────────────────────────

class ELResolution(BaseModel):
    canonical: str
    is_new:    bool

class ELWorker(CanonWorker):

    schema = ELResolution
    system = """Eres un sistema de entity linking para noticias peruanas sobre minería y conflictos sociales.
Dado un cluster de menciones que probablemente refieren a la misma entidad,
determina su nombre canónico y si es una entidad local sin entrada en Wikidata.

Reglas:
- canonical: el nombre más completo y formal de la entidad
- is_new: true para entidades locales peruanas (comunidades campesinas, dirigentes
  locales, proyectos pequeños sin proyección internacional)
- is_new: false para empresas multinacionales, ministerios, ciudades, personajes públicos
- Responde SOLO en JSON"""

    def build_prompt(self, row) -> str:
        mentions  = json.loads(row["mentions"])
        contextos = json.loads(row["contextos"])
        return (
            f"Tipo: {row['type']}\n"
            f"Menciones: {mentions}\n"
            f"Contextos:\n" +
            "\n".join(f'- "{c[:200]}"' for c in contextos)
        )

    def build_result(self, row, res) -> dict:
        wikidata_id = None
        if not res.is_new:
            wikidata_id = wikidata_lookup(res.canonical)
        return {
            "cluster_id":  row["cluster_id"],
            "entity_id":   f"ent_{uuid.uuid4().hex[:8]}",
            "canonical":   res.canonical,
            "wikidata_id": wikidata_id,
            "is_new":      res.is_new,
        }


# ── Relation Canon Worker ─────────────────────────────────────

class RelationTypeResolution(BaseModel):
    canonical:   str   # VERBO_PREPOSICION — OPONE_A, OPERA_EN
    description: str

class RelationCanonWorker(CanonWorker):

    schema = RelationTypeResolution
    system = """Eres un sistema de canonicalización de relaciones para un knowledge graph
de noticias peruanas sobre minería y conflictos sociales.
Dado un cluster de expresiones de relación similares, propón un nombre canónico
en formato VERBO_PREPOSICION en mayúsculas (ej. OPERA_EN, OPONE_A, NEGOCIA_CON)
y una descripción breve.
Responde SOLO en JSON."""

    def build_prompt(self, row) -> str:
        relations = json.loads(row["relation"])
        evidences = json.loads(row["evidence"])
        return (
            f"Expresiones: {relations}\n"
            f"Ejemplos de uso:\n" +
            "\n".join(f'- "{e}"' for e in evidences)
        )

    def build_result(self, row, res) -> dict:
        return {
            "cluster_id":  row["cluster_id"],
            "type_id":     f"rt_{uuid.uuid4().hex[:6]}",
            "canonical":   res.canonical,
            "description": res.description,
        }


# ── Event Canon Worker ────────────────────────────────────────

class EventTypeResolution(BaseModel):
    canonical:   str   # Categoria.Tipo — Conflicto.Protesta, Inst.Decisión
    description: str

class EventCanonWorker(CanonWorker):

    schema = EventTypeResolution
    system = """Eres un sistema de canonicalización de eventos para un knowledge graph
de noticias peruanas sobre minería y conflictos sociales.
Dado un cluster de tipos de evento similares, propón un nombre canónico
en formato Categoria.Tipo (ej. Conflicto.Protesta, Inst.Decisión, Econ.Inversión)
y una descripción breve.
Responde SOLO en JSON."""

    def build_prompt(self, row) -> str:
        event_types = json.loads(row["event_type"])
        triggers    = json.loads(row["trigger"])
        return (
            f"Tipos de evento: {event_types}\n"
            f"Triggers: {triggers}"
        )

    def build_result(self, row, res) -> dict:
        return {
            "cluster_id":  row["cluster_id"],
            "type_id":     f"et_{uuid.uuid4().hex[:6]}",
            "canonical":   res.canonical,
            "description": res.description,
        }