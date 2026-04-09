import json
import ollama
import pandas as pd
from pydantic import BaseModel
from typing import Optional
from distrib import BaseWorker
from ..factual.backfill import _build_mention_to_entity


ARGUMENT_TYPES = {
    "responsibility":   "El actor señala quién tiene la obligación o culpa de actuar",
    "danger_threats":   "El actor advierte sobre riesgos, daños o consecuencias negativas",
    "utility_benefits": "El actor destaca ventajas, beneficios o utilidad de algo",
    "others":           "Argumento que no encaja en las categorías anteriores",
}

ARG_LIST = "\n".join(f"- {k}: {v}" for k, v in ARGUMENT_TYPES.items())

class Argument(BaseModel):
    sentence:      str
    claimant:      Optional[str]  # tal como aparece en el texto
    argument_type: str            # responsibility | danger_threats | utility_benefits | others
    target:        Optional[str]  # tal como aparece en el texto, null si no está explícito

class ArgumentResult(BaseModel):
    arguments: list[Argument]


SYSTEM_PROMPT = f"""Eres un analista de argumentación para noticias peruanas sobre minería y conflictos sociales.
Dado un artículo, extrae los argumentos explícitos usando la tipología de Wodak (2003).

Tipos de argumento:
{ARG_LIST}

Reglas:
- Solo extrae oraciones que contengan un argumento explícito — no background ni descripción neutral
- claimant: quién hace el argumento tal como aparece en el texto (persona, organización, institución)
- target: a quién va dirigido el argumento si está explícito, null si no
- No inferas lo que no está en el texto
- Responde SOLO en JSON"""

class ArgumentMiner(BaseWorker):

    def process_row(self, row) -> dict | None:
        response = ollama.chat(
            model="qwen2.5:14b",
            format=ArgumentResult.model_json_schema(),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Artículo:\n{row['body']}"}
            ]
        )
        result = ArgumentResult.model_validate_json(response.message.content)

        if not result.arguments:
            return None

        return {
            "id":        row["id"],
            "arguments": json.dumps(
                [a.model_dump() for a in result.arguments],
                ensure_ascii=False
            ),
        }

    def on_error(self, row, exc: Exception) -> dict | None:
        print(f"✗ {row['id']}: {exc}")
        return None
    
def explode_arguments(raw_csv: str, output_csv: str):
    df = pd.read_csv(raw_csv)
    rows = []
    for _, doc in df.iterrows():
        for arg in json.loads(doc["arguments"]):
            rows.append({
                "id":            doc["id"],
                "argument_type": arg["argument_type"],
                "claimant":      arg["claimant"],
                "target":        arg["target"],
                "sentence":      arg["sentence"],
            })
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"{len(rows)} argumentos extraídos de {len(df)} documentos")


def link_arguments_to_kb(
    arguments_csv: str,
    extracciones_csv: str,
    menciones_raw_csv: str,
    clusters_csv: str,
    entidades_csv: str,
    output_csv: str
):
    args_df = pd.read_csv(arguments_csv)

    # Reusar helper de backfill — mention_id → entity_id canónico
    mention_to_entity = _build_mention_to_entity(
        menciones_raw_csv, clusters_csv, entidades_csv
    )

    # (doc_id, texto_mención_lower) → mention_id
    text_to_mention = {}
    for _, doc in pd.read_csv(extracciones_csv).iterrows():
        for ent in json.loads(doc["entities"]):
            key = (doc["id"], ent["name"].lower())
            text_to_mention[key] = f"{doc['id']}__{ent['id']}"

    def resolve(doc_id, name) -> str | None:
        if not name:
            return None
        mention_id = text_to_mention.get((doc_id, name.lower()))
        if not mention_id:
            return None
        return mention_to_entity.get(mention_id)

    args_df["claimant_id"] = args_df.apply(
        lambda r: resolve(r["id"], r["claimant"]), axis=1
    )
    args_df["target_id"] = args_df.apply(
        lambda r: resolve(r["id"], r["target"]), axis=1
    )

    args_df.to_csv(output_csv, index=False)

    total    = len(args_df)
    resolved = args_df["claimant_id"].notna().sum()
    print(f"Claimants resueltos: {resolved}/{total}")
    print(f"Sin resolver:        {total - resolved}/{total} → cola de revisión manual")

