import json
from pydantic import BaseModel
from typing import Optional, Literal

from ..distrib import BaseWorker

class Entity(BaseModel):
    id: str         # e00X
    name: str       # texto libre
    type: Literal["PER", "ORG", "LOC", "PROJ", "NORM", "DATE", "MONEY"]
    confidence: Literal["high", "medium", "low"]

class Relation(BaseModel):
    id: str         # r00X
    subject: str    # entity id
    relation: str   # texto libre en infinitivo
    object: str     # entity id
    evidence: str   # fragmento textual que sustenta la relación
    confidence: Literal["high", "medium", "low"]

class Event(BaseModel):
    id: str         # ev00X
    type: str       # texto libre
    trigger: str    # palabr o frase que dispara el evento
    arguments: dict # roles libres
    confidence: Literal["high", "medium", "low"]

class KB(BaseModel):
    entities: list[Entity]
    relations: list[Relation]
    events: list[Event]

SYSTEM_PROMPT = """Eres un extractor de información para noticias peruanas sobre minería y conflictos sociales.

Tipos de entidad con ejemplos:
- PER: personas → "Pedro Castillo", "Juan Quispe", "el presidente de la comunidad"
- ORG: organizaciones → "MMG Limited", "Ministerio de Energía y Minas",
       "comunidad campesina de Fuerabamba", "Frente de Defensa Ambiental"
- LOC: lugares → "Apurímac", "río Challhuahuacho", "cerro Huancané", "provincia de Cotabambas"
- PROJ: proyectos mineros → "Las Bambas", "Tía María", "Conga", "Antapaccay", "Quellaveco"
- NORM: normas legales → "Ley 29785", "DS 001-2012-EM", "EIA aprobado en 2014"
- DATE: fechas y períodos → "ayer", "el 15 de marzo", "desde 2019", "durante tres días"
- MONEY: montos e inversiones → "S/ 2 millones", "USD 1.4 billones", "50% de regalías"

Reglas de extracción:
- Asigna IDs secuenciales: entidades e001, e002... / relaciones r001, r002... / eventos ev001, ev002...
- Para relaciones, usa lenguaje natural breve en infinitivo: "oponerse a", "operar en", "anunciar"
- Para eventos, describe el tipo libremente: "bloqueo de vía", "mesa de diálogo", "derrame de relave"
- No inferas lo que no está explícito
- Responde SOLO en JSON

Ejemplo:
  texto: "La comunidad de Fuerabamba bloqueó ayer el acceso a Las Bambas,
          operada por MMG Limited, exigiendo una mesa de diálogo."

  entities: [
    {"id": "e001", "text": "comunidad de Fuerabamba", "type": "ORG",  "confidence": "high"},
    {"id": "e002", "text": "Las Bambas",              "type": "PROJ", "confidence": "high"},
    {"id": "e003", "text": "MMG Limited",             "type": "ORG",  "confidence": "high"},
    {"id": "e004", "text": "ayer",                    "type": "DATE", "confidence": "high"}
  ]

  relations: [
    {
      "id": "r001",
      "subject": "e003",
      "relation": "operar en",
      "object": "e002",
      "confidence": "high",
      "evidence": "Las Bambas, operada por MMG Limited"
    },
    {
      "id": "r002",
      "subject": "e001",
      "relation": "oponerse a",
      "object": "e003",
      "confidence": "medium",
      "evidence": "la comunidad de Fuerabamba bloqueó el acceso a Las Bambas"
    }
  ]

  events: [
    {
      "id": "ev001",
      "type": "bloqueo de vía",
      "trigger": "bloqueó",
      "arguments": {"actor": "e001", "lugar": "e002", "fecha": "e004"},
      "confidence": "high"
    }
  ]"""


def build_prompt(row):
    return f"""Noticia:
Texto: {row['body']}

Extrae todas las entidades, relaciones y eventos presentes en el texto."""

class KBuilder(BaseWorker):
    def __init__(self, 
            system_prompt,
            model_schema: KB,
            model = "qwen2.5:14b",         
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)

        self.system_prompt = system_prompt
        self.model_schema = model_schema
        self.model = model

    def process_row(self, row) -> dict | None:
        import ollama

        stream = ollama.chat(
            model = self.model,
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": build_prompt(row)}
            ],
            format = self.model_schema.model_json_schema()
        )
        extraction = self.model_schema.model_validate_json(stream.message.content)

        return {
            "id":           row['id'],
            "entities":     json.dumps(extraction.model_dump()['entities'], ensure_ascii=False),
            "relations":    json.dumps(extraction.model_dump()['relations'], ensure_ascii=False),
            "events":       json.dumps(extraction.model_dump()['events'], ensure_ascii=False)
        }
    
    def on_error(self, row, exc: Exception) -> dict | None:
        print(f"  Art {row['id']}: ERROR — {exc}")
        return {
            "id":           row["id"],
            "entities":     list(),
            "relations":    list(),
            "events":       list()
        }