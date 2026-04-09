import json
import ollama
import pandas as pd
from pydantic import BaseModel
from distrib import BaseWorker


FRAMES = ["problem", "diagnosis", "responsibles", "solution"]

class FrameScore(BaseModel):
    frame:    str
    present:  bool
    evidence: str

class FramingResult(BaseModel):
    frames: list[FrameScore]


SYSTEM_PROMPT = """Eres un analista de medios especializado en noticias peruanas sobre minería y conflictos sociales.
Dado un artículo, identifica qué frames de Entman (1993) están presentes.

Frames:
- problem:       el artículo define o presenta un problema central del conflicto
- diagnosis:     el artículo atribuye causas o explica por qué ocurre el problema
- responsibles:  el artículo señala quién tiene la culpa o responsabilidad
- solution:      el artículo propone o menciona una salida o resolución

Reglas:
- Un frame está presente si el texto lo enfatiza explícitamente, no solo lo menciona de pasada
- Devuelve los cuatro frames — los no presentes con present: false y evidence: ""
- evidence: fragmento breve del texto que justifica la presencia del frame
- Responde SOLO en JSON"""


class Framer(BaseWorker):

    def process_row(self, row) -> dict | None:
        stream = ollama.chat(
            model="qwen2.5:14b",
            format=FramingResult.model_json_schema(),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Artículo:\n{row['body']}"}
            ]
        )
        result = FramingResult.model_validate_json(stream.message.content)

        row_out = {"id": row["id"]}
        for fs in result.frames:
            row_out[f"frame_{fs.frame}"]          = fs.present
            row_out[f"frame_{fs.frame}_evidence"] = fs.evidence

        return row_out

    def on_error(self, row, exc: Exception) -> dict | None:
        print(f"✗ {row['id']}: {exc}")
        row_out = {"id": row["id"]}
        for frame in FRAMES:
            row_out[f"frame_{frame}"]          = None
            row_out[f"frame_{frame}_evidence"] = None
        return row_out
