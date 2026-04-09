import os, json
import pandas as pd
from time import time

from ..distrib import BaseWorker

SYSTEM_PROMPT = """Eres un clasificador experto de noticias en español. Tu tarea es determinar si un artículo
trata sobre un CONFLICTO SOCIAL o DENUNCIA UNA PROBLEMÁTICA SOCIAL hacia una EMPRESA MINERA.

SÍ es conflicto/evento minero:
- Protestas comunitarias contra proyectos o empresas mineras
- Huelgas o paros de trabajadores mineros
- Demandas judiciales contra empresas mineras por daño ambiental o social
- Derrames, accidentes o emergencias en operaciones mineras que afectan comunidades
- Enfrentamientos entre pobladores y fuerzas del orden por actividad minera
- Oposición de pueblos indígenas a proyectos de exploración minera
- Reportes de conflictos sociales vinculados a la minería

0 = NO ES CONFLICTO MINERO, 1 = SÍ ES CONFLICTO MINERO
En empresa_minera escribe ÚNICAMENTE el nombre de la empresa o proyecto minero si se menciona explícitamente, o '' si no.
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "classification": {"type": "string", "enum": ["0", "1"]},
        "reason":         {"type": "string", "description": "Explicación breve en una sola línea, sin saltos de línea."},
        "empresa_minera": {"type": "string", "description": (
        "Nombre de la empresa o proyecto minero mencionado explícitamente en el artículo. "
        "Ejemplos válidos: 'Antamina', 'MMG Las Bambas', 'Yanacocha', 'Tía María', 'Southern Copper'. "
        "Si no se menciona ninguna empresa o proyecto minero específico, devolver exactamente: ''"
        )
    }
    },
    "required": ["classification", "reason", "empresa_minera"],
}

class LLMJob(BaseWorker):
    def __init__(self, 
            system_prompt,
            response_schema,         
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.system_prompt = system_prompt
        self.response_schema = response_schema

    def response(self, row):
        """Implementar en subclase."""
        raise NotImplementedError

    def on_error(self, row, exc: Exception) -> dict | None:
        print(f"  Art {row['id']}: ERROR — {exc}")
        return {
                "id":             row["id"],
                "llm_class":      "ERROR",
                "llm_reason":     str(exc),
                "llm_minera":     ''
        }

    def process_row(self, row) -> dict | None:
        results = self.response(row)
        return {
            "id":             row['id'],
            "llm_class":      results["classification"],
            "llm_reason":     results["reason"].replace("\n", " "),
            "llm_minera":     results["empresa_minera"].replace("\n", " ")
        }

class Anthropic(LLMJob):
    def __init__(self, 
            api_key, 
            model = "claude-sonnet-4-20250514", 
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)

        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def response(self, row):
        stream = self.client.messages.create(
            model = self.model,
            max_tokens=200,
            system=self.system_prompt,
            messages=[{"role": "user", "content": f"Clasifica:\n\n{row['body']}"}],
        ).to_dict()

        time.sleep(0.5)

        return json.loads(
            stream['content'][0]['text'][7:-3]
        )

class Google(LLMJob):
    def __init__(self, 
            api_key, 
            model = 'gemini-2.5-flash',
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)

        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.config = genai.types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            response_mime_type="application/json",
            response_schema=self.response_schema
        )
        self.model = model

    def response(self, row):
        stream = self.client.models.generate_content(
            model=self.model,
            contents=f"Clasifica:\n\n{row['body']}",
            config=self.config
        )

        time.sleep(0.5)

        return stream.parsed

class Ollama(LLMJob):
    def __init__(self, 
            model = "qwen2.5:14b",
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)

        self.model = model

    def response(self, row):
        import ollama
        stream = ollama.chat(
            model = self.model,
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Clasifica:\n\n{row['body']}"}
            ],
            format=self.response_schema
        )

        return json.loads(stream["message"]["content"])