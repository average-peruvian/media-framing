import numpy as np
import pandas as pd
import json, time
from tqdm import tqdm
import os

SEED_DESCRIPTIONS_CONFLICT = [
    "Conflicto social comunidad protesta empresa minera contaminación agua territorio",
    "Huelga trabajadores mineros sindicato paro labores despidos condiciones seguridad mina",
    "Protesta pobladores bloqueo carretera oposición proyecto minero comunidades indígenas",
    "Demanda judicial comunidad contra empresa minera daño ambiental contaminación ríos metales pesados",
    "Derrame relaves tóxicos emergencia ambiental evacuación familias afectadas minería",
    "Enfrentamiento policía manifestantes minera derechos humanos represión detenidos dirigentes",
    "Conflicto territorial minería tierras ancestrales pueblos originarios consulta previa",
    "Accidente minero trabajadores atrapados derrumbe túnel rescate seguridad laboral",
    "Defensoría pueblo conflictos sociales activos actividad minera disputas territoriales",
    "Organizaciones ambientalistas oposición exploración minera humedales fuentes agua contaminación",
]

SEED_SENTENCES_CONFLICT = [
    "Comunidades campesinas protestan contra proyecto minero por contaminación del agua y falta de consulta previa",
    "Trabajadores mineros inician huelga exigiendo mejores condiciones de seguridad tras accidente en la mina",
    "Pobladores bloquean carretera en oposición a empresa minera que opera en sus territorios ancestrales",
    "Tribunal ordena indemnización a comunidades afectadas por contaminación minera de ríos con metales pesados",
    "Derrame de relaves tóxicos provoca emergencia ambiental y evacuación de familias cerca de operación minera",
    "Organizaciones de derechos humanos condenan detención de dirigentes comunitarios que protestaban contra minera",
    "Pueblos indígenas se oponen a exploración minera en territorios sagrados y fuentes de agua ancestrales",
    "Sindicato minero declara paro de labores tras colapso de túnel que dejó trabajadores atrapados",
    "Defensoría del Pueblo reporta decenas de conflictos sociales activos vinculados a la actividad minera",
    "Demanda colectiva contra empresa minera por daños a la salud de pobladores por contaminación ambiental",
]

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

"""
def stage_w2v(
    articles,
    seeds,
    threshold = 0.35,
    vector_size = 200,
    window = 8,
    min_count = 1,
    epochs = 100      
):
    from gensim.models import Word2Vec
    from sklearn.feature_extraction.text import TfidfVectorizer

    print('Tokenizing.')
    all_tokenized = [janitor.tokenize(janitor.clean_text(t)) for t in tqdm(articles)] 
    seeds_tokenized = [janitor.tokenize(janitor.clean_text(t)) for t in tqdm(seeds)]
    corpus = all_tokenized + seeds_tokenized
    print('\n')

    print('Training W2V.')
    model = Word2Vec(
            sentences=corpus,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4,
            epochs=epochs,
            sg=1,  # Skip-gram (better for small corpora)
            seed=42,
        )
    wv = model.wv
    print('\n')

    print('Setting feature weights via TF-IDF.')
    corpus_strings = [" ".join(tokens) for tokens in all_tokenized + seeds_tokenized]
    tfidf = TfidfVectorizer()
    tfidf.fit(corpus_strings)
    vocab_tfidf = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))
    print('\n')

    def doc_vector(tokens: list[str]):
        # TF-IDF-weighted average of word vectors.
        vectors = []
        weights = []
        for token in tokens:
            if token in wv:
                vectors.append(wv[token])
                weights.append(vocab_tfidf.get(token, 1.0))
        if not vectors:
            return np.zeros(vector_size)
        vectors = np.array(vectors)
        weights = np.array(weights).reshape(-1, 1)
        return np.average(vectors, axis=0, weights=weights.flatten())
    
    print('Extracting characteristics.')
    article_vecs = np.array([doc_vector(tokens) for tokens in tqdm(all_tokenized)])
    seed_vecs = np.array([doc_vector(tokens) for tokens in tqdm(seeds_tokenized)])
    print('\n')

    from sklearn.metrics.pairwise import cosine_similarity

    print('Making similarity matrix.')
    sim_matrix = cosine_similarity(article_vecs, seed_vecs)
    max_similarities = sim_matrix.max(axis=1)
    scores = [simil >= threshold for simil in max_similarities]
    print('\n')

    return scores

def stage_transformer(
    articles,
    threshold = 0.35,
    model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
):
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    model = SentenceTransformer(model_name)
    seed_embeddings = model.encode(SEED_SENTENCES_CONFLICT, show_progress_bar=False)
    article_embeddings = model.encode(articles, show_progress_bar=True)

    sim_matrix = cosine_similarity(article_embeddings, seed_embeddings)
    max_similarities = sim_matrix.max(axis=1)
    scores = [simil >= threshold for simil in max_similarities]

    return scores

def stage_zero_shot(
    df,
    threshold = 0.5,
    model_name = 'joeddav/xlm-roberta-large-xnli'
):
    from transformers import pipeline as hf_pipeline
    classifier = hf_pipeline(
        "zero-shot-classification",
        model=model_name, 
        device=-1
    )

    candidate_labels = [
        "conflicto social, protesta o evento relacionado con una empresa minera",
        "noticias de entretenimiento, deportes, finanzas u otros temas no relacionados con conflictos mineros",
    ]

    zs_scores = []
    for _, row in df.iterrows():
        result = classifier(
            row['text'],
            candidate_labels=candidate_labels,
            hypothesis_template='Este artículo trata sobre {}.'
        )
        zs_scores.append(
            result['scores'][result['labels'].index(candidate_labels[0])] >= threshold
        )

    return zs_scores
"""

def stage_anthropic_judge(
    df,
    api_key,
    system_prompt
):
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    judgments = []

    for _, row in tqdm(df.iterrows(),total=df.shape[0]):
        try:
            response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=200,
                    system=system_prompt,
                    messages=[{"role": "user", "content": f"Clasifica:\n\n{row['body']}"}],
                ).to_dict()
            result = json.loads(response['content'][0]['text'][7:-3])
            judgments.append({
                "id": row["id"],
                "llm_class": result["classification"],
                "llm_confidence": result["confidence"],
                "llm_reason": result["reason"],
            })
            time.sleep(0.5)
        except Exception as e:
            print(f"  Art {row['id']:2d}: ERROR — {e}")
            judgments.append({
                "id": row["id"], 
                "llm_class": "ERROR",
                "llm_confidence": 0, 
                "llm_reason": str(e)}
            )

    return judgments





def stage_google_judge(
    df,
    api_key,
    system_prompt,
    model = 'gemini-2.5-flash',
    output_csv = "judgments.csv"
):
    from google import genai
    client = genai.Client(api_key=api_key)
    config = genai.types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA
    )

    if os.path.exists(output_csv):
        done_ids = set(pd.read_csv(output_csv)["id"].tolist())
        print(f"Retomando — {len(done_ids)} artículos ya procesados")
    else:
        done_ids = set()

    for _, row in tqdm(df.iterrows(),total=df.shape[0]):
        if row["id"] in done_ids:
            continue
        
        try:
            response = response = client.models.generate_content(
                model=model,
                contents=f"Clasifica:\n\n{row['body']}",
                config=config
            )
            parsed = response.parsed
            result = {
                "id":             row["id"],
                "llm_class":      parsed["classification"],
                "llm_confidence": parsed["confidence"],
                "llm_reason":     parsed["reason"],
            }
        except Exception as e:
            print(f"  Art {row['id']}: ERROR — {e}")
            result = {
                "id":             row["id"],
                "llm_class":      "ERROR",
                "llm_confidence": 0,
                "llm_reason":     str(e),
            }

        pd.DataFrame([result]).to_csv(
            output_csv,
            mode="a",
            header=not os.path.exists(output_csv),
            index=False,
            encoding='utf-8'
        )
        time.sleep(0.5)

    return pd.read_csv(output_csv)

def stage_ollama_judge(
    df,
    system_prompt,
    schema,
    model = 'gemini-2.5-flash',
    output_csv = "judgments.csv"
):
    import ollama

    if os.path.exists(output_csv):
        done_ids = set(pd.read_csv(output_csv)["id"].tolist())
        print(f"Retomando — {len(done_ids)} artículos ya procesados")
    else:
        done_ids = set()

    for _, row in tqdm(df.iterrows(),total=df.shape[0]):
        if row["id"] in done_ids:
            continue

        try:
            response = ollama.chat(
                model = model,
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Clasifica:\n\n{row['body']}"}
                ],
                format=schema
            )
            parsed = json.loads(response["message"]["content"])
            result = {
                "id":             row["id"],
                "llm_class":      parsed["classification"],
                "llm_reason":     parsed["reason"].replace("\n", " "),
                "llm_minera":     parsed["empresa_minera"].replace("\n", " ")
            }

        except Exception as e:
            print(f"  Art {row['id']}: ERROR — {e}")
            result = {
                "id":             row["id"],
                "llm_class":      "ERROR",
                "llm_reason":     str(e),
                "llm_minero":     ''
            }

        pd.DataFrame([result]).to_csv(
            output_csv,
            mode="a",
            header=not os.path.exists(output_csv),
            index=False,
            encoding="utf8",
        )

    return pd.read_csv(output_csv, encoding="utf8")
