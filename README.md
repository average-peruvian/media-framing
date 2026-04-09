# Red de Actores-Actantes aplicada para la minera Las Bambas

Pipeline de extracción de conocimiento y análisis retórico sobre noticias peruanas de conflictos mineros.

Desarrollado como parte de una tesis de pregrado sobre análisis de medios y construcción automatizada de bases de conocimiento en español.

## Descripción general

El sistema construye un knowledge graph factual a partir de noticias de conflictos mineros peruanos, y lo complementa con una capa de análisis retórico que captura frames, argumentos, tópicos y sentimiento. El pipeline está diseñado para ejecutarse de forma distribuida en Google Colab, con cada etapa escrita como un worker reanudable que persiste su output en Google Drive.

La extracción conjunta de entidades, relaciones y eventos se realiza con Qwen2.5 (14B/32B) vía ollama. La resolución global — clustering de menciones, entity linking y canonicalización de tipos — opera sobre el corpus completo una vez que la extracción distribuida finaliza.

## Componentes

### Retrieval

| Componente | Descripción |
|---|---|
| Scraping | Extracción de texto desde URLs con fallback automático a Wayback Machine. |
| Clasificación | Filtrado de relevancia temática mediante LLM (Ollama, Google Gemini, Anthropic Claude). |

### Factual

| Componente | Descripción |
|---|---|
| Extracción | Extracción conjunta NER + RE + EE por documento con schema cerrado via Pydantic. Entidades: PER, ORG, LOC, PROJ, NORM, DATE, MONEY. Relaciones y tipos de evento en lenguaje natural libre. |
| Embeddings | Embeddings de menciones, relaciones y eventos con `paraphrase-multilingual-MiniLM-L12-v2`. Las menciones se embeddean como `nombre [SEP] cuerpo del artículo`. |
| Clustering | HDBSCAN por tipo de entidad para agrupar menciones que refieren a la misma entidad real. Relaciones y eventos se clustean globalmente. |
| Entity Linking | Resolución de clusters a entidades canónicas con Qwen2.5 32B. Para entidades conocidas, lookup en Wikidata via API. Entidades locales (comunidades, dirigentes, proyectos pequeños) se marcan como nuevas en la KB propia. |
| Canonicalización | Tipos de relación en formato `VERBO_PREPOSICION` (OPONE_A, OPERA_EN). Tipos de evento en formato `Categoría.Tipo` (Conflicto.Protesta, Inst.Decisión). Ambos emergen del corpus, no de un schema predefinido. |
| Backfill | Reemplazo de IDs locales por entity_ids canónicos en el corpus completo. Output: `grafo.csv` (tripletas) y `eventos_resueltos.csv`. |

### Rhetoric

| Componente | Descripción |
|---|---|
| Sentiment | Análisis de sentimiento por artículo con SaBERT (`VerificadoProfesional/SaBERT-Spanish-Sentiment-Analysis`). |
| Topics | Modelado de tópicos con BERTopic sobre el corpus completo. Soporta asignación incremental a nuevos documentos sin re-entrenamiento. |
| Framing | Clasificación de frames de Entman (1993) por artículo: problem, diagnosis, responsibles, solution. Incluye fragmento de evidencia por frame. |
| Arguments | Extracción de argumentos de Wodak (2003): responsibility, danger_threats, utility_benefits, others. Argumentos linkeados a entity_ids canónicos del KB. |

### Scripts

| Script | Descripción |
|---|---|
| `scripts/validate_corpus.py` | Herramienta tkinter de anotación manual de corpus. Navegación por teclado, highlighting de keywords, soporte para predicciones LLM en panel lateral, exportación de etiquetas a CSV separado. |

## Archivos de datos

```
Entrada
  urls.csv                   id | url | media_name | publish_date | title
  noticias_metadata.csv      id | fecha | fuente | ...
  noticias_cuerpo.csv        id | body

Retrieval
  clasificacion.csv          id | llm_class | llm_reason | llm_minera
  noticias_relevantes.csv    id | body | keywords         artículos filtrados

Factual — Fase 1
  extracciones.csv           id | entities | relations | events

Factual — Fase 2
  menciones_raw.csv          id | local_id | mention_id | entity_id
  relaciones_raw.csv         relation_id | id | local_id | relation | evidence | confidence
  eventos_raw.csv            event_id | id | local_id | event_type | trigger | confidence
  clusters.csv               mention_id | cluster_id
  relaciones_clusters.csv    relation_id | cluster_id
  eventos_clusters.csv       event_id | cluster_id
  entidades_canonicas.csv    cluster_id | entity_id | canonical | wikidata_id | is_new
  tipos_relacion.csv         cluster_id | type_id | canonical | description
  tipos_evento.csv           cluster_id | type_id | canonical | description
  grafo.csv                  subject | relation_type_id | object | doc_id | confidence | evidence
  eventos_resueltos.csv      event_type_id | doc_id | trigger | arguments | confidence

Rhetoric
  sentiment.csv              id | class | confidence
  topics_por_doc.csv         id | topic_id
  topics_keywords.csv        topic_id | count | keywords | label
  framing.csv                id | frame_problem | frame_diagnosis | frame_responsibles | frame_solution (+ _evidence)
  arguments_linked.csv       id | argument_type | claimant | claimant_id | target | target_id | sentence
```

## Modelos

| Componente | Modelo | Ejecución |
|---|---|---|
| Clasificación de relevancia | `qwen2.5:14b` via ollama | distribuida |
| Extracción NER + RE + EE | `qwen2.5:14b` via ollama | distribuida |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | distribuida |
| Entity Linking | `qwen2.5:32b` via ollama | única |
| Canonicalización | `qwen2.5:32b` via ollama | única |
| Sentiment | `SaBERT-Spanish-Sentiment-Analysis` | distribuida |
| Topics | BERTopic + `paraphrase-multilingual-MiniLM-L12-v2` | única |
| Framing | `qwen2.5:14b` via ollama | distribuida |
| Arguments | `qwen2.5:14b` via ollama | distribuida |

## Notebooks

```
notebooks/
├── factual/
│   ├── entities.ipynb       único — explosión + embed + cluster + EL de menciones
│   ├── relations.ipynb      único — explosión + embed + cluster + canonicalización de relaciones
│   ├── events.ipynb         único — explosión + embed + cluster + canonicalización de eventos
│   └── backfill.ipynb       único — construcción del grafo final
├── rhetoric/
│   ├── sentiment.ipynb      distribuido — análisis de sentimiento
│   ├── framer.ipynb         distribuido — clasificación de frames de Entman
│   ├── argument.ipynb       distribuido — extracción de argumentos de Wodak
│   └── topics.ipynb         único — modelado de tópicos
├── judge.ipynb              distribuido — clasificación de relevancia
├── scraper.ipynb            distribuido — scraping de URLs con fallback Wayback
├── kb.ipynb                 distribuido — extracción NER + RE + EE por documento
└── merge.ipynb              único — consolida outputs distribuidos y linkea argumentos al KB
```

## Orden de ejecución

```
scraper → [merge bloque 1] → judge → kb → [merge bloque 2]
                                               │
                             ┌─────────────────┼─────────────────┐
                    factual/entities  factual/relations  factual/events
                             └─────────────────┼─────────────────┘
                                        factual/backfill

              rhetoric/sentiment    rhetoric/framer    rhetoric/argument    rhetoric/topics
              └──────────────────────────────────────────────────────────────────────────┘
                                           [merge bloque 3]
```

Los notebooks de rhetoric son independientes entre sí y de la fase factual — solo requieren `noticias_relevantes.csv`. Pueden correr en paralelo desde que el filtrado de relevancia esté completo.

`factual/entities`, `factual/relations` y `factual/events` son independientes entre sí. `factual/backfill` requiere que los tres hayan terminado.

## Estructura del proyecto

```
media-framing/
├── medianalysis/
│   ├── retrieval/
│   │   ├── scraping.py         ScraperWorker — extracción con fallback Wayback Machine
│   │   └── judges.py           LLMJob, Ollama, Google, Anthropic — clasificación de relevancia
│   ├── factual/
│   │   ├── kb.py               KBuilder, KB — extracción conjunta NER + RE + EE
│   │   ├── grift.py            EntityGrifter, RelationGrifter, ExplodeEventsWorker
│   │   ├── embed.py            Embedder, MentionEmbedder, RelationEmbedder, EventEmbedder
│   │   ├── cluster.py          cluster_generic — clustering HDBSCAN
│   │   ├── canonize.py         ELWorker, RelationCanonWorker, EventCanonWorker
│   │   └── backfill.py         backfill — construcción del grafo final
│   ├── rhetoric/
│   │   ├── sentiment.py        Sentimentalist — SaBERT
│   │   ├── topics.py           fit_topics, assign_topics — BERTopic
│   │   ├── frame.py            Framer — frames de Entman (1993)
│   │   └── argument.py         ArgumentMiner — argumentos de Wodak (2003)
│   ├── distrib.py              BaseWorker, merge_workers
│   └── preprocessing.py        TextPreprocessor
├── notebooks/
│   ├── factual/
│   │   ├── backfill.ipynb
│   │   ├── entities.ipynb
│   │   ├── events.ipynb
│   │   └── relations.ipynb
│   ├── rhetoric/
│   │   ├── argument.ipynb
│   │   ├── framer.ipynb
│   │   ├── sentiment.ipynb
│   │   └── topics.ipynb
│   ├── judge.ipynb
│   ├── kb.ipynb
│   ├── merge.ipynb
│   └── scraper.ipynb
├── scripts/
│   └── validate_corpus.py      Herramienta tkinter de anotación manual
└── pipeline.ipynb              Orquestación completa del pipeline
```

## Instalación

```bash
git clone https://github.com/average-peruvian/media-framing.git
cd media-framing
pip install -e .
```

Dependencias principales: `ollama`, `sentence-transformers`, `hdbscan`, `umap-learn`, `bertopic`, `transformers`, `torch`, `pydantic`, `requests`, `newspaper3k`, `waybackpy`.

## Referencias

- Entman, R. M. (1993). Framing: Toward clarification of a fractured paradigm. *Journal of Communication*, 43(4), 51–58.
- Wodak, R. (2003). Aspects of critical discourse analysis. *Zeitschrift für Angewandte Linguistik*, 36, 5–31.
- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv:2203.05794*.