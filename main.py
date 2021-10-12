from pipeline import SemanticSearchPipeline

# MODELNAME = "distilbert-base-uncased"
# MODELNAME = "vicgalle/xlm-roberta-large-xnli-anli"
MODELNAME = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"

searcher = SemanticSearchPipeline(model=MODELNAME, device='cuda')

o = searcher(
    query="Mamá, te quiero", context=["Hola, qué tal?", "Adios.", "Mamá, te quiero"]
)
print(o)


o = searcher(
    query=["Mamá, te quiero", "Hasta luego"],
    context=["Hola, qué tal?", "Adios.", "Mamá, te quiero"],
    temperature=0.01,
)
print(o)