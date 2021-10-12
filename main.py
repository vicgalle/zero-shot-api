from pipeline import SemanticSearchPipeline, ZeroShotNERPipeline

TEST_SS = False
TEST_NER = True

if TEST_SS:
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

if TEST_NER:
    ner = ZeroShotNERPipeline()
    labels = ["University", "Soccer Team", "City", "Country"]
    query = ["The Complutense University of Madrid is the place in which I graduated",
            "Atletico  played against Real Madrid"]


    o = ner(query, labels)
    print(o)
    