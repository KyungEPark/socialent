## Categorization validation using keywords and corpora
import os
import re
import pandas as pd
import numpy as np
from util import *
from sentence_transformers import SentenceTransformer, util


# Call list of keywords
keywords = call_csv('data/keywordlist.csv')
print(keywords.head())

# Run preprocessing
category_keywords = preprocess_keywords(keywords)

# Call text corpora (to be updated after the text corpora is ready)
examples = [
    "Wir sind ein junges Gründerteam aus drei Personen. Nachdem jeder von uns seine ganz eigenen Erfahrung mit der Verschmutzung der Umwelt gemacht hat, kamen wir auf die Idee selbst Akzente zu setzen, um etwas aktiv dagegen zu tun. Für die Produktion greifen wir auf die Expertise eines exklusiven Partnernetzwerks zurück. Neben einem traditionsreichen deutschen Glashersteller, steht uns ein Unternehmen zur Seite, das sich durch mehr als 25 Jahre Erfahrung in der Produktion von Spezialgläsern auszeichnet.",
    "MELINA BUCHER Handtaschen vereinen Ethik mit Ästhetik. Handgefertigt in unserem Atelier in Süddeutschland, werden Jahrhunderte altes Täschnerhandwerk mit zukunftsweisenden Materialien kombiniert. Meisterliche Qualität trifft auf zirkuläres Design - für einzigartige Taschen im Einklang mit der Natur. ",
    "Der Starkmacher e.V. ist eine Art Ideen- und Projektbörse von und für Menschen, denen die Förderung und Entfaltung von Potentialen in Jugendlichen am Herzen liegt. Ziel ist es, ehrenamtliche und professionelle Kräfte in einer guten Zusammenarbeit zu bündeln, Netzwerke zu schaffen und auszubauen und wichtige Kompetenzen im Bereich der Jugend- und Bildungsarbeit wirksam auszuschöpfen. Die Projekte des Starkmacher e.V. werden von verschiedenen EU-Förderprogrammen unterstützt.",
    "Wir sind profitabel aber auch menschenfreundlich.",
    "Atemschutz ist ein Menschenrecht."
]

## Get scores based on sentence embeddings - get result using sentence-transformers

# Load embedding model (discuss best for German)
model = SentenceTransformer("distiluse-base-multilingual-cased-v2")


# Embed keywords and average per category
category_vectors = {}
for category, keywords in category_keywords.items():
    keyword_embeddings = model.encode(keywords, convert_to_tensor=True)
    category_vector = keyword_embeddings.mean(dim=0)
    category_vectors[category] = category_vector

# Embed the texts
examples = [preprocess_text(text) for text in examples]
text_embeddings = model.encode(examples, convert_to_tensor=True)

# Compute similarity between each text and each category
scores = []
for i, text_vec in enumerate(text_embeddings):
    row = {'text': examples[i]}
    for category, cat_vec in category_vectors.items():
        similarity = util.cos_sim(text_vec, cat_vec).item()
        row[category + '_score'] = similarity
        row[category] = 'Yes' if similarity > 0.5 else 'No'
    scores.append(row)
## NOTE: We should play with the threshold when comparing with the prompt version

# Create DataFrame
df_semantic = pd.DataFrame(scores)

# Save results
df_semantic.to_csv('data/semantic_scores.csv', index=False)