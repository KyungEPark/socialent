from util import *
import openai
from openai import OpenAI
import requests
import pandas as pd
import os


client = OpenAI(
    api_key = os.environ.get("OPENAI_API_KEY"),
)


# Prompt for Hybrid Identity narratives
hybprompt = "Analysiere den Website-Text daraufhin, ob das soziale Unternehmen sowohl seine soziale Mission als auch seine wirtschaftliche Kompetenz betont. Achte darauf, ob explizit darauf hingewiesen wird, dass soziale Wirkung und unternehmerischer Erfolg gleichermaßen wichtig sind. Suche nach Formulierungen, die auf eine bewusste Verbindung von moralischer Verantwortung und unternehmerischer Professionalität hindeuten. Prüfe, ob sowohl messbare soziale Ziele als auch betriebliche Effizienz oder Wachstum hervorgehoben werden, um verschiedene Anspruchsgruppen anzusprechen."
transprompt = "Untersuche, ob das Unternehmen auf seiner Website Zertifizierungen (z. B. B Corp), Impact-Reports oder die Einhaltung von Standards wie CSRD hervorhebt. Achte auf Hinweise, dass durch diese Maßnahmen Transparenz geschaffen und Vertrauen aufgebaut werden soll. Suche nach Textstellen, in denen die Veröffentlichung von Berichten, die Teilnahme an Audits oder die Übernahme von regulatorischen Vorgaben betont wird. Prüfe, ob diese Elemente genutzt werden, um die Glaubwürdigkeit und Verantwortlichkeit des Unternehmens nach außen zu demonstrieren."
storyprompt = "Analysiere, ob auf der Website persönliche Geschichten von Gründer:innen, Erfahrungsberichte von Begünstigten oder emotionale Fallstudien, Bilder und Videos präsentiert werden. Achte darauf, ob diese Inhalte genutzt werden, um die Mission und Werte des Unternehmens authentisch und glaubwürdig zu vermitteln. Suche nach Hinweisen auf die Lebenswege der Gründer:innen, direkte Zitate oder Geschichten von Betroffenen sowie nach visuellen Elementen, die Nähe und Vertrauen schaffen. Prüfe, ob das Storytelling dazu dient, die emotionale Bindung zu den Stakeholdern zu stärken."
socialprompt = "Untersuche, ob das Unternehmen explizit auf aktuelle gesellschaftliche Bewegungen oder Debatten wie Klimaschutz, Diversität, Inklusion oder soziale Gerechtigkeit Bezug nimmt. Achte darauf, ob das Unternehmen sich mit diesen Themen sichtbar positioniert oder Partnerschaften mit entsprechenden Initiativen eingeht. Suche nach Textstellen, in denen Engagement für oder Unterstützung von gesellschaftlichen Trends und Werten betont wird. Prüfe, ob diese Bezüge genutzt werden, um normative Legitimität zu gewinnen und sich als Teil einer größeren Bewegung zu präsentieren."

prompts = [hybprompt, transprompt, storyprompt, socialprompt]

# Example texts for testing - TODO: CALL real texts from the website
examples = [
    "Wir sind ein junges Gründerteam aus drei Personen. Nachdem jeder von uns seine ganz eigenen Erfahrung mit der Verschmutzung der Umwelt gemacht hat, kamen wir auf die Idee selbst Akzente zu setzen, um etwas aktiv dagegen zu tun. Für die Produktion greifen wir auf die Expertise eines exklusiven Partnernetzwerks zurück. Neben einem traditionsreichen deutschen Glashersteller, steht uns ein Unternehmen zur Seite, das sich durch mehr als 25 Jahre Erfahrung in der Produktion von Spezialgläsern auszeichnet.",
    "MELINA BUCHER Handtaschen vereinen Ethik mit Ästhetik. Handgefertigt in unserem Atelier in Süddeutschland, werden Jahrhunderte altes Täschnerhandwerk mit zukunftsweisenden Materialien kombiniert. Meisterliche Qualität trifft auf zirkuläres Design - für einzigartige Taschen im Einklang mit der Natur. ",
    "Der Starkmacher e.V. ist eine Art Ideen- und Projektbörse von und für Menschen, denen die Förderung und Entfaltung von Potentialen in Jugendlichen am Herzen liegt. Ziel ist es, ehrenamtliche und professionelle Kräfte in einer guten Zusammenarbeit zu bündeln, Netzwerke zu schaffen und auszubauen und wichtige Kompetenzen im Bereich der Jugend- und Bildungsarbeit wirksam auszuschöpfen. Die Projekte des Starkmacher e.V. werden von verschiedenen EU-Förderprogrammen unterstützt.",
    "Wir sind profitabel aber auch menschenfreundlich.",
    "Atemschutz ist ein Menschenrecht."
]

# Function to call OpenAI API with the hybrid prompt
# Collect responses in a dictionary
results = {f"Prompt {i+1}": [] for i in range(len(prompts))}

for i, prompt in enumerate(prompts):
    for example in examples:
        response = call_openai(prompt, example)
        results[f"Prompt {i+1}"].append(response)

# Create DataFrame: rows=examples, columns=prompts
df = pd.DataFrame(results, index=[f"Example {i+1}" for i in range(len(examples))])
df.to_csv('data/prompt_results.csv', index=True)
