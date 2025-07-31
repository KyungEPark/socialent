from util import *
import huggingface
import argparse
import os

def main(model, savefile):
    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")

    # Call data for the analysis
    texts = pd.read_parquet('data/subsettext100_socialco_20250718.parquet')

    # Prompt for Hybrid Identity narratives
    hybprompt = "Analysiere den Website-Text daraufhin, ob das soziale Unternehmen sowohl seine soziale Mission als auch seine wirtschaftliche Kompetenz betont. Achte darauf, ob explizit darauf hingewiesen wird, dass soziale Wirkung und unternehmerischer Erfolg gleichermaßen wichtig sind. Suche nach Formulierungen, die auf eine bewusste Verbindung von moralischer Verantwortung und unternehmerischer Professionalität hindeuten. Prüfe, ob sowohl messbare soziale Ziele als auch betriebliche Effizienz oder Wachstum hervorgehoben werden, um verschiedene Anspruchsgruppen anzusprechen."
    transprompt = "Untersuche, ob das Unternehmen auf seiner Website Zertifizierungen (z. B. B Corp), Impact-Reports oder die Einhaltung von Standards wie CSRD hervorhebt. Achte auf Hinweise, dass durch diese Maßnahmen Transparenz geschaffen und Vertrauen aufgebaut werden soll. Suche nach Textstellen, in denen die Veröffentlichung von Berichten, die Teilnahme an Audits oder die Übernahme von regulatorischen Vorgaben betont wird. Prüfe, ob diese Elemente genutzt werden, um die Glaubwürdigkeit und Verantwortlichkeit des Unternehmens nach außen zu demonstrieren."
    storyprompt = "Analysiere, ob auf der Website persönliche Geschichten von Gründer:innen, Erfahrungsberichte von Begünstigten oder emotionale Fallstudien, Bilder und Videos präsentiert werden. Achte darauf, ob diese Inhalte genutzt werden, um die Mission und Werte des Unternehmens authentisch und glaubwürdig zu vermitteln. Suche nach Hinweisen auf die Lebenswege der Gründer:innen, direkte Zitate oder Geschichten von Betroffenen sowie nach visuellen Elementen, die Nähe und Vertrauen schaffen. Prüfe, ob das Storytelling dazu dient, die emotionale Bindung zu den Stakeholdern zu stärken."
    socialprompt = "Untersuche, ob das Unternehmen explizit auf aktuelle gesellschaftliche Bewegungen oder Debatten wie Klimaschutz, Diversität, Inklusion oder soziale Gerechtigkeit Bezug nimmt. Achte darauf, ob das Unternehmen sich mit diesen Themen sichtbar positioniert oder Partnerschaften mit entsprechenden Initiativen eingeht. Suche nach Textstellen, in denen Engagement für oder Unterstützung von gesellschaftlichen Trends und Werten betont wird. Prüfe, ob diese Bezüge genutzt werden, um normative Legitimität zu gewinnen und sich als Teil einer größeren Bewegung zu präsentieren."

    prompts = [hybprompt, transprompt, storyprompt, socialprompt]

    # Function to call Huggingface with the hybrid prompt
    for row in texts.itertuples():
        text = row.text  # Assuming 'text' is the column name containing the text data
        for prompt in prompts:
            response = load_huggingface_model_locked(prompt, text, model, hf_token)
            colname = prompts.index(prompt)
            colnames = ['hybrid_identity', 'transparency', 'storytelling', 'social_reference']
            if colnames[colname] not in texts.columns:
                texts[colnames[colname]] = None
            texts.at[row.Index, colnames[colname]] = response
    
    # Save the results to a file
    savefile = os.path.join('/data', os.path.basename(savefile))
    os.makedirs(os.path.dirname(savefile), exist_ok=True)
    texts.to_parquet(savefile, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, required=True,
                        help="Path to the data file")
    parser.add_argument("--n", type=int, required=True,
                        help="Number of examples used")
    parser.add_argument("--labeled_file", type=str, required=True,
                        help="Path to the labeled file")    
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to the performance file")
    args = parser.parse_args()

    main(args.filename, args.n, args.labeled_file, args.output_file)