from common import Genres, Subareas, Output, datasets_path, read_tabular, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser
import uuid
from nltk.tokenize import sent_tokenize
import re
from nltk.tokenize.punkt import PunktSentenceTokenizer
dataset_name = "conclusion_extraction_ibm_claim_evidence_levy14"
dataset_file = "conclusion_extraction_ibm_claim_evidence_levy.json"
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

space_remover = re.compile("\s+")

if __name__ == "__main__":
    arg_parser = ArgumentParser(description="What dataset will be processed?")
    add_seed_arg(arg_parser)
    arg_parser.add_argument("-f", "--front_add", default=1, type=int, help="Add # of articles in front of found evidence paragraph")
    arg_parser.add_argument("-b", "--back_add", default=1, type=int, help="Add # of articles in back of found evidence paragraph")
    args = arg_parser.parse_known_args()[0]
    set_seed(args)

    data_path = datasets_path() / "ibm-claim-evidence" / "2014_7_18_ibm_CDCdata.csv"
    articles_folder = datasets_path() / "ibm-claim-evidence" / "wiki12_articles"

    metadata = Metadata(dataset_name)
    output = Output(dataset_name)
    output.append_definition("""Split the following Wikipedia section into sentences that constitute claims and those that are not.
    A claim is an assertion that an argument tries to prove and holds a stance on the given topic.
     Prepend each sentence that contains a claim with Claim: and a sentence that does not contain claim with Not-Claim.
    """)

    dataset = read_tabular(data_path)

    dataset["Article"] = dataset["Article"].str.replace(" ", "_")
    dataset["wiki_article"] = ""

    for article_path in articles_folder.iterdir():
        if not (dataset["Article"] == article_path.name).sum():
            continue
        with open(article_path, "r") as f:
            article_contents = f.readlines()
            article_contents = [c for c in article_contents if c != "\n"]
            article_text = " ".join(article_contents)
            sentences = sent_tokenize(article_text)
            sentence_cleaned = [space_remover.sub("", c.lower()) for c in sentences]

        article_data = dataset[dataset["Article"] == article_path.name]
        iterator = article_data.iterrows()
        _ , article_row = next(iterator)
        topic = article_row["Topic"]

        iterator_exhausted = False
        section_size = 10
        section_indices = range(0,len(sentences), section_size)
        sections_sentences = []
        sections_sentence_cleaned = []

        for section_index in section_indices:
            if section_index + section_size < len(sentences):
                section_index_end = len(sentences)
            else:
                section_index_end = section_index+section_size

            sections_sentences.append(sentences[section_index:section_index_end])
            sections_sentence_cleaned.append(sentence_cleaned[section_index:section_index_end])

        for section_idx, section_sentence_cleaned in enumerate(sections_sentence_cleaned):
            data_points = ""
            section_sentence = sections_sentences[section_idx]
            section_text = " ".join(section_sentence)
            for i, am in enumerate(section_sentence_cleaned):
                for _, article_row in article_data.iterrows():
                    clamin_match = space_remover.sub("", article_row["Claim"].lower())
                    if clamin_match in am or am in clamin_match or similar(clamin_match,am)>0.9:
                        data_points += f"Claim: {section_sentence[i]}\n"
                    else:
                        data_points += f"Not-claim: {section_sentence[i]}\n"

            prompt = f"Topic: {topic} \nArticle: {section_text}"
            output.append_instance(1, prompt, [data_points])

    output.append_genre(Genres.WIKIPEDIA)
    output.append_subarea(Subareas.MINING)

    metadata.add_genre(Genres.WIKIPEDIA)
    metadata.add_subarea(Subareas.MINING)
    metadata.add_dataset(dataset_file)
    output.write_output(dataset_file)

    metadata.add_evaluation_metric("f1_macro")
    metadata.write_metadata()
