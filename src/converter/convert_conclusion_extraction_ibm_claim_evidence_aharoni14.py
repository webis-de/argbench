from common import Genres, Output, Subareas, datasets_path, read_tabular, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser
import uuid
import re

dataset_name = "conclusion_extraction_ibm_claim_evidence_aharoni14"
dataset_file = "conclusion_extraction_ibm_claim_evidence_aharoni14.json"

space_remover = re.compile("\s+")

if __name__ == "__main__":
    arg_parser = ArgumentParser(description="What dataset will be processed?")
    add_seed_arg(arg_parser)
    arg_parser.add_argument("-f", "--front_add", default=1, type=int, help="Add # of articles in front of found evidence paragraph")
    arg_parser.add_argument("-b", "--back_add", default=1, type=int, help="Add # of articles in back of found evidence paragraph")
    args = arg_parser.parse_known_args()[0]
    set_seed(args)

    data_path = datasets_path() / "ibm-claim-evidence" / "2014_7_18_ibm_CDEdata.csv"
    articles_folder = datasets_path() / "ibm-claim-evidence" / "wiki12_articles"

    metadata = Metadata(dataset_name)
    output = Output(dataset_name)
    output.append_definition("Given the following Wikipedia section, extract all claims  the given context. A claim is an assertion that an argument tries to prove. ")

    dataset = read_tabular(data_path)

    dataset["Article"] = dataset["Article"].str.replace(" ", "_")
    dataset["wiki_article"] = ""

    for article_path in articles_folder.iterdir():
        if not (dataset["Article"] == article_path.name).sum():
            continue
        with open(article_path, "r") as f:
            article_contents = f.readlines()
            article_contents = [c for c in article_contents if c != "\n"]
            article_match = [space_remover.sub("", c.lower()) for c in article_contents]

        article_data = dataset[dataset["Article"] == article_path.name]

        for row_idx, article_row in article_data.iterrows():
            evidence_match = space_remover.sub("", article_row["CDE"].lower())
            article_matched = []
            for i, am in enumerate(article_match):
                if (evidence_match in am or
                    am in evidence_match or
                    evidence_match[:5] in am or
                    evidence_match[-5:] in am):
                    article_matched.append(i)
            if len(article_matched) < 1:
                print(article_contents)
                print(article_row["CDE"])
                print(article_matched)
                raise Exception()

            article_window_start = article_matched[0] - args.back_add
            article_window_end = article_matched[-1] + args.front_add
            if article_window_end > (len(article_contents) - 1):
                article_window_end = len(article_contents) - 1
            if article_window_start < 0:
                article_window_start = 0

            collected_article = "".join(article_contents[article_window_start:article_window_end])
            dataset.loc[row_idx, "wiki_article"] = collected_article

    for row in dataset.iterrows():
        idx = row[0]
        row = row[1]
        id = str(uuid.uuid4())
        topic = row["Topic"]
        prompt = f"Topic: {topic} \nArticle: {row['wiki_article']}"
        label = row["Claim"]
        output.append_instance(id, prompt, [label])

    output.append_genre(Genres.WIKIPEDIA)
    output.append_subarea(Subareas.MINING)

    metadata.add_genre(Genres.WIKIPEDIA)
    metadata.add_subarea(Subareas.MINING)
    metadata.add_dataset(dataset_file)
    output.write_output(dataset_file)

    metadata.add_evaluation_metric("f1_macro")
    metadata.write_metadata()
