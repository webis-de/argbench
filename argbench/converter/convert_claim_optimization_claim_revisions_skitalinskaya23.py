from common import Genres, Output, Skills, datasets_path, read_tabular, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser
import pandas as pd


dataset_name = "claim_optimization_claim_revisions_skitalinskaya23"


def process_split(dataset, metadata, split):
    output = Output(dataset_name)
    dataset_file = f"claim_optimization_claim_revisions_{split}_skitalinskaya23.json"
    output.append_definition("Given the following input argumentative claim with context information on the " +
                             "debate, rewrite the claim such that the output claim improves upon input claim in terms of text quality " +
                             "and argument quality, and preserves the meaning of the claim as far as possible.  Do not explain.")

    unrevised_docs = dataset[dataset["revision_id"] == 1][["claim_id", "revision_id", "claim_text", "thesis"]]
    last_revision_docs = dataset[dataset["revision_id"] == dataset["max_revision_id"]][["claim_id", "revision_id", "claim_text"]]

    data = pd.merge(unrevised_docs, last_revision_docs, suffixes=("_unr", "_rev"), on="claim_id")

    for claim in data.iterrows():
        claim = claim[1]
        unrevised_text = claim["claim_text_unr"]
        revised_text = claim["claim_text_rev"]
        claim_id = claim["claim_id"]
        thesis = claim["thesis"]
        instance_input = f"Thesis: {thesis}\nClaim: {unrevised_text}"
        output.append_instance(claim_id, instance_input, [revised_text])

    metadata.add_dataset(dataset_file, split)

    output.append_genre(Genres.DEBATE_PORTALS)
    output.append_subarea(Skills.QUALITY_ASSESSMENT)
    output.write_output(dataset_file)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)

    data_path = datasets_path() / "claim-revisions" / "acl23_revised.csv"

    metadata = Metadata(dataset_name)

    dataset = read_tabular(data_path)

    print("Train")
    process_split(dataset[dataset["data_split"] == "dev"], metadata, "val")
    process_split(dataset[dataset["data_split"] == "train"], metadata, "train")
    print("Test")
    process_split(dataset[dataset["data_split"] == "test"], metadata, "test")
    metadata.add_evaluation_metric("generation-score")


    
    metadata.add_genre(Genres.DEBATE_PORTALS)
    metadata.add_skill(Skills.QUALITY_ASSESSMENT)
    metadata.write_metadata()
