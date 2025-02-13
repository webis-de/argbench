from common import Genres, Output, Subareas, read_tabular, Metadata, add_seed_arg, set_seed, datasets_path
from argparse import ArgumentParser
import pandas as pd
import uuid


dataset_name = "claim_improvement_detection_claim_revisions_skitalinskaya23"
dataset_file = "claim_improvement_detection_claim_revisions_skitalinskaya23.json"


def process_data(dataset, metadata):
    output = Output(dataset_name)
    output.append_definition("Judge if original claim was improved in revised claim. Possible outputs: Improved if revised claim is better than original, Not Improved if revised claim did not improve original claim.")

    original_dataset = dataset.loc[dataset.groupby('claim_id')['revision_id'].idxmin()]
    final_claims = dataset.loc[dataset.groupby('claim_id')['revision_id'].idxmax()]
    claims_dataset = pd.merge(
        original_dataset, final_claims[["claim_id", "claim_text"]],
        on="claim_id",
        suffixes=("_original", "_revised"))

    for row in claims_dataset.iterrows():
        row = row[1]
        prompt_improve = f"Original claim: {row['claim_text_original']}\nRevised claim: {row['claim_text_revised']}"
        id_improve = str(uuid.uuid4())
        output.append_instance(id_improve, prompt_improve, ["Improved"])
        prompt_no_improve = f"Original claim: {row['claim_text_revised']}\nRevised claim: {row['claim_text_original']}"
        id_no_improve = str(uuid.uuid4())
        output.append_instance(id_no_improve, prompt_no_improve, ["Not Improved"])

    metadata.add_dataset(dataset_file)
    output.append_genre(Genres.DEBATES)
    output.append_subarea(Subareas.MINING)
    output.write_output(dataset_file)


if __name__ == "__main__":
    arg_parser = ArgumentParser(description="What dataset will be processed?")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)

    data_path = datasets_path() / "claim-revisions-arg-ranking" / "eacl21_list_for_ranking.csv"

    metadata = Metadata(dataset_name)

    dataset = read_tabular(data_path)

    process_data(dataset, metadata)

    metadata.add_genre(Genres.DEBATES)
    metadata.add_skill(Subareas.MINING)
    
    metadata.write_metadata()
