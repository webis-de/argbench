from common import Output, read_tabular, Metadata, add_seed_arg, set_seed, datasets_path
from argparse import ArgumentParser
import uuid


dataset_name = "suboptimal_claim_detection_claim_revisions_skitalinskaya23"
dataset_file = "suboptimal_claim_detection_claim_revisions_skitalinskaya23.json"


def process_data(dataset, metadata):
    output = Output(dataset_name)
    output.append_definition("Judge if claim can be improved by revising it. Possible outputs: Improvable if revision should be made, Non-Improvable if no revision is necessary.")

    original_claims = dataset.loc[dataset.groupby('claim_id')['revision_id'].idxmin()]
    final_claims = dataset.loc[dataset.groupby('claim_id')['revision_id'].idxmax()]

    for row in original_claims.iterrows():
        row = row[1]
        prompt = f"Claim: {row['claim_text']}"
        id = str(uuid.uuid4())
        output.append_instance(id, prompt, ["Improvable"])

    for row in final_claims.iterrows():
        row = row[1]
        prompt = f"Claim: {row['claim_text']}"
        id = str(uuid.uuid4())
        output.append_instance(id, prompt, ["Non-Improvable"])

    metadata.add_dataset(dataset_file)
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

    metadata.add_evaluation_metric("f1_macro")
    metadata.write_metadata()
