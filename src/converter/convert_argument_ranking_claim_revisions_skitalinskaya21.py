from common import Output, read_tabular, Metadata, add_seed_arg, set_seed, datasets_path
from argparse import ArgumentParser
from random import shuffle


dataset_name = "argument_ranking_claim_revisions_skitalinskaya23"
dataset_file = "argument_ranking_claim_revisions_skitalinskaya23.json"


def process_data(dataset, metadata):
    output = Output(dataset_name)
    output.append_definition("Rank the following reformulations of the same claim according to their quality. All the arguments should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., [4] > [2]. Only respond with the ranking results, do not say any word or explain.")

    dataset_claims = dataset["claim_id"].unique()

    for claim in dataset_claims:
        print(claim)
        prompt = "Claims:"
        response = ""
        claim_revisions = [
            (row["claim_id"], row["claim_text"], row["revision_id"])
            for _, row in dataset[dataset["claim_id"] == claim].iterrows()
        ]
        shuffle(claim_revisions)

        claim_revisions = [(i, claim_id, text, rev_id) for i, (claim_id, text, rev_id) in enumerate(claim_revisions)]

        for i, _, text, _ in claim_revisions:
            prompt += f" [{i}] {text}"

        claim_revisions = sorted(claim_revisions, key=lambda x: x[3], reverse=True)

        response = " > ".join([f"[{i}]" for i, _, _, _ in claim_revisions])

        output.append_instance(claim_revisions[0][1], prompt, [response])

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
