from common import Output, read_tabular, Metadata, add_seed_arg, set_seed, datasets_path, Genres, Skills
from argparse import ArgumentParser
from random import shuffle, sample


dataset_name = "argument_ranking_claim_revisions_skitalinskaya23"
dataset_file = "argument_ranking_claim_revisions_{split}_skitalinskaya23.json"

mapping = {0: "Worse", 1: "Better"}

def process_data(dataset, metadata, split):
    output = Output(dataset_name)
    output.append_definition("""Given the following argument pair, is the first argument better or worse 
    than the second argument. Only respond with better or worse.""")


    for i, record in dataset.iterrows():

        prompt = f"Argument 1: {record['v2_text']}\nArgument 2: {record['v1_text']}"

        output.append_instance(record['v1_id'] + record['v2_id'], prompt, [mapping[record["label"]]])

    output.append_genre(Genres.DEBATE_PORTALS)
    output.append_subarea(Skills.QUALITY_ASSESSMENT)
    split_dataset_file = dataset_file.format(split=split)
    metadata.add_dataset(split_dataset_file,split)
    output.write_output(split_dataset_file)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    

    data_path = datasets_path() / "claim-revisions-arg-ranking" / "eacl21_extended.csv"

    metadata = Metadata(dataset_name)
    dataset = read_tabular(data_path)
    unique_claim_ids = dataset["claim_id"].unique().tolist()
    size_test_claim_ids = len(unique_claim_ids) * 2 //10


    test_claim_ids = sample(unique_claim_ids, size_test_claim_ids)
    training_claim_ids = [claim_id for claim_id in unique_claim_ids if claim_id not in test_claim_ids]

    val_claim_ids = sample(training_claim_ids, size_test_claim_ids)
    training_claim_ids = [claim_id for claim_id in training_claim_ids if claim_id not in val_claim_ids]

    df_test = dataset[dataset["claim_id"].isin(test_claim_ids)]
    df_training = dataset[dataset["claim_id"].isin(training_claim_ids)]
    df_val = dataset[dataset["claim_id"].isin(val_claim_ids)]

    process_data(df_training, metadata, "train")
    process_data(df_test, metadata, "test")
    process_data(df_val, metadata, "val")

    metadata.add_evaluation_metric("fscore")
    metadata.add_genre(Genres.DEBATE_PORTALS)
    metadata.add_skill(Skills.QUALITY_ASSESSMENT)
    metadata.write_metadata()
