from common import Output, read_tabular, Metadata, add_seed_arg, set_seed, datasets_path, Genres, Subareas
from argparse import ArgumentParser
from random import shuffle, sample


dataset_name = "argument_ranking_claim_revisions_skitalinskaya23"
dataset_file = "argument_ranking_claim_revisions_{split}_skitalinskaya23.json"

mapping = {0: "Worse", 1: "Better"}

def process_data(dataset, metadata, split):
    output = Output(dataset_name)
    output.append_definition("""Given the following argument pair, is the first argument better or worse 
    than the second argument. Only respond with better or worse, do not say any word or explain. 
    Only respond with better or worse, do not say any word or explain.""")


    for i, record in dataset.iterrows():

        prompt = f"Argument 1: {record['v2_text']}\nArgument 2: {record['v1_text']}"

        output.append_instance(record['v1_id'] + record['v2_id'], prompt, [mapping[record["label"]]])

    output.append_genre(Genres.DEBATES)
    output.append_subarea(Subareas.RANKING)
    split_dataset_file = dataset_file.format(split=split)
    metadata.add_dataset(split_dataset_file)
    output.write_output(split_dataset_file)


if __name__ == "__main__":
    arg_parser = ArgumentParser(description="What dataset will be processed?")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)

    data_path = datasets_path() / "claim-revisions-arg-ranking" / "eacl21_extended.csv"

    metadata = Metadata(dataset_name)

    dataset = read_tabular(data_path)
    unique_claim_ids = dataset["claim_id"].unique().tolist()
    size_test_claim_ids = len(unique_claim_ids) * 2 //10
    test_claim_ids = sample(unique_claim_ids, size_test_claim_ids)

    df_test = dataset[dataset["claim_id"].isin(test_claim_ids)]
    print(len(df_test))
    df_training = dataset[~dataset["claim_id"].isin(test_claim_ids)]
    print(len(df_training))
    process_data(df_training, metadata, "train")
    process_data(df_test, metadata, "test")

    metadata.add_evaluation_metric("f1_macro")
    metadata.add_genre(Genres.DEBATES)
    metadata.add_subarea(Subareas.RANKING)
    metadata.write_metadata()
