from uuid import uuid4
from common import Output, datasets_path, read_tabular, tasks_path, Metadata, add_seed_arg, set_seed, Genres, Subareas
from argparse import ArgumentParser

dataset_name = "fallacy_detection_logic_goffredo23"

def process_data(data_path, data_name, data_split, metadata):
    dataset = read_tabular(data_path)
    output = Output(dataset_name)
    output.append_definition("Identify logical fallacies present in the article text. " +
                             "Possible fallacies:" +
                             "intentional - some intentional (sometimes subconscious) action/choice to incorrectly support an argument. " +
                             "faulty generalization - an informal fallacy wherein a conclusion is drawn about all or many instances of a phenomenon on the basis of one or a few instances of that phenomenon. is an example of jumping to conclusions. " +
                             "ad hominem - instead of addressing someone's argument or position, you irrelevantly attack the person or some aspect of the person who is making the argument. " +
                             "fallacy of relevance - introducing premises or conclusions that have nothing to do with the subject matter. " +
                             "false causality - statement that jumps to a conclusion implying a causal relationship without supporting evidence. " +
                             "appeal to emotion - manipulation of the recipient's emotions in order to win an argument. " +
                             "fallacy of extension - attacking an exaggerated or caricatured version of your opponent's position. " +
                             "fallacy of logic - an error in the logical structure of an argument. " +
                             "fallacy of credibility - attempts to disprove an argument by attacking the character of the speaker. " +
                             "ad populum - a fallacious argument which is based on affirming that something is real or better because the majority thinks so" +
                             "equivocation - when a key term or phrase in an argument is used in an ambiguous way, with one meaning in one portion of the argument and then another meaning in another portion of the argument. " +
                             "circular reasoning - when the end of an argument comes back to the beginning without having proven itself. " +
                             "false dilemma - presenting only two options or sides when there are many options or sides.")

    for row in dataset.iterrows():
        row = row[1]
        id = str(uuid4())
        prompt = f"Article: {row['source_article']}"
        if "logical_fallacies" in row:
            fallacies = row["logical_fallacies"]
        elif "updated_label" in row:
            fallacies = row["updated_label"]
        else:
            raise Exception("No labels found!")
        output.append_instance(id, prompt, [fallacies])

    output.append_genre(Genres.DEBATE_PORTALS)
    output.append_genre(Genres.DEBATES)
    output.append_subarea(Subareas.MINING)
    output.write_output(data_name)
    metadata.add_dataset(data_name, data_split)


if __name__ == "__main__":
    # Input arguments for dataset generation
    arg_parser = ArgumentParser(description="What dataset will be processed?")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args) # Seed random number generation

    data_path = datasets_path() / "logic" # path to data

    metadata = Metadata(dataset_name)

    process_data(data_path / "climate_train.csv", "fallacy_detection_logic_climate_train_goffredo23.json", "train", metadata)
    process_data(data_path / "climate_test.csv", "fallacy_detection_logic_climate_test_goffredo23.json", "test", metadata)
    process_data(data_path / "climate_dev.csv", "fallacy_detection_logic_climate_dev_goffredo23.json", "dev", metadata)
    process_data(data_path / "edu_train.csv", "fallacy_detection_logic_edu_train_goffredo23.json", "train", metadata)
    process_data(data_path / "edu_test.csv", "fallacy_detection_logic_edu_test_goffredo23.json", "test", metadata)
    process_data(data_path / "edu_dev.csv", "fallacy_detection_logic_edu_dev_goffredo23.json", "dev", metadata)

    metadata.add_evaluation_metric("f1_macro")
    metadata.add_genre(Genres.DEBATE_PORTALS)
    metadata.add_genre(Genres.DEBATES)
    metadata.add_subarea(Subareas.MINING)
    metadata.write_metadata()
