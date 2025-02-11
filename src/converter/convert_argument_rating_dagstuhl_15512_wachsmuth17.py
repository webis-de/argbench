from common import Genres, Output, Subareas, datasets_path, read_tabular, tasks_path, Metadata, add_seed_arg, set_seed, find_topic_size_to_split
from argparse import ArgumentParser
import random
import re
import uuid

p = re.compile(r"\(([a-zA-Z]+)\)")
QUALITY_SCORES = ["Low", "Average", "High"]
DATASET_NAME = "argument_rating_dagstuhl_15512_{dimension}_wachsmuth17"

def aggregate_labels_and_split(dataset, column):
    arguments = dataset.groupby(["argument", "issue"])[["argument", "issue", column]].apply(lambda row: row[column].value_counts().index[0]).reset_index()

    df_test, df_train = find_topic_size_to_split(arguments, "issue")
    return df_test, df_train

def make_output(dataset, metadata, column, aspect_description, dataset_file, split):
    dimension = column.replace(' ', '_')
    dataset_name = DATASET_NAME.replace('{dimension}', dimension)
    output = Output(dataset_name)
    output.append_definition(f"""Judge the quality of argument according to quality aspect: {column}. 
                            Quality aspect description: {aspect_description} Possible outputs:
                            Low if arguments aspect quality is low, Average if argument's aspect quality is average,
                            High if arguments aspect quality is high. Do not explain.""")

    for row in dataset.iterrows():
        row = row[1]
        prompt = f"Argument: {row['argument']}"
        response = p.findall(row[0])[0]
        id = str(uuid.uuid4())

        output.append_instance(id, prompt, [response])

    metadata.add_dataset(dataset_file, split)
    output.append_genre(Genres.DEBATE_PORTALS)
    output.append_subarea(Subareas.QUALITY_ASSESSMENT)
    output.write_output(dataset_file)

if __name__ == "__main__":
    arg_parser = ArgumentParser()
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)

    metadata = Metadata(DATASET_NAME)

    dataset_path = str(datasets_path()
                       / "dagstuhl-15512-argquality-corpus-v2"
                       / "dagstuhl-15512-argquality-corpus-annotated.csv")

    dataset = read_tabular(dataset_path)
    dataset = dataset.dropna()

    local_acceptability_description = "A premise of an argument is acceptable if it is rationally worthy of being believed to be true."
    appropriateness_description = "Argumentation has an appropriate style if the used language supports the creation of credibility and emotions as well as if it is proportional to the issue."
    arrangement_description = "Argumentation is arranged properly if it presents the issue, the arguments, and its conclusion in the right order."
    clarity_description = "Argumentation has a clear style if it uses correct and widely unambiguous language as well as if it avoids unnecessary complexity and deviation from the issue."
    cogency_description = "An argument is cogent if it has acceptable premises that are relevant to its conclusion and that are sufficient to draw the conclusion."
    global_acceptability_description = "Argumentation is acceptable if the target audience accepts both the consideration of the stated arguments for the issue and the way they are stated."
    global_relevance_description = "Argumentation is relevant if it contributes to the issue’s resolution, i.e., if it states arguments or other information that help to arrive at an ultimate conclusion."
    global_sufficiency_description = "Argumentation is sufficient if it adequately rebuts those counterarguments to it that can be anticipated."
    reasonableness_description = "Argumentation is reasonable if it contributes to the issue’s resolution in a sufficient way that is acceptable to the target audience."
    local_relevance_description = "A premise of an argument is relevant if it contributes to the acceptance or rejection of the argument’s conclusion."
    credibility_description = "Argumentation creates credibility if it conveys arguments and similar in a way that makes the author worthy of credence."
    emotional_appeal_description = "Argumentation makes a successful emotional appeal if it creates emotions in a way that makes the target audience more open to the author’s arguments."
    local_sufficiency_description = "An argument’s premises are sufficient if, together, they give enough support to make it rational to draw its conclusion."
    effectiveness_description = "Argumentation is effective if it persuades the target audience of (or corroborates agreement with) the author’s stance on the issue."
    overall_quality_description = "How good an argument is holistically."

    tasks = [
        ("overall quality", overall_quality_description),
        ("effectiveness", effectiveness_description),
        ("local acceptability", local_acceptability_description),
        ("appropriateness", appropriateness_description),
        ("arrangement", arrangement_description),
        ("clarity", clarity_description),
        ("cogency", cogency_description),
        ("global acceptability", global_acceptability_description),
        ("global relevance", global_relevance_description),
        ("global sufficiency", global_sufficiency_description),
        ("reasonableness", reasonableness_description),
        ("local relevance", local_relevance_description),
        ("credibility", credibility_description),
        ("emotional appeal", emotional_appeal_description),
        ("sufficiency", local_sufficiency_description)
    ]
    for task_name, description in tasks:

        df_test, df_train = aggregate_labels_and_split(dataset, task_name)
        test_filename = f"argument_rating_dagstuhl_15512_{task_name.replace(' ', '_')}_test_wachsmuth17.json"
        train_filename = f"argument_rating_dagstuhl_15512_{task_name.replace(' ', '_')}_train_wachsmuth17.json"
        make_output(df_test, metadata, task_name, description, test_filename, "test")
        make_output(df_train, metadata, task_name, description, train_filename, "train")

    metadata.add_evaluation_metric("f1_macro")
    metadata.add_genre(Genres.DEBATE_PORTALS)
    metadata.add_subarea(Subareas.QUALITY_ASSESSMENT)
    metadata.write_metadata()
