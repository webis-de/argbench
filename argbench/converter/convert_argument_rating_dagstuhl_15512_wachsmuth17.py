import pandas as pd

from common import Genres, Output, Skills, datasets_path, read_tabular, tasks_path, Metadata, add_seed_arg, set_seed, find_topic_size_to_split
from argparse import ArgumentParser
import random
import re
import uuid

p = re.compile(r"\(([a-zA-Z]+)\)")
QUALITY_SCORES = ["Low", "Average", "High"]
DATASET_NAME = "argument_rating_dagstuhl_15512_wachsmuth17"

def aggregate_labels_and_split(dataset, tasks):
    all_arguments_dfs_aggregated = []
    for task_name, description in tasks:

        arguments = dataset.groupby(["argument", "issue", "#id"])[["argument", "issue", "#id", task_name]].apply(lambda row: row[task_name].value_counts().index[0]).reset_index()
        arguments["quality-aspect"] = task_name
        arguments["quality-aspect-description"] = description
        all_arguments_dfs_aggregated.append(arguments)
    df_arguments_aggregated = pd.concat(all_arguments_dfs_aggregated)
    df_argument_ids = dataset[["#id", "issue"]].drop_duplicates()
    df_test, df_train = find_topic_size_to_split(df_argument_ids, "issue", 0.2)
    df_val, df_train = find_topic_size_to_split(df_train, "issue", 0.25)
    all_dis = len(dataset["argument"].drop_duplicates())
    print(f"all ids {all_dis}")
    print(f"all {len(df_argument_ids)}")
    print(f" test {len(df_test)}")
    print(f" val {len(df_val)}")
    print(f"train {len(df_train)}")
    df_test = df_arguments_aggregated[df_arguments_aggregated["#id"].isin(df_test["#id"])]
    df_train = df_arguments_aggregated[df_arguments_aggregated["#id"].isin(df_train["#id"])]
    df_val = df_arguments_aggregated[df_arguments_aggregated["#id"].isin(df_val["#id"])]

    return df_test, df_val, df_train

def make_output(dataset, metadata, dataset_file, split):

    output = Output(DATASET_NAME)
    output.append_definition(f"""Judge the quality of the argument according to quality aspect. Possible outputs:
                            Low if arguments aspect quality is low, Average if argument's aspect quality is average,
                            High if arguments aspect quality is high.""")

    for row in dataset.iterrows():
        row = row[1]
        quality_aspect = row["quality-aspect"]
        quality_aspect_description = row["quality-aspect-description"]
        prompt = f"Argument: {row['argument']}\nQuality Aspect: {quality_aspect}\nQuality Aspect Definition: {quality_aspect_description}"

        response = p.findall(row[0])[0]
        id = str(uuid.uuid4())

        output.append_instance(id, prompt, [response])

    metadata.add_dataset(dataset_file, split)
    output.append_genre(Genres.DEBATE_PORTALS)
    output.append_subarea(Skills.QUALITY_ASSESSMENT)
    output.write_output(dataset_file)

if __name__ == "__main__":
    arg_parser = ArgumentParser()
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]

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

    metadata = Metadata(DATASET_NAME)
    test_filename = f"argument_rating_dagstuhl_15512_test_wachsmuth17.json"
    train_filename = f"argument_rating_dagstuhl_15512_train_wachsmuth17.json"
    val_filename = f"argument_rating_dagstuhl_15512_val_wachsmuth17.json"

    df_test, df_val, df_train = aggregate_labels_and_split(dataset, tasks)
    print(f"length of test is {len(df_test)}")
    print(f"length of train is {len(df_train)}")
    print(f"length of val is {len(df_val)}")
    make_output(df_test, metadata,  test_filename, "test")
    make_output(df_val, metadata,  val_filename, "val")
    make_output(df_train, metadata, train_filename, "train")






    metadata.add_evaluation_metric("fscore")
    metadata.add_genre(Genres.DEBATE_PORTALS)
    metadata.add_skill(Skills.QUALITY_ASSESSMENT)
    metadata.write_metadata()
