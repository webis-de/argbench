import random
from common import Output, datasets_path, read_tabular, tasks_path, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser
import re
import uuid

DATASET_NAME = "wachsmuth_dagstuhl_quality_pairvise_rank"
p = re.compile(r"\(([a-zA-Z]+)\)")
QUALITY_MAPPING = {
    "Low": 1,
    "Average": 2,
    "High": 3
}

RANK_MAPPING = [
    "[0] > [1]",
    "[1] > [0]",
    "[0] = [1]"
]

def make_output(dataset, metadata, column, aspect_description, dataset_name):
    output = Output(DATASET_NAME)

    output.append_definition(f"Rank the following arguments on the given topic according to quality aspect: {column}. Quality aspect description: {aspect_description} Both arguments should be included and listed using identifiers, in descending order of relevance. The output format should be: [0] > [1], [1] > [0] or [0] = [1] if both arguments are equally good. Only respond with the ranking results, do not say any word or explain.")

    arguments = dataset.groupby(["argument"]).apply(lambda row: row[column].value_counts().index[0]).reset_index()

    for row in arguments.iterrows():
        row = row[1]
        compare_argument = arguments.sample(1).iloc[0]
        prompt = f"Arguments:\n[0] {row['argument']}\n[1] {compare_argument['argument']}"
        row_label = p.findall(row[0])[0]
        compare_label = p.findall(compare_argument[0])[0]
        id = str(uuid.uuid4())

        if QUALITY_MAPPING[row_label] == QUALITY_MAPPING[compare_label]:
            rank_mapping = RANK_MAPPING[2]
        elif QUALITY_MAPPING[row_label] > QUALITY_MAPPING[compare_label]:
            rank_mapping = RANK_MAPPING[0]
        else:
            rank_mapping = RANK_MAPPING[1]

        output.append_positive_example(prompt, rank_mapping, "")

        wrong_stance = random.choice([s for s in RANK_MAPPING if s != rank_mapping])
        output.append_negative_example(prompt, wrong_stance, "")

        output.append_instance(id, prompt, [rank_mapping])

    metadata.add_dataset(dataset_name)
    output.write_output(dataset_name)

if __name__ == "__main__":
    arg_parser = ArgumentParser(description="Program to convert gretz20 ibm quality dataset into appropriate form")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)

    metadata = Metadata(DATASET_NAME)

    dataset_path = str(datasets_path()
                       / "argument-quality"
                       / "wachsmuth17-computational-argumentation-quality-assessment-in-natural-language"
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

    make_output(dataset, metadata, "overall quality", overall_quality_description, "wachsmuth17_dagstuhl_overall_quality_pairvise_rank.json")

    make_output(dataset, metadata, "effectiveness", effectiveness_description, "wachsmuth17_dagstuhl_effectiveness_pairvise_rank.json")

    make_output(dataset, metadata, "local acceptability", local_acceptability_description, "wachsmuth17_dagstuhl_local_acceptability_pairvise_rank.json")

    make_output(dataset, metadata, "appropriateness", appropriateness_description, "wachsmuth17_dagstuhl_appropriateness_pairvise_rank.json")

    make_output(dataset, metadata, "arrangement", arrangement_description, "wachsmuth17_dagstuhl_arrangement_pairvise_rank.json")

    make_output(dataset, metadata, "clarity", clarity_description, "wachsmuth17_dagstuhl_clarity_pairvise_rank.json")

    make_output(dataset, metadata, "cogency", cogency_description, "wachsmuth17_dagstuhl_cogency_pairvise_rank.json")

    make_output(dataset, metadata, "global acceptability", global_acceptability_description, "wachsmuth17_dagstuhl_global_acceptability_pairvise_rank.json")

    make_output(dataset, metadata, "global relevance", global_relevance_description, "wachsmuth17_dagstuhl_global_relevance_pairvise_rank.json")

    make_output(dataset, metadata, "global sufficiency", global_sufficiency_description, "wachsmuth17_dagstuhl_global_sufficiency_pairvise_rank.json")

    make_output(dataset, metadata, "reasonableness", reasonableness_description, "wachsmuth17_dagstuhl_reasonableness_pairvise_rank.json")

    make_output(dataset, metadata, "local relevance", local_relevance_description, "wachsmuth17_dagstuhl_local_relevance_pairvise_rank.json")

    make_output(dataset, metadata, "credibility", credibility_description, "wachsmuth17_dagstuhl_credibility_pairvise_rank.json")

    make_output(dataset, metadata, "emotional appeal", emotional_appeal_description, "wachsmuth17_dagstuhl_emotional_appeal_pairvise_rank.json")

    make_output(dataset, metadata, "sufficiency", local_sufficiency_description, "wachsmuth17_dagstuhl_sufficiency_pairvise_rank.json")

    metadata.add_evaluation_metric("f1_macro")
    metadata.write_metadata()
