from common import Output, datasets_path, read_tabular, tasks_path, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser
import numpy as np
import uuid

DATASET_NAME = "argument_ranking_dagstuhl_15512_full_rank_wachsmuth17"
QUALITY_MAPPING = {
    "Low": 2,
    "Average": 1,
    "High": 0
}

def make_output(dataset, metadata, column, aspect_description, dataset_name):
    output = Output(DATASET_NAME)

    output.append_definition(f"Rank the following arguments on the given topic according to quality aspect: {column}. Quality aspect description: {aspect_description} Both arguments should be included and listed using identifiers, in descending order of relevance. The output format should be: [0] > [1], [1] > [0] or [0] = [1] if both arguments are equally good. Only respond with the ranking results, do not say any word or explain.")

    arguments = dataset.groupby(["argument"]).apply(lambda row: row[column].value_counts().index[0]).reset_index()
    arguments[0] = arguments[0].str.extract(r"\(([a-zA-Z]+)\)")[0]

    for row in arguments.iterrows():
        row = row[1]
        other_labels = [q for q in QUALITY_MAPPING if q != row[0]]
        compare_argument_1 = arguments[arguments[0] == other_labels[0]].sample(1).iloc[0]
        compare_argument_2 = arguments[arguments[0] == other_labels[1]].sample(1).iloc[0]
        prompt = f"Arguments:\n[0] {row['argument']}\n[1] {compare_argument_1['argument']}\n[2] {compare_argument_2['argument']}"
        id = str(uuid.uuid4())

        ranks = np.argsort([
            QUALITY_MAPPING[label]
            for label in [row[0], compare_argument_1[0], compare_argument_2[0]]
        ])
        rank_mapping = " > ".join(["[{}]".format(r) for r in ranks])
        wrong_mapping = " > ".join(["[{}]".format(r) for r in np.flip(ranks)])

        output.append_positive_example(prompt, rank_mapping, "")

        output.append_negative_example(prompt, wrong_mapping, "")

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

    make_output(dataset, metadata, "overall quality", overall_quality_description, "argument_ranking_dagstuhl_15512_full_rank_overall_quality_wachsmuth17.json")

    make_output(dataset, metadata, "effectiveness", effectiveness_description, "argument_ranking_dagstuhl_15512_full_rank_effectiveness_wachsmuth17.json")

    make_output(dataset, metadata, "local acceptability", local_acceptability_description, "argument_ranking_dagstuhl_15512_full_rank_local_acceptability_wachsmuth17.json")

    make_output(dataset, metadata, "appropriateness", appropriateness_description, "argument_ranking_dagstuhl_15512_full_rank_appropriateness_wachsmuth17.json")

    make_output(dataset, metadata, "arrangement", arrangement_description, "argument_ranking_dagstuhl_15512_full_rank_arrangement_wachsmuth17.json")

    make_output(dataset, metadata, "clarity", clarity_description, "argument_ranking_dagstuhl_15512_full_rank_clarity_wachsmuth17.json")

    make_output(dataset, metadata, "cogency", cogency_description, "argument_ranking_dagstuhl_15512_full_rank_cogency_wachsmuth17.json")

    make_output(dataset, metadata, "global acceptability", global_acceptability_description, "argument_ranking_dagstuhl_15512_full_rank_global_acceptability_wachsmuth17.json")

    make_output(dataset, metadata, "global relevance", global_relevance_description, "argument_ranking_dagstuhl_15512_full_rank_global_relevance_wachsmuth17.json")

    make_output(dataset, metadata, "global sufficiency", global_sufficiency_description, "argument_ranking_dagstuhl_15512_full_rank_global_sufficiency_wachsmuth17.json")

    make_output(dataset, metadata, "reasonableness", reasonableness_description, "argument_ranking_dagstuhl_15512_full_rank_reasonableness_wachsmuth17.json")

    make_output(dataset, metadata, "local relevance", local_relevance_description, "argument_ranking_dagstuhl_15512_full_rank_local_relevance_wachsmuth17.json")

    make_output(dataset, metadata, "credibility", credibility_description, "argument_ranking_dagstuhl_15512_full_rank_credibility_wachsmuth17.json")

    make_output(dataset, metadata, "emotional appeal", emotional_appeal_description, "argument_ranking_dagstuhl_15512_full_rank_emotional_appeal_wachsmuth17.json")

    make_output(dataset, metadata, "sufficiency", local_sufficiency_description, "argument_ranking_dagstuhl_15512_full_rank_sufficiency_wachsmuth17.json")

    metadata.add_evaluation_metric("f1_macro")
    metadata.write_metadata()
