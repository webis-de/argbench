from common import Output, read_tabular, Metadata, add_seed_arg, set_seed, datasets_path, Genres, Skills
from argparse import ArgumentParser
import uuid


dataset_name = "claim_improvement_suggestion_claim_revisions_skitalinskaya23"
dataset_template = "claim_improvement_suggestion_claim_revisions_{split}_skitalinskaya23.json"


# def process_data(dataset, metadata, split):
#     output = Output(dataset_name)
#     output.append_definition("Judge if claim can be improved by revising it. Possible outputs: Improvable if revision should be made, Non-Improvable if no revision is necessary.")
#
#     original_claims = dataset.loc[dataset.groupby('claim_id')['revision_id'].idxmin()]
#     final_claims = dataset.loc[dataset.groupby('claim_id')['revision_id'].idxmax()]
#
#     for row in original_claims.iterrows():
#         row = row[1]
#         prompt = f"Claim: {row['claim_text']}"
#         id = str(uuid.uuid4())
#         output.append_instance(id, prompt, ["Improvable"])
#
#     for row in final_claims.iterrows():
#         row = row[1]
#         prompt = f"Claim: {row['claim_text']}"
#         id = str(uuid.uuid4())
#         output.append_instance(id, prompt, ["Non-Improvable"])
#
#     metadata.add_dataset(dataset_file)
#     output.append_genre(Genres.DEBATES)
#     output.append_subarea(Subareas.MINING)
#     output.write_output(dataset_file)

def process_data(dataset, metadata, split):
    quality_issues = {"Links", "Typo or grammar", "Clarification", "Other"}
    labels = {"Links", "Typo or grammar", "Clarification"}
    dataset = dataset[dataset["revision_type"]!="CLEAN"]
    dataset["revision_type"] = dataset["revision_type"].apply(lambda x: "Clarified claim" if x == "Clarified argument" else x)
    dataset["revision_type"] = dataset["revision_type"].apply(lambda x: "Clarification" if x == "Clarified claim" else x)
    dataset["revision_type"] = dataset["revision_type"].apply(lambda x: "Clarification" if x == "Clarified claim" else x)
    dataset["revision_type"] = dataset["revision_type"].apply(lambda x: "Typo or grammar" if x == "Typo or grammar correction" else x)
    dataset["revision_type"] = dataset["revision_type"].apply(lambda x: "Links" if x == "Corrected or added links" else x)
    dataset["revision_type"] = dataset["revision_type"].apply(lambda x: "Other" if x not in labels else x)
    dataset_file = dataset_template.format(split=split)
    output = Output(dataset_name)
    output.append_definition("""Given an argumentative claim, does the following quality issue match the following claim.
     Available quality issues are Clarification, Typo/Grammar, Links, or Other. If the quality issue matches the claim, output Match.
     If the quality issue does not apply to the claim, output No-match. Only output Match or No-match.""")
    all_counts = {label:0 for label in quality_issues}
    for row in dataset.iterrows():
        row = row[1]
        for quality_issue in quality_issues:
            prompt = f"Quality Issue: {quality_issue}\nClaim: {row['claim_text']}"
            id = quality_issue + "-" +str(uuid.uuid4())
            if row["revision_type"] == quality_issue:
                output.append_instance(id, prompt, ["Match"])
                all_counts[quality_issue] +=  1
            else:
                output.append_instance(id, prompt, ["No-match"])
    print(all_counts)
    metadata.add_dataset(dataset_file, split)
    output.append_genre(Genres.DEBATE_PORTALS)
    output.append_subarea(Skills.QUALITY_ASSESSMENT)
    output.write_output(dataset_file)

if __name__ == "__main__":
    arg_parser = ArgumentParser(description="What dataset will be processed?")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    

    data_path = datasets_path() / "claim-revisions" / "acl23_revised.csv"
    metadata = Metadata(dataset_name)
    dataset = read_tabular(data_path)
    #dataset["data_split"] = dataset["data_split"].map(lambda x: "train" if x == "train" or x =="dev" else "test")
    train_dataset = dataset[dataset["data_split"] == "train"]
    test_dataset = dataset[dataset["data_split"] == "test"]
    val_dataset = dataset[dataset["data_split"] == "dev"]
    print(f"train {len(train_dataset)}")
    print(f"test {len(test_dataset)}")
    print(f"dev {len(val_dataset)}")
    process_data(train_dataset, metadata, "train")
    process_data(test_dataset, metadata, "test")
    process_data(val_dataset, metadata, "val")

    metadata.add_genre(Genres.DEBATE_PORTALS)
    metadata.add_skill(Skills.QUALITY_ASSESSMENT)
    metadata.add_evaluation_metric("fscore")
    metadata.write_metadata()
