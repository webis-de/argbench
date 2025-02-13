import random

from common import Genres, Output, Subareas, datasets_path, read_tabular, Metadata, add_seed_arg, set_seed, \
    find_topic_size_to_split
from argparse import ArgumentParser
import uuid
import re
import math
dataset_name = "premise_extraction_ibm_claim_evidence_aharoni14"

template_file_study = "premise_extraction_study_ibm_claim_evidence_{split}_aharoni14.json"
template_file_anecdote = "premise_extraction_anecdote_ibm_claim_evidence_{split}_aharoni14.json"
template_file_expert = "premise_extraction_expert_ibm_claim_evidence_{split}_aharoni14.json"
space_remover = re.compile("\s+")



def preprocess_dataset(dataset, metadata, split):
    output_study = Output(dataset_name)
    output_study.append_definition("""Given the following Wikipedia section context, claim, detect study evidence on the given context that supports the claim
                              A claim is an assertion that an argument tries to prove. Study evidence is the results of a quantitative analysis of data, 
                              given as numbers, or as conclusions.""")
    output_expert = Output(dataset_name)
    output_expert.append_definition("""Given the following Wikipedia section context, claim, detect expert evidence on the given context that supports the claim
                              A claim is an assertion that an argument tries to prove. Expert evidence is a testimony by a person/group/committee/organization with some known expertise/authority on the topic.""")

    output_anecdotal = Output(dataset_name)
    output_anecdotal.append_definition("""Given the following Wikipedia section context, claim, detect expert evidence on the given context that supports the claim
                              A claim is an assertion that an argument tries to prove. Anecotal evidence is a description of an episode(s), centered on individual(s) or clearly located in place and/or in time.""")



    dataset["Article"] = dataset["Article"].str.replace(" ", "_")
    dataset["wiki_article"] = ""

    for article_path in articles_folder.iterdir():
        if not (dataset["Article"] == article_path.name).sum():
            continue
        with open(article_path, "r") as f:
            article_contents = f.readlines()
            article_contents = [c for c in article_contents if c != "\n"]
            article_match = [space_remover.sub("", c.lower()) for c in article_contents]

        article_data = dataset[dataset["Article"] == article_path.name]

        for row_idx, article_row in article_data.iterrows():
            evidence_match = space_remover.sub("", article_row["CDE"].lower())
            article_matched = []
            for i, am in enumerate(article_match):
                if (evidence_match in am or
                        am in evidence_match or
                        evidence_match[:5] in am or
                        evidence_match[-5:] in am):
                    article_matched.append(i)
            if len(article_matched) < 1:
                print(article_contents)
                print(article_row["CDE"])
                print(article_matched)
                raise Exception()

            article_window_start = article_matched[0] - args.back_add
            article_window_end = article_matched[-1] + args.front_add
            if article_window_end > (len(article_contents) - 1):
                article_window_end = len(article_contents) - 1
            if article_window_start < 0:
                article_window_start = 0

            collected_article = "".join(article_contents[article_window_start:article_window_end])
            dataset.loc[row_idx, "wiki_article"] = collected_article


    for article, article_rows in dataset.groupby("Article"):
        row = article_rows.iloc[0]
        id = str(uuid.uuid4())
        topic = row["Topic"]
        prompt = f"Topic: {topic}\nCLaim: {row['Claim']}\nArticle: {row['wiki_article']}"
        evidence_strings_study = ""
        evidence_strings_expert = ""
        evidence_string_anecdotes = ""
        for record_index, record in article_rows.iterrows():
            if record["Type 1"] == "STUDY" or record["Type 2"] == "STUDY":
                evidence = record["CDE"]
                evidence_strings_study = evidence_strings_study + f"Study: {evidence} \n"
            if record["Type 1"] == "EXPERT" or record["Type 2"] == "EXPERT":
                evidence = record["CDE"]
                evidence_strings_expert = evidence_strings_expert + f"Expert: {evidence} \n"
            if record["Type 1"] == "ANECDOTAL" or record["Type 2"] == "ANECDOTAL":
                evidence = record["CDE"]
                evidence_string_anecdotes = evidence_string_anecdotes + f"Anecdotal: {evidence} \n"

        output_expert.append_instance(id, prompt, [evidence_strings_expert])
        output_study.append_instance(id, prompt, [evidence_strings_study])
        output_anecdotal.append_instance(id, prompt, [evidence_string_anecdotes])

    dataset_file_study = template_file_study.replace("{split}",split)
    dataset_file_expert = template_file_expert.format(split=split)
    dataset_file_anecdote = template_file_anecdote.format(split=split)

    output_study.append_genre(Genres.WIKIPEDIA)
    output_study.append_subarea(Subareas.MINING)
    output_study.write_output(dataset_file_study)
    output_expert.append_genre(Genres.WIKIPEDIA)
    output_expert.append_subarea(Subareas.MINING)
    output_expert.write_output(dataset_file_expert)
    output_anecdotal.append_genre(Genres.WIKIPEDIA)
    output_anecdotal.append_subarea(Subareas.MINING)
    output_anecdotal.write_output(dataset_file_anecdote)
    metadata.add_dataset(dataset_file_study, split)
    metadata.add_dataset(dataset_file_anecdote, split)
    metadata.add_dataset(dataset_file_expert, split)

if __name__ == "__main__":
    arg_parser = ArgumentParser(description="What dataset will be processed?")
    add_seed_arg(arg_parser)
    arg_parser.add_argument("-f", "--front_add", default=1, type=int, help="Add # of articles in front of found evidence paragraph")
    arg_parser.add_argument("-b", "--back_add", default=1, type=int, help="Add # of articles in back of found evidence paragraph")
    args = arg_parser.parse_known_args()[0]
    set_seed(args)

    data_path = datasets_path() / "ibm-claim-evidence" / "2014_7_18_ibm_CDEdata.csv"
    articles_folder = datasets_path() / "ibm-claim-evidence" / "wiki12_articles"

    metadata = Metadata(dataset_name)
    dataset = read_tabular(data_path)
    df_test, df_train = find_topic_size_to_split(dataset, "Topic")

    print(len(df_train))
    print(len(df_test))

    metadata.add_genre(Genres.WIKIPEDIA)
    metadata.add_skill(Subareas.MINING)
    preprocess_dataset(df_train, metadata, "train")
    preprocess_dataset(df_test, metadata, "test")

    
    metadata.write_metadata()
