from typing import List
from uuid import uuid4
from common import Output, datasets_path, Metadata, add_seed_arg, set_seed, Genres, Subareas
from argparse import ArgumentParser
from pathlib import Path
from dataclasses import dataclass, field

DATASET_NAME = "argument_unit_segmentation_webis_editorials_entity_alkhatib16"
DATASET_PATH = "argument_unit_segmentation_webis_editorials_entity_alkhatib16.json"

SEGMENT_LABEL_MAPPING = {
    "statistics": "Statistics",
    "common-ground": "Common Ground",
    "assumption": "Assumption",
    "testimony": "Testimony",
    "anecdote": "Anecdote"
}

@dataclass
class DiscourseUnit:
    label: str
    tokens: List[str] = field(default_factory=list)


def extract_file(file):

    editorial = []
    with open(file, "r") as f:
        for line in f:
            line = line.strip()
            parsed_line = line.split("\t")
            if len(parsed_line) > 3:
                parsed_line = [parsed_line[0], parsed_line[1], "\t".join(parsed_line[2:])]
            if parsed_line[1] in ["title", "par-sep"]:
                continue

            if not len(editorial):
                editorial.append(DiscourseUnit(parsed_line[1], [parsed_line[2]]))
            elif (parsed_line[1] == "continued" and editorial[-1].label == "continued"):
                editorial[-1].tokens.append(parsed_line[2])
            elif parsed_line[1] == "no-unit" and (editorial[-1].label == "no-unit" or editorial[-1].label == "continued"):
                editorial[-1].label = parsed_line[1]
                editorial[-1].tokens.append(parsed_line[2])
            elif (parsed_line[1] == "continued" and editorial[-1].label != "continued") or (parsed_line[1] == "no-unit" and editorial[-1].label != "no-unit"):
                editorial.append(DiscourseUnit(parsed_line[1], [parsed_line[2]]))
            elif parsed_line[1] != "continued" and editorial[-1].label == "continued":
                editorial[-1].label = parsed_line[1]
                editorial[-1].tokens.append(parsed_line[2])
            elif parsed_line[1] != "continued" and parsed_line[1] != "no-unit":
                editorial.append(DiscourseUnit(parsed_line[1], [parsed_line[2]]))

    return editorial



def process_folder(path: Path):

    editorials = []
    for f in path.iterdir():
        editorial = extract_file(f)
        editorials.append(editorial)

    return editorials


if __name__ == "__main__":

    arg_parser = ArgumentParser(description="Program to convert ajjour unit segmentation dataset into appropriate form")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args)

    dataset_path = (datasets_path()
                    / "editorials"
                    / "txt"
                    / "txt"
                    / "complete-annotated-final")

    metadata = Metadata(DATASET_NAME)
    output = Output(DATASET_NAME)
    output.append_definition("Given document replace all relevant argument words with word mask. " +
                             "Available word masks: \n" +
                             "Common Ground -  The unit states common knowledge, a self-evident fact, an accepted truth, or similar.\n" +
                             "Assumption - The unit states an assumption, conclusion, judgment, or opinion of the author, a general observation, possibly false fact, or similar.\n" +
                             "Testimony -  The unit gives evidence by stating or quoting that a proposition was made by some expert, authority, witness, group, organization, or similar.\n" +
                             "Statistics - The unit gives evidence by stating or quoting the results or conclusions of quantitative research, studies, empirical data analyses, or similar. A reference may but needs not always be given.\n" +
                             "Anecdote - The unit gives evidence by stating personal experience of the author, an anecdote, a concrete example, an instance, a specific event, or similar.\n" +
                             "Output should be original document but all words related to above mentioned argumentation units must be replaced by word mask name.")

    editorials = process_folder(dataset_path)

    for editorial in editorials:
        prompt = ""
        extracted = ""
        for unit in editorial:
            combined_tokens = " ".join(unit.tokens)
            prompt += " " + combined_tokens
            if unit.label in ["par-sep", "no-unit", "other"]:
                extracted += " " + combined_tokens
            else:
                for _ in combined_tokens.split(" "):
                    extracted += " " + SEGMENT_LABEL_MAPPING[unit.label]

        prompt = prompt.strip()
        extracted = extracted.strip()
        id = str(uuid4())
        output.append_instance(id, prompt, [extracted])

    output.append_genre(Genres.ESSAYS)
    output.append_subarea(Subareas.MINING)
    output.write_output(DATASET_PATH)

    metadata.add_genre(Genres.ESSAYS)
    metadata.add_subarea(Subareas.MINING)
    metadata.add_dataset(DATASET_PATH)
    metadata.add_evaluation_metric("f1_macro")
    metadata.write_metadata()
