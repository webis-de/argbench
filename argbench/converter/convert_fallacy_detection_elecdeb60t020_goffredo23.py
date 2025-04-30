from dataclasses import dataclass, field
from common import Output, datasets_path, Metadata, add_seed_arg, set_seed, Genres, Skills
from argparse import ArgumentParser
import uuid

DATASET_NAME = "fallacy_detection_elecdeb60t020_goffredo23"

@dataclass
class FallacySnippet:
    document: str = None
    spans: list = None

map ={
    "AdHominem" : "Ad Hominem",
    "AppealtoEmotion": "Appeal to Emotion",
    "AppealtoAuthority": "Appeal to Authority",
    "Slipperyslope": "Slippery Slope",
    "FalseCause": "False Cause",
    "Slogans": "Slogans"
}
def process_dataset(data_files, output_file, metadata, split_name):
    """
    Process elecdeb60t020 datafile
    """
    output = Output(DATASET_NAME)
    output.append_definition("Given the following argument, split the argument into spans that contains one of the following fallacy. In case a span does not contain a fallacy, simply prepend it with No-fallacy"+
                             "The split spans should be separated by newlines and be output in the exact order they appear in the argument. Add before each span that covers a fallacy the name of the fallacy and a colon.\n" +
                             "Do not explain and do not rephrase anything in the argument." +
                             "Here are the candidate fallacies: Ad Hominem: When the argument becomes an excessive attack on an arguer’s position\n" +
                             "Appeal to Emotion: The unessential loading of the argument with emotional language to exploit the audience emotional instinct.\n" +
                             "Appeal to Authority: It occurs when the arguer relies on the endorsement of an authority figure or a group consensus without providing sufficient evidence. It may also involve the citation of non-experts or the majority to support their claim.\n" +
                             "Slippery Slope: This fallacy implies that an improbable or exaggerated consequence could result from a particular action.\n" +
                             "False Cause: The misinterpretation of the correlation of two events for causation.\n" +
                             "Slogans: It is a brief and striking phrase used to provoke excitement of the audience, and is often accompanied by another type of fallacy called argument by repetition.\n"
                             )

    for data_file in data_files:
        token_file = open(data_file, "r")

        snippets = []

        temp_snippet = FallacySnippet()
        doc = ""
        temp_snippet.document = doc
        temp_snippet.spans = []
        fallacy_span = False
        current_span = ""
        current_label = ""
        for line in token_file:
            if line == "\n":
                if current_span:
                    temp_snippet.spans.append((current_span, current_label))

                snippets.append(temp_snippet)
                temp_snippet = FallacySnippet()
                doc = ""
                fallacy_span = False
                temp_snippet.document = doc
                temp_snippet.spans = []
                current_span = ""
                current_label = ""

                continue

            fields = line.split("\t")

            snippet_idx = int(fields[0])
            token = fields[1]
            label = fields[4].strip()
            if not temp_snippet.document:
                temp_snippet.document = token
            else:
                temp_snippet.document = temp_snippet.document + " " + token
            if label[0] == "B":
                if current_span and fallacy_span:
                    temp_snippet.spans.append((current_span, current_label))
                    current_span = token
                    current_label = map[label[2:]]
                    fallacy_span = True
                elif current_span and not fallacy_span:
                    temp_snippet.spans.append((current_span, current_label))
                    current_span = token
                    current_label = map[label[2:]]
                    fallacy_span = True
                elif not current_span and not fallacy_span:
                    current_span = token
                    current_label = map[label[2:]]
                    fallacy_span = True
            elif label == "O":
                if fallacy_span:
                    temp_snippet.spans.append((current_span, current_label))
                    current_span = token
                    current_label = "No-fallacy"
                    fallacy_span = False
                else:
                    if current_span:
                        current_span = current_span + " " + token
                    else:
                        current_span = token
                        current_label = "No-fallacy"
            elif label[0] == "I":
                current_span = current_span + " " + token

        for snippet in snippets:
            id = str(uuid.uuid4())
            prompt = snippet.document
            model_out = []

            # print(snippet)
            for label_span in snippet.spans:
                model_out.append(f"{label_span[1]}: {label_span[0]}")

            model_out = "\n".join(model_out)

            output.append_instance(id, prompt, [model_out])

    output.append_genre(Genres.DEBATES)
    output.append_subarea(Skills.REASONING)
    output.write_output(output_file)
    
    metadata.add_dataset(output_file, split_name)
    metadata.add_evaluation_metric("fallacy-fscore")

if __name__ == "__main__":
    arg_parser = ArgumentParser(description="What dataset will be processed?")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args) # Seed random number generation

    data_path = datasets_path() / "elecdeb60t020" # path to data


    metadata = Metadata(DATASET_NAME)

    process_dataset(
        [data_path / "dev.conll"],
        "fallacy_detection_elecdeb60t020_val_goffredo23.json",
        metadata,
        "val"
    )

    process_dataset(
        [data_path / "train.conll"],
        "fallacy_detection_elecdeb60t020_train_goffredo23.json",
        metadata,
        "train"
    )

    process_dataset(
        [data_path / "test.conll"],
        "fallacy_detection_elecdeb60t020_test_goffredo23.json",
        metadata,
        "test"
    )

    metadata.add_genre(Genres.DEBATES)
    metadata.add_skill(Skills.REASONING)
    metadata.write_metadata()
