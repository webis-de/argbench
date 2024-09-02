from dataclasses import dataclass, field
from common import Output, datasets_path, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser
import uuid

DATASET_NAME = "argument_fallacy_classification_elecdeb60t020_goffredo23"

@dataclass
class FallacySnippet:
    tokens: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    label_spans: list = field(default_factory=list)


def process_dataset(data_file, output_file, metadata, split_name):
    """
    Process elecdeb60t020 datafile
    """
    output = Output(DATASET_NAME)
    output.append_definition("Given an argument snippet, extract substring that contains fallacy. Extracted fallacies should be separated by newlines and follow following format: [Fallacy Name]: [Extracted Substring].\n" +
                             "Available Fallacy names: AdHominem: When the argument becomes an excessive attack on an arguer’s position\n" +
                             "AppealtoEmotion: The unessential loading of the argument with emotional language to exploit the audience emotional instinct.\n" +
                             "AppealtoAuthority: It occurs when the arguer relies on the endorsement of an authority figure or a group consensus without providing sufficient evidence. It may also involve the citation of non-experts or the majority to support their claim.\n" +
                             "Slipperyslope: This fallacy implies that an improbable or exaggerated consequence could result from a particular action.\n" +
                             "FalseCause: The misinterpretation of the correlation of two events for causation.\n" +
                             "Slogans: It is a brief and striking phrase used to provoke excitement of the audience, and is often accompanied by another type of fallacy called argument by repetition.")

    token_file = open(data_file, "r")

    snippets = []

    temp_snippet = FallacySnippet()
    temp_label_span = []

    for line in token_file:
        if line == "\n":
            if temp_label_span:
                temp_label_span.append(snippet_idx + 1)
                temp_snippet.label_spans.append(temp_label_span)
                temp_label_span = []
            snippets.append(temp_snippet)
            temp_snippet = FallacySnippet()
            continue

        fields = line.split("\t")

        snippet_idx = int(fields[0])
        token = fields[1]
        label = fields[4].strip()

        if label[0] == "B":
            temp_label_span.append(snippet_idx)
        elif label == "O" and temp_label_span:
            temp_label_span.append(snippet_idx)
            temp_snippet.label_spans.append(temp_label_span)
            temp_label_span = []

        temp_snippet.tokens.append(token)
        temp_snippet.labels.append(label)


    for snippet in snippets:
        id = str(uuid.uuid4())
        prompt = " ".join(snippet.tokens)

        model_out = []
        # print(snippet)
        for label_span in snippet.label_spans:
            span_class = snippet.labels[label_span[0]][2:]
            span_tokens = " ".join(snippet.tokens[label_span[0]:label_span[1]])
            model_out.append(f"{span_class}: {span_tokens}")

        model_out = "\n".join(model_out)

        output.append_instance(id, prompt, [model_out])

    output.write_output(output_file)
    metadata.add_evaluation_metric("f1_macro")
    metadata.add_dataset(output_file, split_name)

if __name__ == "__main__":
    arg_parser = ArgumentParser(description="What dataset will be processed?")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args) # Seed random number generation

    data_path = datasets_path() / "elecdeb60t020" # path to data


    metadata = Metadata(DATASET_NAME)

    process_dataset(
        data_path / "train.conll",
        "argument_fallacy_classification_elecdeb60t020_train_goffredo23.json",
        metadata,
        "train"
    )

    process_dataset(
        data_path / "test.conll",
        "argument_fallacy_classification_elecdeb60t020_test_goffredo23.json",
        metadata,
        "test"
    )

    process_dataset(
        data_path / "dev.conll",
        "argument_fallacy_classification_elecdeb60t020_dev_goffredo23.json",
        metadata,
        "dev"
    )

    metadata.write_metadata()
