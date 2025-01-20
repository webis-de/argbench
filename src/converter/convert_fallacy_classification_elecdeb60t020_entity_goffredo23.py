from dataclasses import dataclass, field
from common import Genres, Output, Subareas, datasets_path, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser
import uuid

DATASET_NAME = "fallacy_extraction_elecdeb60t020"

map = {
    "AdHominem": "Ad Hominem",
    "AppealtoEmotion" : "Appeal to Emotion",
    "AppealtoAuthority" : "Appeal to Authority",
    "Slipperyslope" : "Slippery Slope",
    "FalseCause" : "False Cause",
    "Slogans" : "Slogans",
}
@dataclass
class FallacyDoc:
    tokens: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    label_spans: list = field(default_factory=list)


def process_dataset(data_file, output_file, metadata, split_name):
    """
    Process elecdeb60t020 datafile
    """
    output = Output(DATASET_NAME)
    output.append_definition("Given the following document, extract spans of text the cover a fallacy of the following fallacies."
                             "Prepend the span with the fallacy name." +
                             "Possible Fallacies are: "
                             "Ad Hominem: When the span becomes an excessive attack on an arguer’s position\n" +
                             "Appeal to Emotion: The span is loaded with emotional language to exploit the audience emotional instinct.\n" +
                             "Appeal to Authority: the span occurs when the arguer relies on the endorsement of an authority figure or a group consensus without providing sufficient evidence. It may also involve the citation of non-experts or the majority to support their claim.\n" +
                             "Slippery slope: This span implies that an improbable or exaggerated consequence could result from a particular action.\n" +
                             "False Cause: The span is a  misinterpretation of the correlation of two events for causation.\n" +
                             "Slogans: the span is a brief and striking phrase used to provoke excitement of the audience, and is often accompanied by another type of fallacy called argument by repetition.")

    token_file = open(data_file, "r")

    docs = []

    doc = FallacyDoc()
    temp_label_span = []

    for line in token_file:
        if line == "\n":
            if temp_label_span:
                temp_label_span.append(id + 1)
                doc.label_spans.append(temp_label_span)
                temp_label_span = []
            docs.append(doc)
            doc = FallacyDoc()
            continue

        fields = line.split("\t")

        id = int(fields[0])
        token = fields[1]
        label = fields[4].strip()

        if label[0] == "B":
            temp_label_span.append(id)
        elif label == "O" and temp_label_span:
            temp_label_span.append(id)
            doc.label_spans.append(temp_label_span)
            temp_label_span = []

        doc.tokens.append(token)
        doc.labels.append(label)


    for doc in docs:
        id = str(uuid.uuid4())
        prompt = " ".join(doc.tokens)
        prompt = f"Document: {prompt}"

        model_out = []
        spans_start=False
        for t, l in zip(doc.tokens, doc.labels):

            if l.startswith("B"):
                spans_start=True
                model_out.append(map[l[2:]]+":")
                model_out.append(t)
            elif l.startswith("I"):
                model_out.append(t)
            else:
                if spans_start:
                    model_out.append("\n")
                    spans_start = False


        model_out = " ".join(model_out)

        output.append_instance(id, prompt, [model_out])

    output.append_genre(Genres.DEBATE_PORTALS)
    output.append_subarea(Subareas.MINING)
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
        "argument_fallacy_classification_elecdeb60t020_entity_train_goffredo23.json",
        metadata,
        "train"
    )

    process_dataset(
        data_path / "test.conll",
        "argument_fallacy_classification_elecdeb60t020_entity_test_goffredo23.json",
        metadata,
        "test"
    )

    process_dataset(
        data_path / "dev.conll",
        "argument_fallacy_classification_elecdeb60t020_entity_dev_goffredo23.json",
        metadata,
        "dev"
    )

    metadata.add_genre(Genres.DEBATE_PORTALS)
    metadata.add_subarea(Subareas.MINING)
    metadata.write_metadata()
