from dataclasses import dataclass, field
from common import Genres, Output, Subareas, datasets_path, Metadata, add_seed_arg, set_seed
from argparse import ArgumentParser
import uuid

DATASET_NAME = "fallacy_extraction_elecdeb60t020_goffredo23"

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
    output.append_definition("Given the following document, extract a span of text the cover one fallacy of the following fallacies. Extract only one span of text. "
                             "Prepend the span with the fallacy name and a colon. For example: Appeal to Emotion: he is a sick guy he can not be blamed.\n" +
                             "Possible Fallacies are: "
                             "Ad Hominem: When the span becomes an excessive attack on an arguer’s position\n" +
                             "Appeal to Emotion: The span is loaded with emotional language to exploit the audience emotional instinct.\n" +
                             "Appeal to Authority: the span occurs when the arguer relies on the endorsement of an authority figure or a group consensus without providing sufficient evidence. It may also involve the citation of non-experts or the majority to support their claim.\n" +
                             "Slippery slope: This span implies that an improbable or exaggerated consequence could result from a particular action.\n" +
                             "False Cause: The span is a misinterpretation of the correlation of two events for causation.\n" +
                             "Slogans: the span is a brief and striking phrase used to provoke excitement of the audience, and is often accompanied by another type of fallacy called argument by repetition.")

    token_file = open(data_file, "r")

    docs = []



    for line in token_file:
        if line == "\n":
            continue
        fields = line.split("\t")
        id = int(fields[0])
        token = fields[1]
        label = fields[4].strip()
        if id == 0:
            doc = FallacyDoc()
            docs.append(doc)
        doc.tokens.append(token)
        doc.labels.append(label)


    for doc in docs:
        id = str(uuid.uuid4())
        prompt = " ".join(doc.tokens)
        prompt = f"Document: {prompt}"

        fallacies = []
        fallacy = ""
        spans_start=False
        for t, l in zip(doc.tokens, doc.labels):

            if l.startswith("B"):
                spans_start=True
                fallacy+= map[l[2:]]+":"
                fallacy+= " " + t
            elif l.startswith("I"):
                fallacy+= " " + t
            else:
                if spans_start:
                    spans_start = False
                    fallacies.append(fallacy)
                    fallacy = ""



        output.append_instance(id, prompt, fallacies)

    output.append_genre(Genres.DEBATE_PORTALS)
    output.append_subarea(Subareas.MINING)
    output.write_output(output_file)
    metadata.add_evaluation_metric("f1_macro")
    metadata.add_dataset(output_file, split_name)
    return len(docs)
if __name__ == "__main__":
    arg_parser = ArgumentParser(description="What dataset will be processed?")
    add_seed_arg(arg_parser)
    args = arg_parser.parse_known_args()[0]
    set_seed(args) # Seed random number generation

    data_path = datasets_path() / "elecdeb60t020" # path to data


    metadata = Metadata(DATASET_NAME)

    train_count = process_dataset(
        data_path / "train.conll",
        "fallacy_extraction_elecdeb60t020_train_goffredo23.json",
        metadata,
        "train"
    )

    test_count = process_dataset(
        data_path / "test.conll",
        "fallacy_extraction_elecdeb60t020_test_goffredo23.json",
        metadata,
        "test"
    )

    dev_count = process_dataset(
        data_path / "dev.conll",
        "fallacy_extraction_elecdeb60t020_dev_goffredo23.json",
        metadata,
        "dev"
    )
    print (f"""found {dev_count + train_count + test_count} docs
                    {dev_count } dev docs\n
                    {train_count} train doc\n
                    {test_count} test doc\n
           f""")

    metadata.add_genre(Genres.DEBATE_PORTALS)
    metadata.add_subarea(Subareas.MINING)
    metadata.write_metadata()
