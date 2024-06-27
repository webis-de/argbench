from .common import tasks_path
import pytest
import json

DATASET_CANON_SIZE = [
    {
        "name": "stab17_argument_relation_identification", # essays
        "canon_size": 402
    },
    {
        "name": "peldzus15_argument_relation_identification", # microtexts1
        "canon_size": 112
    },
    {
        "name": "skepstedt18_argument_relation_identification", # microtexts2
        "canon_size": 112
    },
    {
        "name": "park18_cornell_erulemaking_classification", # erulemaking 781 x cartesian products of argument units
        "canon_size": 45782
    },
    {
        "name": "menini18_relation_identification", # political argumentation, unbalanced. Balanced - 1462
        "canon_size": 1907
    },
    {
        "name": "skitalinskaya23_claim_optimization", # claim revisions 121504 currently in dataset
        "canon_size": 124312 # In the paper original 210,222 revision history pairs were reduced to 198,089 instances of main labels, which may explain why number is lover
        # In readme there exists original unrevised dataset, we use revised
    },
    {
        "name": "aharoni14_claim_evidence_extraction",
        "canon_size": 1291
    },
    {
        "name": "aharoni14_claim_extraction", # 1392 CDCs, 1387 in CDC file, 1291 related to CDEs
        "canon_size": 1392
    },
    {
        "name": "alkhatib16_argumentation_strategy_mining_unit_segmentation",
        "canon_size": 300
    },
    {
        "name": "habernal18_implicit_warrant_identification", # 1970 high-quality instances
        "canon_size": 1970
    },
    {
        "name": "habernal18-ad-hominem-detection", # 1800 annotated OPs
        "canon_size": 1800
    }
]

@pytest.mark.parametrize("dataset_size", DATASET_CANON_SIZE)
def test_dataset_canonical_size(dataset_size):
    with open(tasks_path() / "metadata.json", "r") as f:
        metadata = json.load(f)

    total_instances = 0
    for file in metadata[dataset_size["name"]]["file_list"]:
        file_path = tasks_path() / dataset_size["name"] / file
        with open(file_path, "r") as f:
            dataset = json.load(f)
            total_instances += len(dataset["Instances"])

    assert total_instances == dataset_size["canon_size"], "Instance amount is not equal to canon instance amount"
