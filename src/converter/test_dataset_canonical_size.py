from .common import tasks_path
import pytest
import json

DATASET_CANON_SIZE = [
    {
        "name": "argument_unit_relation_identification_essays_stab17", # essays
        "canon_size": 402
    },
    {
        "name": "premise_generation_microtexts_v1_skeppstedt18", # microtexts1
        "canon_size": 112
    },
    {
        "name": "premise_generation_microtexts_v2_skeppstedt18", # microtexts2
        "canon_size": 112
    },
    {
        "name": "argument_unit_classification_erulemaking_park18", # erulemaking 781 x cartesian products of argument units
        "canon_size": 45782
    },
    {
        "name": "argument_relation_identification_political_debates_menini18", # political argumentation, unbalanced. Balanced - 1462
        "canon_size": 1907
    },
    {
        "name": "claim_optimization_claim_revisions_skitalinskaya23", # claim revisions 121504 currently in dataset
        "canon_size": 124312 # In the paper original 210,222 revision history pairs were reduced to 198,089 instances of main labels, which may explain why number is lover
        # In readme there exists original unrevised dataset, we use revised
    },
    {
        "name": "evidence_extraction_ibm_claim_evidence_aharoni14",
        "canon_size": 1291
    },
    {
        "name": "conclusion_extraction_ibm_claim_evidence_aharoni14", # 1392 CDCs, 1387 in CDC file, 1291 related to CDEs
        "canon_size": 1392
    },
    {
        "name": "argument_unit_segmentation_webis_editorials_alkhatib16",
        "canon_size": 300
    },
    {
        "name": "warrant_identification_semeval_2018_task_12_habernal18", # 1970 high-quality instances
        "canon_size": 1970
    },
    {
        "name": "fallacy_detection_cmv_adhominem_habernal18", # 1800 annotated OPs
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
