from common import Output, read_tabular, datasets_path, tasks_path
import json

def print_data_size(dataset_name, *data_paths):
    total_instances = 0

    for path in data_paths:
        with open(path, "r") as f:
            dataset = json.load(f)
            total_instances += len(dataset["Instances"])

    print(f"| {dataset_name} | {total_instances} |")


if __name__ == "__main__":
    ajjour_frame_identification = tasks_path() / "ajjour_frame_identification" / "ajjour_frame_identification.json"
    print_data_size("Unit Segmentation of Argumentative texts - Frame Identification", ajjour_frame_identification)

    ajjour_frame_identification = tasks_path() / "ajjour_frame_identification_stance" / "ajjour_frame_identification_stance.json"
    print_data_size("Unit Segmentation of Argumentative texts - Stance Classification", ajjour_frame_identification)

    ein_dor_premise_detection = tasks_path() / "ein_dor_20_argument_detection" / "ein_dor_20_argument_detection.json"
    print_data_size("Corpus Wide Argument Mining - a Working Solution", ein_dor_premise_detection)

    gretz20_ibm_quaity_rank_train = tasks_path() / "gretz20_ibm_quality_rank_30k" / "gretz20_ibm_quality_rank_30k_train.json"
    gretz20_ibm_quaity_rank_test = tasks_path() / "gretz20_ibm_quality_rank_30k" / "gretz20_ibm_quality_rank_30k_test.json"
    gretz20_ibm_quaity_rank_dev = tasks_path() / "gretz20_ibm_quality_rank_30k" / "gretz20_ibm_quality_rank_30k_dev.json"

    print_data_size(
        "A Large Scale Dataset for Argument Quality Ranking: Construction and Analysis - Quality Classification",
        gretz20_ibm_quaity_rank_train,
        gretz20_ibm_quaity_rank_test,
        gretz20_ibm_quaity_rank_dev)

    gretz20_ibm_quaity_rank_train = tasks_path() / "gretz20_ibm_quality_rank_30k_pairvise_rank" / "gretz20_ibm_quality_rank_30k_pairvise_rank_train.json"
    gretz20_ibm_quaity_rank_test = tasks_path() / "gretz20_ibm_quality_rank_30k_pairvise_rank" / "gretz20_ibm_quality_rank_30k_pairvise_rank_test.json"
    gretz20_ibm_quaity_rank_dev = tasks_path() / "gretz20_ibm_quality_rank_30k_pairvise_rank" / "gretz20_ibm_quality_rank_30k_pairvise_rank_dev.json"

    print_data_size(
        "A Large Scale Dataset for Argument Quality Ranking: Construction and Analysis - Pairvise Ranking",
        gretz20_ibm_quaity_rank_train,
        gretz20_ibm_quaity_rank_test,
        gretz20_ibm_quaity_rank_dev)


    gretz20_ibm_quaity_rank_train = tasks_path() / "gretz20_ibm_quality_rank_30k_rank" / "gretz20_ibm_quality_rank_30k_rank_train.json"
    gretz20_ibm_quaity_rank_test = tasks_path() / "gretz20_ibm_quality_rank_30k_rank" / "gretz20_ibm_quality_rank_30k_rank_test.json"
    gretz20_ibm_quaity_rank_dev = tasks_path() / "gretz20_ibm_quality_rank_30k_rank" / "gretz20_ibm_quality_rank_30k_rank_dev.json"

    print_data_size(
        "A Large Scale Dataset for Argument Quality Ranking: Construction and Analysis - Full Ranking",
        gretz20_ibm_quaity_rank_train,
        gretz20_ibm_quaity_rank_test,
        gretz20_ibm_quaity_rank_dev)



    gretz20_conclusion_generation_train =  tasks_path() / "gretz20_conclusion_generation" / "gretz20_conclusion_generation_train.json"
    gretz20_conclusion_generation_test =  tasks_path() / "gretz20_conclusion_generation" / "gretz20_conclusion_generation_test.json"
    gretz20_conclusion_generation_dev =  tasks_path() / "gretz20_conclusion_generation" / "gretz20_conclusion_generation_dev.json"

    print_data_size(
        "The workweek is the best time to start a family - A Study of GPT-2 Based Claim Generation",
        gretz20_conclusion_generation_dev,
        gretz20_conclusion_generation_test,
        gretz20_conclusion_generation_train
    )

    wachsmuth_dagstuhl_quality_overall_quality = tasks_path() / "wachsmuth_dagstuhl_quality" / "wachsmuth17_dagstuhl_overall_quality.json"
    wachsmuth_dagstuhl_quality_effectiveness = tasks_path() / "wachsmuth_dagstuhl_quality" / "wachsmuth17_dagstuhl_effectiveness.json"
    wachsmuth_dagstuhl_quality_local_acceptability = tasks_path() / "wachsmuth_dagstuhl_quality" / "wachsmuth17_dagstuhl_local_acceptability.json"
    wachsmuth_dagstuhl_quality_appropriateness = tasks_path() / "wachsmuth_dagstuhl_quality" / "wachsmuth17_dagstuhl_appropriateness.json"
    wachsmuth_dagstuhl_quality_arrangement = tasks_path() / "wachsmuth_dagstuhl_quality" / "wachsmuth17_dagstuhl_arrangement.json"
    wachsmuth_dagstuhl_quality_clarity = tasks_path() / "wachsmuth_dagstuhl_quality" / "wachsmuth17_dagstuhl_clarity.json"
    wachsmuth_dagstuhl_quality_cogency = tasks_path() / "wachsmuth_dagstuhl_quality" / "wachsmuth17_dagstuhl_cogency.json"
    wachsmuth_dagstuhl_quality_global_acceptability = tasks_path() / "wachsmuth_dagstuhl_quality" / "wachsmuth17_dagstuhl_global_acceptability.json"
    wachsmuth_dagstuhl_quality_global_relevance = tasks_path() / "wachsmuth_dagstuhl_quality" / "wachsmuth17_dagstuhl_global_relevance.json"
    wachsmuth_dagstuhl_quality_global_sufficiency = tasks_path() / "wachsmuth_dagstuhl_quality" / "wachsmuth17_dagstuhl_global_sufficiency.json"
    wachsmuth_dagstuhl_quality_reasonableness = tasks_path() / "wachsmuth_dagstuhl_quality" / "wachsmuth17_dagstuhl_reasonableness.json"
    wachsmuth_dagstuhl_quality_local_relevance = tasks_path() / "wachsmuth_dagstuhl_quality" / "wachsmuth17_dagstuhl_local_relevance.json"
    wachsmuth_dagstuhl_quality_credibility = tasks_path() / "wachsmuth_dagstuhl_quality" / "wachsmuth17_dagstuhl_credibility.json"
    wachsmuth_dagstuhl_quality_emotional_appeal = tasks_path() / "wachsmuth_dagstuhl_quality" / "wachsmuth17_dagstuhl_emotional_appeal.json"
    wachsmuth_dagstuhl_quality_sufficiency = tasks_path() / "wachsmuth_dagstuhl_quality" / "wachsmuth17_dagstuhl_sufficiency.json"

    print_data_size(
        "Computational Argumentation Quality Assesment in Natural Language - Quality Classification",
        wachsmuth_dagstuhl_quality_overall_quality,
        wachsmuth_dagstuhl_quality_effectiveness,
        wachsmuth_dagstuhl_quality_local_acceptability,
        wachsmuth_dagstuhl_quality_appropriateness,
        wachsmuth_dagstuhl_quality_arrangement,
        wachsmuth_dagstuhl_quality_clarity,
        wachsmuth_dagstuhl_quality_cogency,
        wachsmuth_dagstuhl_quality_global_acceptability,
        wachsmuth_dagstuhl_quality_global_relevance,
        wachsmuth_dagstuhl_quality_global_sufficiency,
        wachsmuth_dagstuhl_quality_reasonableness,
        wachsmuth_dagstuhl_quality_local_relevance,
        wachsmuth_dagstuhl_quality_credibility,
        wachsmuth_dagstuhl_quality_emotional_appeal,
        wachsmuth_dagstuhl_quality_sufficiency
    )

    wachsmuth_dagstuhl_quality_overall_quality = tasks_path() / "wachsmuth_dagstuhl_quality_pairvise_rank" / "wachsmuth17_dagstuhl_overall_quality_pairvise_rank.json"
    wachsmuth_dagstuhl_quality_effectiveness = tasks_path() / "wachsmuth_dagstuhl_quality_pairvise_rank" / "wachsmuth17_dagstuhl_effectiveness_pairvise_rank.json"
    wachsmuth_dagstuhl_quality_local_acceptability = tasks_path() / "wachsmuth_dagstuhl_quality_pairvise_rank" / "wachsmuth17_dagstuhl_local_acceptability_pairvise_rank.json"
    wachsmuth_dagstuhl_quality_appropriateness = tasks_path() / "wachsmuth_dagstuhl_quality_pairvise_rank" / "wachsmuth17_dagstuhl_appropriateness_pairvise_rank.json"
    wachsmuth_dagstuhl_quality_arrangement = tasks_path() / "wachsmuth_dagstuhl_quality_pairvise_rank" / "wachsmuth17_dagstuhl_arrangement_pairvise_rank.json"
    wachsmuth_dagstuhl_quality_clarity = tasks_path() / "wachsmuth_dagstuhl_quality_pairvise_rank" / "wachsmuth17_dagstuhl_clarity_pairvise_rank.json"
    wachsmuth_dagstuhl_quality_cogency = tasks_path() / "wachsmuth_dagstuhl_quality_pairvise_rank" / "wachsmuth17_dagstuhl_cogency_pairvise_rank.json"
    wachsmuth_dagstuhl_quality_global_acceptability = tasks_path() / "wachsmuth_dagstuhl_quality_pairvise_rank" / "wachsmuth17_dagstuhl_global_acceptability_pairvise_rank.json"
    wachsmuth_dagstuhl_quality_global_relevance = tasks_path() / "wachsmuth_dagstuhl_quality_pairvise_rank" / "wachsmuth17_dagstuhl_global_relevance_pairvise_rank.json"
    wachsmuth_dagstuhl_quality_global_sufficiency = tasks_path() / "wachsmuth_dagstuhl_quality_pairvise_rank" / "wachsmuth17_dagstuhl_global_sufficiency_pairvise_rank.json"
    wachsmuth_dagstuhl_quality_reasonableness = tasks_path() / "wachsmuth_dagstuhl_quality_pairvise_rank" / "wachsmuth17_dagstuhl_reasonableness_pairvise_rank.json"
    wachsmuth_dagstuhl_quality_local_relevance = tasks_path() / "wachsmuth_dagstuhl_quality_pairvise_rank" / "wachsmuth17_dagstuhl_local_relevance_pairvise_rank.json"
    wachsmuth_dagstuhl_quality_credibility = tasks_path() / "wachsmuth_dagstuhl_quality_pairvise_rank" / "wachsmuth17_dagstuhl_credibility_pairvise_rank.json"
    wachsmuth_dagstuhl_quality_emotional_appeal = tasks_path() / "wachsmuth_dagstuhl_quality_pairvise_rank" / "wachsmuth17_dagstuhl_emotional_appeal_pairvise_rank.json"
    wachsmuth_dagstuhl_quality_sufficiency = tasks_path() / "wachsmuth_dagstuhl_quality_pairvise_rank" / "wachsmuth17_dagstuhl_sufficiency_pairvise_rank.json"

    print_data_size(
        "Computational Argumentation Quality Assesment in Natural Language - Pairvise Ranking",
        wachsmuth_dagstuhl_quality_overall_quality,
        wachsmuth_dagstuhl_quality_effectiveness,
        wachsmuth_dagstuhl_quality_local_acceptability,
        wachsmuth_dagstuhl_quality_appropriateness,
        wachsmuth_dagstuhl_quality_arrangement,
        wachsmuth_dagstuhl_quality_clarity,
        wachsmuth_dagstuhl_quality_cogency,
        wachsmuth_dagstuhl_quality_global_acceptability,
        wachsmuth_dagstuhl_quality_global_relevance,
        wachsmuth_dagstuhl_quality_global_sufficiency,
        wachsmuth_dagstuhl_quality_reasonableness,
        wachsmuth_dagstuhl_quality_local_relevance,
        wachsmuth_dagstuhl_quality_credibility,
        wachsmuth_dagstuhl_quality_emotional_appeal,
        wachsmuth_dagstuhl_quality_sufficiency
    )

    wachsmuth_dagstuhl_quality_overall_quality = tasks_path() / "wachsmuth_dagstuhl_quality_rank" / "wachsmuth17_dagstuhl_overall_quality_rank.json"
    wachsmuth_dagstuhl_quality_effectiveness = tasks_path() / "wachsmuth_dagstuhl_quality_rank" / "wachsmuth17_dagstuhl_effectiveness_rank.json"
    wachsmuth_dagstuhl_quality_local_acceptability = tasks_path() / "wachsmuth_dagstuhl_quality_rank" / "wachsmuth17_dagstuhl_local_acceptability_rank.json"
    wachsmuth_dagstuhl_quality_appropriateness = tasks_path() / "wachsmuth_dagstuhl_quality_rank" / "wachsmuth17_dagstuhl_appropriateness_rank.json"
    wachsmuth_dagstuhl_quality_arrangement = tasks_path() / "wachsmuth_dagstuhl_quality_rank" / "wachsmuth17_dagstuhl_arrangement_rank.json"
    wachsmuth_dagstuhl_quality_clarity = tasks_path() / "wachsmuth_dagstuhl_quality_rank" / "wachsmuth17_dagstuhl_clarity_rank.json"
    wachsmuth_dagstuhl_quality_cogency = tasks_path() / "wachsmuth_dagstuhl_quality_rank" / "wachsmuth17_dagstuhl_cogency_rank.json"
    wachsmuth_dagstuhl_quality_global_acceptability = tasks_path() / "wachsmuth_dagstuhl_quality_rank" / "wachsmuth17_dagstuhl_global_acceptability_rank.json"
    wachsmuth_dagstuhl_quality_global_relevance = tasks_path() / "wachsmuth_dagstuhl_quality_rank" / "wachsmuth17_dagstuhl_global_relevance_rank.json"
    wachsmuth_dagstuhl_quality_global_sufficiency = tasks_path() / "wachsmuth_dagstuhl_quality_rank" / "wachsmuth17_dagstuhl_global_sufficiency_rank.json"
    wachsmuth_dagstuhl_quality_reasonableness = tasks_path() / "wachsmuth_dagstuhl_quality_rank" / "wachsmuth17_dagstuhl_reasonableness_rank.json"
    wachsmuth_dagstuhl_quality_local_relevance = tasks_path() / "wachsmuth_dagstuhl_quality_rank" / "wachsmuth17_dagstuhl_local_relevance_rank.json"
    wachsmuth_dagstuhl_quality_credibility = tasks_path() / "wachsmuth_dagstuhl_quality_rank" / "wachsmuth17_dagstuhl_credibility_rank.json"
    wachsmuth_dagstuhl_quality_emotional_appeal = tasks_path() / "wachsmuth_dagstuhl_quality_rank" / "wachsmuth17_dagstuhl_emotional_appeal_rank.json"
    wachsmuth_dagstuhl_quality_sufficiency = tasks_path() / "wachsmuth_dagstuhl_quality_rank" / "wachsmuth17_dagstuhl_sufficiency_rank.json"

    print_data_size(
        "Computational Argumentation Quality Assesment in Natural Language - Full Ranking",
        wachsmuth_dagstuhl_quality_overall_quality,
        wachsmuth_dagstuhl_quality_effectiveness,
        wachsmuth_dagstuhl_quality_local_acceptability,
        wachsmuth_dagstuhl_quality_appropriateness,
        wachsmuth_dagstuhl_quality_arrangement,
        wachsmuth_dagstuhl_quality_clarity,
        wachsmuth_dagstuhl_quality_cogency,
        wachsmuth_dagstuhl_quality_global_acceptability,
        wachsmuth_dagstuhl_quality_global_relevance,
        wachsmuth_dagstuhl_quality_global_sufficiency,
        wachsmuth_dagstuhl_quality_reasonableness,
        wachsmuth_dagstuhl_quality_local_relevance,
        wachsmuth_dagstuhl_quality_credibility,
        wachsmuth_dagstuhl_quality_emotional_appeal,
        wachsmuth_dagstuhl_quality_sufficiency
    )

    ajjour_unit_segmentation_essays_train = tasks_path() / "ajjour_unit_segmentation" / "ajjour_unit_segmentation_essays_train.json"
    ajjour_unit_segmentation_essays_test = tasks_path() / "ajjour_unit_segmentation" / "ajjour_unit_segmentation_editorials_test.json"

    ajjour_unit_segmentation_editorials_train = tasks_path() / "ajjour_unit_segmentation" / "ajjour_unit_segmentation_editorials_train.json"
    ajjour_unit_segmentation_editorials_test = tasks_path() / "ajjour_unit_segmentation" / "ajjour_unit_segmentation_editorials_test.json"

    ajjour_unit_segmentation_web_discourse_train = tasks_path() / "ajjour_unit_segmentation" / "ajjour_unit_segmentation_web_discourse_train.json"
    ajjour_unit_segmentation_web_discourse_test = tasks_path() / "ajjour_unit_segmentation" / "ajjour_unit_segmentation_web_discourse_test.json"

    print_data_size(
        "Unit Segmentation of Argumentative Tasks - Extract Argument",
        ajjour_unit_segmentation_editorials_test,
        ajjour_unit_segmentation_editorials_train,
        ajjour_unit_segmentation_essays_test,
        ajjour_unit_segmentation_essays_train,
        ajjour_unit_segmentation_web_discourse_test,
        ajjour_unit_segmentation_web_discourse_train
    )

    ajjour_unit_segmentation_essays_train = tasks_path() / "ajjour_unit_segmentation_entity" / "ajjour_unit_segmentation_entity_essays_train.json"
    ajjour_unit_segmentation_essays_test = tasks_path() / "ajjour_unit_segmentation_entity" / "ajjour_unit_segmentation_entity_editorials_test.json"

    ajjour_unit_segmentation_editorials_train = tasks_path() / "ajjour_unit_segmentation_entity" / "ajjour_unit_segmentation_entity_editorials_train.json"
    ajjour_unit_segmentation_editorials_test = tasks_path() / "ajjour_unit_segmentation_entity" / "ajjour_unit_segmentation_entity_editorials_test.json"

    ajjour_unit_segmentation_web_discourse_train = tasks_path() / "ajjour_unit_segmentation_entity" / "ajjour_unit_segmentation_entity_web_discourse_train.json"
    ajjour_unit_segmentation_web_discourse_test = tasks_path() / "ajjour_unit_segmentation_entity" / "ajjour_unit_segmentation_entity_web_discourse_test.json"

    print_data_size(
        "Unit Segmentation of Argumentative Tasks - Entity Segmentation",
        ajjour_unit_segmentation_editorials_test,
        ajjour_unit_segmentation_editorials_train,
        ajjour_unit_segmentation_essays_test,
        ajjour_unit_segmentation_essays_train,
        ajjour_unit_segmentation_web_discourse_test,
        ajjour_unit_segmentation_web_discourse_train
    )

    stab18_stance_classification_train = tasks_path() / "stab18_stance_classification" / "stab18_stance_classification_train.json"
    stab18_stance_classification_test = tasks_path() / "stab18_stance_classification" / "stab18_stance_classification_test.json"
    stab18_stance_classification_val = tasks_path() / "stab18_stance_classification" / "stab18_stance_classification_val.json"

    print_data_size(
        "Cross-Topic Argument Mining from Heterogeneous Sources",
        stab18_stance_classification_test,
        stab18_stance_classification_train,
        stab18_stance_classification_val
    )

    barhaim21_key_point = tasks_path() / "barhaim21_key_point" / "barhaim21_key_point.json"

    print_data_size(
        "From Arguments to Key Points: Towards Automatic Argument Summarization",
        barhaim21_key_point
    )

    ibmsc_stance_classification_train = tasks_path() / "ibmsc_stance_classification" / "ibmsc_stance_classification_train.json"
    ibmsc_stance_classification_test = tasks_path() / "ibmsc_stance_classification" / "ibmsc_stance_classification_test.json"
    ibmsc_stance_classification_dev = tasks_path() / "ibmsc_stance_classification" / "ibmsc_stance_classification_dev.json"

    print_data_size(
        "Stance Classification of Context-Dependent Claims",
        ibmsc_stance_classification_train,
        ibmsc_stance_classification_dev,
        ibmsc_stance_classification_test
    )

    ukp_aspect_argument_similarity = tasks_path() / "ukp_aspect_argument_similarity" / "ukp_aspect_argument_similarity.json"

    print_data_size(
        "Classification and Clustering of Arguments with Contextualized Word Embeddings",
        ukp_aspect_argument_similarity
    )

    saha_23_argument_extraction_train = tasks_path() / "saha_23_argument_extraction" / "saha_23_argument_extraction_train.json"
    saha_23_argument_extraction_test = tasks_path() / "saha_23_argument_extraction" / "saha_23_argument_extraction_test.json"

    print_data_size(
        "ArgU: A Controllable Factual Argument Generator - Extract Argument",
        saha_23_argument_extraction_test,
        saha_23_argument_extraction_train
    )

    saha_23_argument_extraction_train = tasks_path() / "saha_23_argument_scheme_identification" / "saha_23_argument_scheme_test.json"
    saha_23_argument_extraction_test = tasks_path() / "saha_23_argument_scheme_identification" / "saha_23_argument_scheme_train.json"

    print_data_size(
        "ArgU: A Controllable Factual Argument Generator - Argument Scheme Classification",
        saha_23_argument_extraction_test,
        saha_23_argument_extraction_train
    )

    saha_23_argument_extraction_train = tasks_path() / "saha_23_stance_detection" / "saha_23_stance_detection_test.json"
    saha_23_argument_extraction_test = tasks_path() / "saha_23_stance_detection" / "saha_23_stance_detection_train.json"

    print_data_size(
        "ArgU: A Controllable Factual Argument Generator - Stance Classification",
        saha_23_argument_extraction_test,
        saha_23_argument_extraction_train
    )

    schiller21_aspect_argument_generation = tasks_path() / "schiller21_aspect_argument_generation" / "schiller21_aspect_argument_generation.json"

    print_data_size(
        "Aspect-Controlled Neural Argument Generation",
        schiller21_aspect_argument_generation
    )

    hua_18_counter_argument_generation_train = tasks_path() / "hua_18_counter_argument_generation" / "hua_18_counter_argument_generation_train.json"
    hua_18_counter_argument_generation_test = tasks_path() / "hua_18_counter_argument_generation" / "hua_18_counter_argument_generation_test.json"
    hua_18_counter_argument_generation_valid = tasks_path() / "hua_18_counter_argument_generation" / "hua_18_counter_argument_generation_valid.json"

    print_data_size(
        "Neural Argument Generation Augmented with Externally Retrieved Evidence",
        hua_18_counter_argument_generation_test,
        hua_18_counter_argument_generation_train,
        hua_18_counter_argument_generation_valid
    )

    peldzus15_argument_relation_identification = tasks_path() / "peldzus15_argument_relation_identification" / "peldzus15_argument_relation_identification.json"

    print_data_size(
        "An Annotated Corpus of Argumentative Microtexts - Argument Relation Identification",
        peldzus15_argument_relation_identification
    )

    skepstedt18_argument_relation_identification = tasks_path() / "skepstedt18_argument_relation_identification" / "skepstedt18_argument_relation_identification.json"

    print_data_size(
        "More or less controlled elicitation of argumentative text: Enlarging a microtext corpus via crowdsourcing - Argument Relation Identification",
        skepstedt18_argument_relation_identification
    )

    chakarbarty21_implicit_premise_generation_art_train = tasks_path() / "chakarbarty21_implicit_premise_generation" / "chakarbarty21_implicit_premise_generation_art_train.json"
    chakarbarty21_implicit_premise_generation_art_train_para_comet = tasks_path() / "chakarbarty21_implicit_premise_generation" / "chakarbarty21_implicit_premise_generation_art_train_para_comet.json"
    chakarbarty21_implicit_premise_generation_art_val = tasks_path() / "chakarbarty21_implicit_premise_generation" / "chakarbarty21_implicit_premise_generation_art_val.json"
    chakarbarty21_implicit_premise_generation_art_val_para_comet = tasks_path() / "chakarbarty21_implicit_premise_generation" / "chakarbarty21_implicit_premise_generation_art_val_para_comet.json"
    chakarbarty21_implicit_premise_generation_d1test = tasks_path() / "chakarbarty21_implicit_premise_generation" / "chakarbarty21_implicit_premise_generation_d1test.json"
    chakarbarty21_implicit_premise_generation_d1test_para_comet = tasks_path() / "chakarbarty21_implicit_premise_generation" / "chakarbarty21_implicit_premise_generation_d1test_para_comet.json"
    chakarbarty21_implicit_premise_generation_d2test = tasks_path() / "chakarbarty21_implicit_premise_generation" / "chakarbarty21_implicit_premise_generation_d2test.json"
    chakarbarty21_implicit_premise_generation_d2test_para_comet = tasks_path() / "chakarbarty21_implicit_premise_generation" / "chakarbarty21_implicit_premise_generation_d2test_para_comet.json"
    chakarbarty21_implicit_premise_generation_d3test = tasks_path() / "chakarbarty21_implicit_premise_generation" / "chakarbarty21_implicit_premise_generation_d3test.json"
    chakarbarty21_implicit_premise_generation_d3test_para_comet = tasks_path() / "chakarbarty21_implicit_premise_generation" / "chakarbarty21_implicit_premise_generation_d3test_para_comet.json"

    print_data_size(
        "Implicit Premise Generation with Discourse-aware Commonsense Knowledge Models",
        chakarbarty21_implicit_premise_generation_art_train,
        chakarbarty21_implicit_premise_generation_art_train_para_comet,
        chakarbarty21_implicit_premise_generation_art_val,
        chakarbarty21_implicit_premise_generation_art_val_para_comet,
        chakarbarty21_implicit_premise_generation_d1test,
        chakarbarty21_implicit_premise_generation_d1test_para_comet,
        chakarbarty21_implicit_premise_generation_d2test,
        chakarbarty21_implicit_premise_generation_d2test_para_comet,
        chakarbarty21_implicit_premise_generation_d3test,
        chakarbarty21_implicit_premise_generation_d3test_para_comet
    )

    aharoni14_claim_evidence_extraction = tasks_path() / "aharoni14_claim_evidence_extraction" / "aharoni14_claim_evidence_extraction.json"

    print_data_size(
        "A Benchmark Dataset for Automatic Detection of Claims and Evidence in the Context of Controversial Topics - Evidence Extraction",
        aharoni14_claim_evidence_extraction
    )

    aharoni14_claim_extraction = tasks_path() / "aharoni14_claim_extraction" / "aharoni14_claim_extraction.json"

    print_data_size(
        "A Benchmark Dataset for Automatic Detection of Claims and Evidence in the Context of Controversial Topics - Claim Extraction",
        aharoni14_claim_extraction
    )

    alkhatib16_argumentation_strategy_mining_unit_segmentation = tasks_path() / "alkhatib16_argumentation_strategy_mining_unit_segmentation" / "alkhatib16_argumentation_strategy_mining_unit_segmentation.json"

    print_data_size(
        "A News Editorial Corpus for Mining Argumentation Strategies - Extract Argument",
        alkhatib16_argumentation_strategy_mining_unit_segmentation
    )

    alkhatib16_argumentation_strategy_mining_unit_segmentation_entity = tasks_path() / "alkhatib16_argumentation_strategy_mining_unit_segmentation_entity" / "alkhatib16_argumentation_strategy_mining_unit_segmentation_entity.json"

    print_data_size(
        "A News Editorial Corpus for Mining Argumentation Strategies - Entity Segmentation",
        alkhatib16_argumentation_strategy_mining_unit_segmentation_entity
    )

    habernal18_ad_hominem_detection = tasks_path() / "habernal18-ad-hominem-detection" / "habernal18-ad-hominem-detection.json"

    print_data_size(
        "Before Name-calling: Dynamics and Triggers of Ad Hominem Fallacies in Web Argumentation",
        habernal18_ad_hominem_detection
    )

    habernal18_implicit_warrant_identification_dev = tasks_path() / "habernal18_implicit_warrant_identification" / "habernal18_implicit_warrant_identification_dev.json"
    habernal18_implicit_warrant_identification_train = tasks_path() / "habernal18_implicit_warrant_identification" / "habernal18_implicit_warrant_identification_train.json"
    habernal18_implicit_warrant_identification_test = tasks_path() / "habernal18_implicit_warrant_identification" / "habernal18_implicit_warrant_identification_test.json"

    print_data_size(
        "The Argument Reasoning Comprehension Task: Identification and Reconstruction",
        habernal18_implicit_warrant_identification_dev,
        habernal18_implicit_warrant_identification_test,
        habernal18_implicit_warrant_identification_train
    )

    park18_cornell_erulemaking_classification = tasks_path() / "park18_cornell_erulemaking_classification" / "park18_cornell_erulemaking_classification.json"

    print_data_size(
        "A Corpus of eRulemaking User Comments for Measuring Evaluability of Arguments - Argument Unit Classification",
        park18_cornell_erulemaking_classification
    )

    park18_cornell_erulemaking_identification = tasks_path() / "park18_cornell_erulemaking_identification" / "park18_cornell_erulemaking_identification.json"

    print_data_size(
        "A Corpus of eRulemaking User Comments for Measuring Evaluability of Arguments - Argument Unit Identification",
        park18_cornell_erulemaking_identification
    )

    menini18_relation_identification = tasks_path() / "menini18_relation_identification" / "menini18_relation_identification.json"

    print_data_size(
        "Never Retreat, Never Retract: Argumentation Analysis for Political Speeches - Relation Identification",
        menini18_relation_identification
    )

    skitalinskaya23_claim_optimization_dev = tasks_path() / "skitalinskaya23_claim_optimization" / "skitalinskaya23_claim_optimization_dev.json"
    skitalinskaya23_claim_optimization_test = tasks_path() / "skitalinskaya23_claim_optimization" / "skitalinskaya23_claim_optimization_test.json"
    skitalinskaya23_claim_optimization_train = tasks_path() / "skitalinskaya23_claim_optimization" / "skitalinskaya23_claim_optimization_train.json"

    print_data_size(
        "Claim Optimization in Computational Argumentation - Claim Optimization",
        skitalinskaya23_claim_optimization_dev,
        skitalinskaya23_claim_optimization_test,
        skitalinskaya23_claim_optimization_train
    )

    stab17_argument_relation_identification = tasks_path() / "stab17_argument_relation_identification" / "stab17_argument_relation_identification.json"

    print_data_size(
        "Parsing Argumentation Structures in Persuasive Essays - Argument Relation Identification",
        stab17_argument_relation_identification
    )

    alshomary21_belief_based_argument_generation_train = tasks_path() / "alshomary21_belief_based_argument_generation" / "alshomary21_belief_based_argument_generation_train.json"
    alshomary21_belief_based_argument_generation_test = tasks_path() / "alshomary21_belief_based_argument_generation" / "alshomary21_belief_based_argument_generation_test.json"
    alshomary21_belief_based_argument_generation_valid = tasks_path() / "alshomary21_belief_based_argument_generation" / "alshomary21_belief_based_argument_generation_valid.json"

    print_data_size(
        "Belief-based Generation of Argumentative Claims",
        alshomary21_belief_based_argument_generation_train,
        alshomary21_belief_based_argument_generation_test,
        alshomary21_belief_based_argument_generation_valid
    )
