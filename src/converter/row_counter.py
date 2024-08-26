from common import tasks_path
import json

def print_data_size(dataset_name, *data_paths):
    total_instances = 0

    for path in data_paths:
        with open(path, "r") as f:
            dataset = json.load(f)
            total_instances += len(dataset["Instances"])

    print(f"| {dataset_name} | {total_instances} |")


if __name__ == "__main__":
    frame_identification_webis_argument_framing_19_ajjour19 = (tasks_path() /
                                   "frame_identification_webis_argument_framing_19_ajjour19" /
                                   "frame_identification_webis_argument_framing_19_ajjour19.json")
    print_data_size("Unit Segmentation of Argumentative texts - Frame Identification", frame_identification_webis_argument_framing_19_ajjour19)

    frame_identification_webis_stance_classification_19_ajjour19 = (tasks_path() /
                                                                    "frame_identification_webis_stance_classification_19_ajjour19" /
                                                                    "frame_identification_webis_stance_classification_19_ajjour19.json")
    print_data_size("Unit Segmentation of Argumentative texts - Stance Classification", frame_identification_webis_stance_classification_19_ajjour19)

    ein_dor_premise_detection = tasks_path() / "argument_unit_classificaiton_wikipedia_articles_lexisnexis_eindor20" / "argument_unit_classificaiton_wikipedia_articles_lexisnexis_eindor20.json"
    print_data_size("Corpus Wide Argument Mining - a Working Solution", ein_dor_premise_detection)

    gretz20_ibm_quaity_rank_train = tasks_path() / "argument_ranking_ibm_rank_30k_class_rank_gretz20" / "argument_ranking_ibm_rank_30k_class_rank_train_gretz20.json"
    gretz20_ibm_quaity_rank_test = tasks_path() / "argument_ranking_ibm_rank_30k_class_rank_gretz20" / "argument_ranking_ibm_rank_30k_class_rank_test_gretz20.json"
    gretz20_ibm_quaity_rank_dev = tasks_path() / "argument_ranking_ibm_rank_30k_class_rank_gretz20" / "argument_ranking_ibm_rank_30k_class_rank_dev_gretz20.json"

    print_data_size(
        "A Large Scale Dataset for Argument Quality Ranking: Construction and Analysis - Quality Classification",
        gretz20_ibm_quaity_rank_train,
        gretz20_ibm_quaity_rank_test,
        gretz20_ibm_quaity_rank_dev)

    gretz20_ibm_quaity_rank_train = tasks_path() / "argument_ranking_ibm_rank_30k_pairvise_rank_gretz20" / "argument_ranking_ibm_rank_30k_pairvise_rank_train_gretz20.json"
    gretz20_ibm_quaity_rank_test = tasks_path() / "argument_ranking_ibm_rank_30k_pairvise_rank_gretz20" / "argument_ranking_ibm_rank_30k_pairvise_rank_test_gretz20.json"
    gretz20_ibm_quaity_rank_dev = tasks_path() / "argument_ranking_ibm_rank_30k_pairvise_rank_gretz20" / "argument_ranking_ibm_rank_30k_pairvise_rank_dev_gretz20.json"

    print_data_size(
        "A Large Scale Dataset for Argument Quality Ranking: Construction and Analysis - Pairvise Ranking",
        gretz20_ibm_quaity_rank_train,
        gretz20_ibm_quaity_rank_test,
        gretz20_ibm_quaity_rank_dev)


    gretz20_ibm_quaity_rank_train = tasks_path() / "argument_ranking_ibm_rank_30k_full_rank_gretz20" / "argument_ranking_ibm_rank_30k_full_rank_train_gretz20.json"
    gretz20_ibm_quaity_rank_test = tasks_path() / "argument_ranking_ibm_rank_30k_full_rank_gretz20" / "argument_ranking_ibm_rank_30k_full_rank_train_gretz20.json"
    gretz20_ibm_quaity_rank_dev = tasks_path() / "argument_ranking_ibm_rank_30k_full_rank_gretz20" / "argument_ranking_ibm_rank_30k_full_rank_train_gretz20.json"

    print_data_size(
        "A Large Scale Dataset for Argument Quality Ranking: Construction and Analysis - Full Ranking",
        gretz20_ibm_quaity_rank_train,
        gretz20_ibm_quaity_rank_test,
        gretz20_ibm_quaity_rank_dev)


    gretz20_conclusion_generation_train =  tasks_path() / "conclusion_generation_ibm_claim_generation_gretz20" / "conclusion_generation_ibm_claim_generation_train_gretz20.json"
    gretz20_conclusion_generation_test =  tasks_path() / "conclusion_generation_ibm_claim_generation_gretz20" / "conclusion_generation_ibm_claim_generation_test_gretz20.json"
    gretz20_conclusion_generation_dev =  tasks_path() / "conclusion_generation_ibm_claim_generation_gretz20" / "conclusion_generation_ibm_claim_generation_dev_gretz20.json"

    print_data_size(
        "The workweek is the best time to start a family - A Study of GPT-2 Based Claim Generation",
        gretz20_conclusion_generation_dev,
        gretz20_conclusion_generation_test,
        gretz20_conclusion_generation_train
    )

    wachsmuth_dagstuhl_quality_overall_quality = tasks_path() / "argument_ranking_dagstuhl_15512_class_rank_wachsmuth17" / "argument_ranking_dagstuhl_15512_class_rank_overall_quality_wachsmuth17.json"
    wachsmuth_dagstuhl_quality_effectiveness = tasks_path() / "argument_ranking_dagstuhl_15512_class_rank_wachsmuth17" / "argument_ranking_dagstuhl_15512_class_rank_effectiveness_wachsmuth17.json"
    wachsmuth_dagstuhl_quality_local_acceptability = tasks_path() / "argument_ranking_dagstuhl_15512_class_rank_wachsmuth17" / "argument_ranking_dagstuhl_15512_class_rank_local_acceptability_wachsmuth17.json"
    wachsmuth_dagstuhl_quality_appropriateness = tasks_path() / "argument_ranking_dagstuhl_15512_class_rank_wachsmuth17" / "argument_ranking_dagstuhl_15512_class_rank_appropriateness_wachsmuth17.json"
    wachsmuth_dagstuhl_quality_arrangement = tasks_path() / "argument_ranking_dagstuhl_15512_class_rank_wachsmuth17" / "argument_ranking_dagstuhl_15512_class_rank_arrangement_wachsmuth17.json"
    wachsmuth_dagstuhl_quality_clarity = tasks_path() / "argument_ranking_dagstuhl_15512_class_rank_wachsmuth17" / "argument_ranking_dagstuhl_15512_class_rank_clarity_wachsmuth17.json"
    wachsmuth_dagstuhl_quality_cogency = tasks_path() / "argument_ranking_dagstuhl_15512_class_rank_wachsmuth17" / "argument_ranking_dagstuhl_15512_class_rank_cogency_wachsmuth17.json"
    wachsmuth_dagstuhl_quality_global_acceptability = tasks_path() / "argument_ranking_dagstuhl_15512_class_rank_wachsmuth17" / "argument_ranking_dagstuhl_15512_class_rank_global_acceptability_wachsmuth17.json"
    wachsmuth_dagstuhl_quality_global_relevance = tasks_path() / "argument_ranking_dagstuhl_15512_class_rank_wachsmuth17" / "argument_ranking_dagstuhl_15512_class_rank_global_relevance_wachsmuth17.json"
    wachsmuth_dagstuhl_quality_global_sufficiency = tasks_path() / "argument_ranking_dagstuhl_15512_class_rank_wachsmuth17" / "argument_ranking_dagstuhl_15512_class_rank_global_sufficiency_wachsmuth17.json"
    wachsmuth_dagstuhl_quality_reasonableness = tasks_path() / "argument_ranking_dagstuhl_15512_class_rank_wachsmuth17" / "argument_ranking_dagstuhl_15512_class_rank_reasonableness_wachsmuth17.json"
    wachsmuth_dagstuhl_quality_local_relevance = tasks_path() / "argument_ranking_dagstuhl_15512_class_rank_wachsmuth17" / "argument_ranking_dagstuhl_15512_class_rank_local_relevance_wachsmuth17.json"
    wachsmuth_dagstuhl_quality_credibility = tasks_path() / "argument_ranking_dagstuhl_15512_class_rank_wachsmuth17" / "argument_ranking_dagstuhl_15512_class_rank_credibility_wachsmuth17.json"
    wachsmuth_dagstuhl_quality_emotional_appeal = tasks_path() / "argument_ranking_dagstuhl_15512_class_rank_wachsmuth17" / "argument_ranking_dagstuhl_15512_class_rank_emotional_appeal_wachsmuth17.json"
    wachsmuth_dagstuhl_quality_sufficiency = tasks_path() / "argument_ranking_dagstuhl_15512_class_rank_wachsmuth17" / "argument_ranking_dagstuhl_15512_class_rank_sufficiency_wachsmuth17.json"

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

    ajjour_unit_segmentation_essays_train = tasks_path() / "argument_unit_segmentation_ajjour17" / "argument_unit_segmentation_essays_train_ajjour17.json"
    ajjour_unit_segmentation_essays_test = tasks_path() / "argument_unit_segmentation_ajjour17" / "argument_unit_segmentation_essays_test_ajjour17.json"

    ajjour_unit_segmentation_editorials_train = tasks_path() / "argument_unit_segmentation_ajjour17" / "argument_unit_segmentation_webis_editorials_train_ajjour17.json"
    ajjour_unit_segmentation_editorials_test = tasks_path() / "argument_unit_segmentation_ajjour17" / "argument_unit_segmentation_webis_editorials_test_ajjour17.json"

    ajjour_unit_segmentation_web_discourse_train = tasks_path() / "argument_unit_segmentation_ajjour17" / "argument_unit_segmentation_web_discourse_train_ajjour17.json"
    ajjour_unit_segmentation_web_discourse_test = tasks_path() / "argument_unit_segmentation_ajjour17" / "argument_unit_segmentation_web_discourse_test_ajjour17.json"

    print_data_size(
        "Unit Segmentation of Argumentative Tasks - Extract Argument",
        ajjour_unit_segmentation_editorials_test,
        ajjour_unit_segmentation_editorials_train,
        ajjour_unit_segmentation_essays_test,
        ajjour_unit_segmentation_essays_train,
        ajjour_unit_segmentation_web_discourse_test,
        ajjour_unit_segmentation_web_discourse_train
    )

    ajjour_unit_segmentation_essays_train = tasks_path() / "argument_unit_segmentation_entity_ajjour17" / "argument_unit_segmentation_essays_entity_train_ajjour17.json"
    ajjour_unit_segmentation_essays_test = tasks_path() / "argument_unit_segmentation_entity_ajjour17" / "argument_unit_segmentation_essays_entity_test_ajjour17.json"

    ajjour_unit_segmentation_editorials_train = tasks_path() / "argument_unit_segmentation_entity_ajjour17" / "argument_unit_segmentation_webis_editorials_entity_train_ajjour17.json"
    ajjour_unit_segmentation_editorials_test = tasks_path() / "argument_unit_segmentation_entity_ajjour17" / "argument_unit_segmentation_webis_editorials_entity_test_ajjour17.json"

    ajjour_unit_segmentation_web_discourse_train = tasks_path() / "argument_unit_segmentation_entity_ajjour17" / "argument_unit_segmentation_web_discourse_entity_train_ajjour17.json"
    ajjour_unit_segmentation_web_discourse_test = tasks_path() / "argument_unit_segmentation_entity_ajjour17" / "argument_unit_segmentation_web_discourse_entity_test_ajjour17.json"

    print_data_size(
        "Unit Segmentation of Argumentative Tasks - Entity Segmentation",
        ajjour_unit_segmentation_editorials_test,
        ajjour_unit_segmentation_editorials_train,
        ajjour_unit_segmentation_essays_test,
        ajjour_unit_segmentation_essays_train,
        ajjour_unit_segmentation_web_discourse_test,
        ajjour_unit_segmentation_web_discourse_train
    )

    stab18_stance_classification_train = tasks_path() / "stance_classification_ukp_sentential_stab18" / "stance_classification_ukp_sentential_train_stab18.json"
    stab18_stance_classification_test = tasks_path() / "stance_classification_ukp_sentential_stab18" / "stance_classification_ukp_sentential_test_stab18.json"
    stab18_stance_classification_val = tasks_path() / "stance_classification_ukp_sentential_stab18" / "stance_classification_ukp_sentential_val_stab18.json"

    print_data_size(
        "Cross-Topic Argument Mining from Heterogeneous Sources",
        stab18_stance_classification_test,
        stab18_stance_classification_train,
        stab18_stance_classification_val
    )

    barhaim21_key_point_train = tasks_path() / "stance_classification_ibmsc_barhaim17" / "stance_classification_ibmsc_train_barhaim17.json"
    barhaim21_key_point_test = tasks_path() / "stance_classification_ibmsc_barhaim17" / "stance_classification_ibmsc_test_barhaim17.json"
    barhaim21_key_point_dev = tasks_path() / "stance_classification_ibmsc_barhaim17" / "stance_classification_ibmsc_dev_barhaim17.json"

    print_data_size(
        "From Arguments to Key Points: Towards Automatic Argument Summarization",
        barhaim21_key_point_train,
        barhaim21_key_point_test,
        barhaim21_key_point_dev,
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

    ukp_aspect_argument_similarity = tasks_path() / "argument_similarity_ukp_aspect_reimers19" / "argument_similarity_ukp_aspect_reimers19.json"

    print_data_size(
        "Classification and Clustering of Arguments with Contextualized Word Embeddings",
        ukp_aspect_argument_similarity
    )

    saha_23_argument_extraction_train = tasks_path() / "premise_extraction_argu_saha23" / "premise_extraction_argu_train_saha23.json"
    saha_23_argument_extraction_test = tasks_path() / "premise_extraction_argu_saha23" / "premise_extraction_argu_test_saha23.json"

    print_data_size(
        "ArgU: A Controllable Factual Argument Generator - Extract Argument",
        saha_23_argument_extraction_test,
        saha_23_argument_extraction_train
    )

    saha_23_argument_extraction_train = tasks_path() / "scheme_classification_argu_saha23" / "scheme_classification_argu_train_saha23.json"
    saha_23_argument_extraction_test = tasks_path() / "scheme_classification_argu_saha23" / "scheme_classification_argu_test_saha23.json"

    print_data_size(
        "ArgU: A Controllable Factual Argument Generator - Argument Scheme Classification",
        saha_23_argument_extraction_test,
        saha_23_argument_extraction_train
    )

    saha_23_argument_extraction_train = tasks_path() / "stance_classification_argu_saha23" / "stance_classification_argu_train_saha23.json"
    saha_23_argument_extraction_test = tasks_path() / "stance_classification_argu_saha23" / "stance_classification_argu_test_saha23.json"

    print_data_size(
        "ArgU: A Controllable Factual Argument Generator - Stance Classification",
        saha_23_argument_extraction_test,
        saha_23_argument_extraction_train
    )

    schiller21_aspect_argument_generation = tasks_path() / "aspect_argument_generation_ukp_aspect_schiller21" / "aspect_argument_generation_ukp_aspect_schiller21.json"

    print_data_size(
        "Aspect-Controlled Neural Argument Generation",
        schiller21_aspect_argument_generation
    )

    hua_18_counter_argument_generation_train = tasks_path() / "counter_argument_generation_cmv_hua18" / "counter_argument_generation_cmv_train_hua18.json"
    hua_18_counter_argument_generation_test = tasks_path() / "counter_argument_generation_cmv_hua18" / "counter_argument_generation_cmv_test_hua18.json"
    hua_18_counter_argument_generation_valid = tasks_path() / "counter_argument_generation_cmv_hua18" / "counter_argument_generation_cmv_valid_hua18.json"

    print_data_size(
        "Neural Argument Generation Augmented with Externally Retrieved Evidence",
        hua_18_counter_argument_generation_test,
        hua_18_counter_argument_generation_train,
        hua_18_counter_argument_generation_valid
    )

    peldzus15_argument_relation_identification = tasks_path() / "premise_generation_microtexts_v1_skeppstedt18" / "premise_generation_microtexts_v1_skeppstedt18.json"

    print_data_size(
        "An Annotated Corpus of Argumentative Microtexts v1 - Argument Relation Identification",
        peldzus15_argument_relation_identification
    )

    peldzus15_argument_relation_identification = tasks_path() / "premise_generation_microtexts_v2_skeppstedt18" / "premise_generation_microtexts_v2_skeppstedt18.json"

    print_data_size(
        "An Annotated Corpus of Argumentative Microtexts v2 - Argument Relation Identification",
        peldzus15_argument_relation_identification
    )

    skepstedt18_argument_relation_identification = tasks_path() / "skepstedt18_argument_relation_identification" / "skepstedt18_argument_relation_identification.json"

    print_data_size(
        "More or less controlled elicitation of argumentative text: Enlarging a microtext corpus via crowdsourcing - Argument Relation Identification",
        skepstedt18_argument_relation_identification
    )

    chakarbarty21_implicit_premise_generation_art_train = tasks_path() / "arc_premise_generation_chakarbarty21" / "arc_premise_generation_train_chakarbarty21.json"
    chakarbarty21_implicit_premise_generation_art_train_para_comet = tasks_path() / "arc_premise_generation_chakarbarty21" / "arc_premise_generation_train_para_comet_chakarbarty21.json"
    chakarbarty21_implicit_premise_generation_art_val = tasks_path() / "arc_premise_generation_chakarbarty21" / "arc_premise_generation_val_chakarbarty21.json"
    chakarbarty21_implicit_premise_generation_art_val_para_comet = tasks_path() / "arc_premise_generation_chakarbarty21" / "arc_premise_generation_val_para_comet_chakarbarty21.json"
    chakarbarty21_implicit_premise_generation_d1test = tasks_path() / "arc_premise_generation_chakarbarty21" / "arc_premise_generation_d1test_chakarbarty21.json"
    chakarbarty21_implicit_premise_generation_d1test_para_comet = tasks_path() / "arc_premise_generation_chakarbarty21" / "arc_premise_generation_d1test_para_comet_chakarbarty21.json"
    chakarbarty21_implicit_premise_generation_d2test = tasks_path() / "arc_premise_generation_chakarbarty21" / "arc_premise_generation_d2test_chakarbarty21.json"
    chakarbarty21_implicit_premise_generation_d2test_para_comet = tasks_path() / "arc_premise_generation_chakarbarty21" / "arc_premise_generation_d2test_para_comet_chakarbarty21.json"
    chakarbarty21_implicit_premise_generation_d3test = tasks_path() / "arc_premise_generation_chakarbarty21" / "arc_premise_generation_d3test_chakarbarty21.json"
    chakarbarty21_implicit_premise_generation_d3test_para_comet = tasks_path() / "arc_premise_generation_chakarbarty21" / "arc_premise_generation_d3test_para_comet_chakarbarty21.json"

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

    evidence_extraction_ibm_claim_evidence_aharoni14 = (
        tasks_path() /
        "evidence_extraction_ibm_claim_evidence_aharoni14" /
        "evidence_extraction_ibm_claim_evidence_aharoni14.json")

    print_data_size(
        "A Benchmark Dataset for Automatic Detection of Claims and Evidence in the Context of Controversial Topics - Evidence Extraction",
        evidence_extraction_ibm_claim_evidence_aharoni14
    )

    conclusion_extraction_ibm_claim_evidence_aharoni14 = (
        tasks_path() /
        "conclusion_extraction_ibm_claim_evidence_aharoni14" /
        "conclusion_extraction_ibm_claim_evidence_aharoni14.json")

    print_data_size(
        "A Benchmark Dataset for Automatic Detection of Claims and Evidence in the Context of Controversial Topics - Claim Extraction",
        conclusion_extraction_ibm_claim_evidence_aharoni14
    )

    alkhatib16_argumentation_strategy_mining_unit_segmentation = tasks_path() / "argument_unit_segmentation_webis_editorials_alkhatib16" / "argument_unit_segmentation_webis_editorials_alkhatib16.json"

    print_data_size(
        "A News Editorial Corpus for Mining Argumentation Strategies - Extract Argument",
        alkhatib16_argumentation_strategy_mining_unit_segmentation
    )

    alkhatib16_argumentation_strategy_mining_unit_segmentation_entity = tasks_path() / "argument_unit_segmentation_webis_editorials_entity_alkhatib16" / "argument_unit_segmentation_webis_editorials_entity_alkhatib16.json"

    print_data_size(
        "A News Editorial Corpus for Mining Argumentation Strategies - Entity Segmentation",
        alkhatib16_argumentation_strategy_mining_unit_segmentation_entity
    )

    habernal18_ad_hominem_detection = tasks_path() / "fallacy_detection_cmv_adhominem_habernal18" / "fallacy_detection_cmv_adhominem_habernal18.json"

    print_data_size(
        "Before Name-calling: Dynamics and Triggers of Ad Hominem Fallacies in Web Argumentation",
        habernal18_ad_hominem_detection
    )

    habernal18_implicit_warrant_identification_dev = tasks_path() / "warrant_identification_semeval_2018_task_12_habernal18" / "warrant_identification_semeval_2018_task_12_dev_habernal18.json"
    habernal18_implicit_warrant_identification_train = tasks_path() / "warrant_identification_semeval_2018_task_12_habernal18" / "warrant_identification_semeval_2018_task_12_train_habernal18.json"
    habernal18_implicit_warrant_identification_test = tasks_path() / "warrant_identification_semeval_2018_task_12_habernal18" / "warrant_identification_semeval_2018_task_12_test_habernal18.json"

    print_data_size(
        "The Argument Reasoning Comprehension Task: Identification and Reconstruction",
        habernal18_implicit_warrant_identification_dev,
        habernal18_implicit_warrant_identification_test,
        habernal18_implicit_warrant_identification_train
    )

    park18_cornell_erulemaking_classification = tasks_path() / "argument_unit_classification_erulemaking_park18" / "argument_unit_classification_erulemaking_park18.json"

    print_data_size(
        "A Corpus of eRulemaking User Comments for Measuring Evaluability of Arguments - Argument Unit Classification",
        park18_cornell_erulemaking_classification
    )

    park18_cornell_erulemaking_identification = tasks_path() / "argument_relation_identification_erulemaking_park18" / "argument_relation_identification_erulemaking_park18.json"

    print_data_size(
        "A Corpus of eRulemaking User Comments for Measuring Evaluability of Arguments - Argument Unit Identification",
        park18_cornell_erulemaking_identification
    )

    menini18_relation_identification = tasks_path() / "argument_relation_identification_political_debates_menini18" / "argument_relation_identification_political_debates_menini18.json"

    print_data_size(
        "Never Retreat, Never Retract: Argumentation Analysis for Political Speeches - Relation Identification",
        menini18_relation_identification
    )

    skitalinskaya23_claim_optimization_dev = tasks_path() / "claim_optimization_claim_revisions_skitalinskaya23" / "claim_optimization_claim_revisions_dev_skitalinskaya23.json"
    skitalinskaya23_claim_optimization_test = tasks_path() / "claim_optimization_claim_revisions_skitalinskaya23" / "claim_optimization_claim_revisions_test_skitalinskaya23.json"
    skitalinskaya23_claim_optimization_train = tasks_path() / "claim_optimization_claim_revisions_skitalinskaya23" / "claim_optimization_claim_revisions_train_skitalinskaya23.json"

    print_data_size(
        "Claim Optimization in Computational Argumentation - Claim Optimization",
        skitalinskaya23_claim_optimization_dev,
        skitalinskaya23_claim_optimization_test,
        skitalinskaya23_claim_optimization_train
    )

    stab17_argument_relation_identification = tasks_path() / "argument_unit_relation_identification_essays_stab17" / "argument_unit_relation_identification_essays_stab17.json"

    print_data_size(
        "Parsing Argumentation Structures in Persuasive Essays - Argument Relation Identification",
        stab17_argument_relation_identification
    )

    alshomary21_belief_based_argument_generation_train = tasks_path() / "conclusion_generation_belief_generation_alshomary21" / "conclusion_generation_belief_generation_train_alshomary21.json"
    alshomary21_belief_based_argument_generation_test = tasks_path() / "conclusion_generation_belief_generation_alshomary21" / "conclusion_generation_belief_generation_test_alshomary21.json"
    alshomary21_belief_based_argument_generation_valid = tasks_path() / "conclusion_generation_belief_generation_alshomary21" / "conclusion_generation_belief_generation_valid_alshomary21.json"

    print_data_size(
        "Belief-based Generation of Argumentative Claims",
        alshomary21_belief_based_argument_generation_train,
        alshomary21_belief_based_argument_generation_test,
        alshomary21_belief_based_argument_generation_valid
    )

    # convert_argument_ranking_ibm_evidence_quality_gleize19.py
    # convert_fallacy_detection_elecdeb60t020_goffredo23.py
    # convert_fallacy_detection_logic_jin22.py
    # convert_argument_ranking_ng20.py
