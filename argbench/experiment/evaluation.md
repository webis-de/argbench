# Evaluating Leave One Out Experiments

Leave one out config prototype: [config_leave_one_out.json](configs/config_leave_one_out.json)

| Left Dataset                                                                          | F1     | Precision | Recall | Majority F1 | Random F1 |
|---------------------------------------------------------------------------------------|--------|-----------|--------|-------------|-----------|
| [IBMSC](configs/archive/config_leave_one_out_barhaim17.json)                                  | 0.441  | 0.48      | 0.483  | 0.35        | 0.505     |
| [Key Point Analysis](configs/archive/config_leave_one_out_barhaim21.json)                     | 0.3778 | 0.631     | 0.603  | 0.444       | 0.453     |
| [Premise Detection](configs/archive/config_leave_one_out_eindor20.json)                       | 0.123  | 0.07      | 0.5    | 0.457       | 0.434     |
| [IBM-RANK-30K](configs/archive/config_leave_one_out_gretz20.json)                             | 0.516  | 0.557     | 0.543  | 0.336       | 0.494     |
| [Argument Similarity (UKP-Aspect)](configs/archive/config_leave_one_out_reimers19.json)       | 0.054  | 0.03      | 0.25   | 0.162       | 0.241     |
| [ArgU](configs/archive/config_leave_one_out_saha23.json)                                      | 0.023  | 0.014     | 0.061  | 0.1         | 0.11      |
| [ArgMin](configs/archive/config_leave_one_out_stab17.json)                                    | 0      | 0         | 0      | 0           | 0         |
| [Dagstuhl-15512-ArgQuality-Corpus-v2](configs/archive/config_leave_one_out_wachsmuth17.json)  | 0.24   | 0.303     | 0.243  | 0.21        | 0.321     |
| [Aspect-controlled argument generation](configs/archive/config_leave_one_out_schiller21.json) | 0.0068 | 0.0068    | 0.0068 | 0           | 0         |
| [Argument Frame Identification](configs/archive/config_leave_one_out_ajjour20.json)           | 0.014  | 0.021     | 0.011  | 0.3352      | 0.502     |


