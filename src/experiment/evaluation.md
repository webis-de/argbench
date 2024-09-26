# Evaluating Leave One Out Experiments

Leave one out config prototype: [config_leave_one_out.json](configs/config_leave_one_out.json)

| Left Dataset                                                                    | F1     | Precision | Recall |
|---------------------------------------------------------------------------------|--------|-----------|--------|
| [IBMSC](configs/config_leave_one_out_barhaim17.json)                            | 0.441  | 0.48     | 0.483  |
| [Key Point Analysis](configs/config_leave_one_out_barhaim21.json)               | 0.3778 | 0.631     | 0.603  |
| [Premise Detection](config_leave_one_out_eindor20.json)                         | 0.114  | 0.302     | 0.428  |
| [IBM-RANK-30K](configs/config_leave_one_out_gretz20.json)                       | 0.445  | 0.470     | 0.48   |
| [Argument Similarity (UKP-Aspect)](configs/config_leave_one_out_reimers19.json) | 0.045  | 0.025     | 0.25   |
| [ArgU](configs/config_leave_one_out_saha23.json)                                |        |           |        |


| [Counter Argument Generation](configs/config_leave_one_out_hua18.json) |        |           |        |
