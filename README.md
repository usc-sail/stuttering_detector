# stuttering_detector
A SPAN project using the rtMRI stuttering dataset for temporal segmentation for three categories: silence, fluent speech, and disfluent speech. Designed to be run on Redondo.

Execute multiclass stuttering classification with the following commands:
python generate_multiclass_dataset.py
python multiclass.py --modality video --batch_size 16

Execute binary stuttering detection with the following commands:
python generate_binary_dataset.py
python binary.py --modality both --batch_size 16

Execute ablation study on binary stuttering detection with the following commands:
python generate_binary_dataset.py
python ablation.py --batch_size 16 --articulator TR