# stuttering_detector
A SPAN project using the rtMRI stuttering dataset for temporal segmentation for three categories: silence, fluent speech, and disfluent speech. Designed to be run on Redondo.

Execute multiclass stuttering classification with the following commands:
python generate_multiclass_dataset.py
python multiclass.py --modality video --batch_size 16