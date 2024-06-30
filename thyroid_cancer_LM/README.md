# project description
    The project code for the paper "Multi-region nomogram for predicting central lymph node metastasis in papillary thyroid carcinoma using multimodal imaging: A multicenter study" 

    Includes model training and results display
 
# environmental dependency
    numpy               1.24.3
	pandas              2.0.3
	torch               1.7.0
	torchvision         0.8.0
	scikit-learn        1.3.2
	pyradiomics         3.1.0

# directory structure
    root_path
    thyroid_cancer_LM
    ├── configs/                            // configuration files
    ├── dataset/                            // the dataset for each cohorts
    │   ├── fat
    │   ├── intratumor/
    │   ├── peritumor/
    │   └── merge_region/
    │       ├── external_test_cohort1
    │       ├── internal_test_cohort
    │       ├── primary_cohort
    ├── data                                // initial data
    ├── evaluation_indicators/              // model evaluation index
    ├── evaluation_during_trainning/        // inital evaluation
    ├── evaluation_visualization/           // result visualization
    ├── image_preprocessing_toolkit/        // image preprocessing
    ├── models/                             
    ├── runner/                             // model training
    │   ├── Model_Dict/
    │   ├── Runner_Utils/
    │   ├── Running_Dict/
    │   └── node_cla.py                     // training file
    ├── Utils/                              // project toolkit
    └── README.md                           // help file

# instructions
 	The model training file is located at ./runner/node_cla.py

    Parameters during the model training process are saved in ./runner/Running_Dict

    The best parameters stored in ./runner/Model_Dict

    The evaluation_visualization include important visualization code for the paper
 
 
 