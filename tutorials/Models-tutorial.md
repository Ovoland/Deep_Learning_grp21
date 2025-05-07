# Accessing models from models the scratch folder 

This file contains instructions on how to access the models from the shared folder. A model is provided under its respective license. By using a model, you agree to comply with the terms of that license, including conditions related to reproduction, distribution, and modification, as well as disclaimers of warranties and liabilities. All additional models will be added to `huggingface_models_v2/` folder. 

**NOTE: A model below is simply an example. You're encouraged to select the model that best suits your project needs. Feel free to use any other available models, provided their licenses permit your intended use. If due to the size constraints of your home repository you would like us to add more models to the shared space, please, get in touch.**

## Phi4-mini-Instruct
This model is available under the MIT license (see the license and additional documents in licenses folder). 

Instructions: 

**Step 1**: copy the `phi4_mini_example.py` to your home folder. 

**Step 2**: in your home folder, create an `.env` file - this is the file which will contain the environmental variables, such as a path the model and it's weights. 

**Step 3**: in the `.env` file, specify the path to the model. 

Option 1 (**recommended**): `PHI4_PATH_SCRATCH=/scratch/models/huggingface_models_v2/Phi-4-mini-instruct`

Option 2: `PHI4_PATH_SCRATCH=/scratch/models/huggingface/models--microsoft--Phi-4-mini-instruct/snapshots/<snapshot_hash>`

Example of the content of the `.env` file (if using Option 1): 

```
PHI4_PATH_SCRATCH=/scratch/models/huggingface_models_v2/Phi-4-mini-instruct
```

Example of the content of the `.env` file (if using Option 2): 

```
PHI4_PATH_SCRATCH=/scratch/models/huggingface/models--microsoft--Phi-4-mini-instruct/snapshots/c0fb9e74abda11b496b7907a9c6c9009a7a0488f
```

**Step 4**: run the python file, while mouting both your home folder, and the scratch folder. Example of the command below: 

Example of the command to run the code: 

```
runai submit --image <your-image> --gpu 1 --pvc course-ee-559-scratch:/scratch --pvc home:/pvc/home --command -- python3 /pvc/home/phi4_mini_example.py --results_path /pvc/home/
```