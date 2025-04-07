# dtMol - Diffusion Transformer for Molecular Docking

Model and data files can be obtained upon request.

### Installation
```bash
pip install -e setup.py 
```

### Training
```bash
torchrun dtmol/dtmol_train.py --batch_size 4 --world_size 2 --tr_sde VE --tr_sigma_max 4. --model_name e3nn --pert_mole_sigma_max 2. --warm 1 --share_attention --record_intermediate ~/scratch/dtmol/test/
```

### Hardware Requirements
We trained the model using 4XA100 GPUs, GPU with >20 GB memory is required to host the model.
