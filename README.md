# UnifiedPDESolvers

Original PyTorch implementation of UPS proposed in the paper "[UPS: Towards Building Foundation Models for PDE Solving via Cross-Modal Adaptation](https://arxiv.org/abs/2403.07187)". UPS is developed for solving diverse spatiotemporal PDEs defined over various domains, dimensions, and resolutions. It unifies different PDEs into a consistent representation space and processes diverse collections of PDE data using a unified network architecture that combines LLMs with domain-specific neural operators.

## Requirements
```
pip install -r requirements.txt
```
Note that the `attrdict` package might not be compatible for python 3.10 or newer versions. If getting `ImportError: cannot import name 'Mapping' from 'collections'`, change 
```
from collections import Mapping
```
to 
```
from collections.abc import Mapping
```

## Training models
1. Download [PDEBench datasets](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986) to `./datasets`
2. Generate the PDE metadata
```
python3 generate_text_embeddings.py
```
3. Generate the data files for data loading
```
python3 generate_data.py
```
4. Use an existing configuration file or add a new one to `./configs`
5. Run training
```
python3 main.py --config configs/config_file_name.yaml 
```
Model checkpoints will be released later.
