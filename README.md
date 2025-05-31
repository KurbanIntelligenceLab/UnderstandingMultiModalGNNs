# Understanding Multi-Modal GNNs for Molecular Property Prediction

This repository provides a unified framework for benchmarking and understanding multi-modal graph neural networks (GNNs) for molecular property prediction. Our models integrate 3D molecular structure with textual and chemical context, leveraging both graph and language modalities to predict quantum chemical properties.

**Spotlight Paper at CVPR 2025 MM4Mat Workshop**

For full details, see: [Understanding the Capabilities of Molecular Graph Neural Networks in Materials Science Through Multimodal Learning and Physical Context Encoding](https://arxiv.org/abs/2505.12137) ([arXiv:2505.12137](https://arxiv.org/abs/2505.12137)) by Can Polat, Hasan Kurban, Erchin Serpedin, and Mustafa Kurban. Presented as a Spotlight Paper at the [Multimodal Learning for Materials Science (MM4Mat) Workshop, CVPR 2025](https://sites.google.com/view/mm4mat).

## Dataset

- **Main dataset:** [QM9 with PubChem Annotations and CLIP Embeddings](https://figshare.com/articles/dataset/QM9_with_PubChem_Annotations_as_in_Understanding_the_Capabilities_of_Molecular_Graph_Neural_Networks_in_Materials_Science_Through_Multimodal_Learning_and_Physical_Context_Encoding_/29203637?file=55004297)
- **Format:** PyTorch file (`data/data_v3_with_clip_768.pt`), where each molecule includes atomic structure and a CLIP embedding.

## Project Structure

```
.
├── data/                # Datasets (QM9 + CLIP embeddings)
├── models/              # Model definitions (SchNet, DimeNet++, Equiformer, FAENet, and multi-modal variants)
├── train_scripts/       # Training scripts for each model
├── requirements.txt     # Python dependencies
├── LICENSE              # License file
└── README.md            # Project documentation
```

## Models

Implemented models (in `models/`):
- **SchNet**
- **DimeNet++** 
- **Equiformer** 
- **FAENet** 

Each model has a `*Multi` variant (e.g., `SchNetMulti`, `DimeNetMulti`, etc.) that fuses molecular graph features with CLIP embeddings using gated or cross-attention fusion.

## Training & Evaluation

Training scripts are in `train_scripts/`:
- `train_modified_schnet.py`
- `train_modified_dimenet.py`
- `train_modified_equiformer.py`
- `train_modified_faenet.py`

All scripts:
- Load the QM9+CLIP dataset
- Perform 3-fold cross-validation for each target property
- Log metrics to TensorBoard and text files
- Save best model checkpoints and error logs per fold

### Example Usage

To train a multi-modal SchNet model:
```bash
python train_scripts/train_modified_schnet.py
```

Replace with the appropriate script for other models.

## Setup & Dependencies

- Python 3.12+
- PyTorch
- PyTorch Geometric
- scikit-learn
- tensorboard
- numpy
- e3nn

Install dependencies:
```bash
pip install -r requirements.txt
```

## Results & Logging

- Training, validation, and test losses are logged to TensorBoard (`logs/`) and text files (`runs/`).
- Best model checkpoints are saved per fold in `runs/<model_name>/target_<target_index>/fold_<fold_number>/`.

## References
- SchNet: [Schütt et al., 2017](https://arxiv.org/abs/1706.08566)
- DimeNet++: [Klicpera et al., 2020b](https://arxiv.org/abs/2011.14115)
- Equiformer: [Liao et al., 2023](https://arxiv.org/abs/2206.11990)
- FAENet: [Duval et al., 2023](https://arxiv.org/abs/2305.05577)

## Dataset Citation
If you use/utilize the dataset, please cite:

```bibtex
@article{polat2025understanding,
  title={Understanding the Capabilities of Molecular Graph Neural Networks in Materials Science Through Multimodal Learning and Physical Context Encoding},
  author={Polat, Can and Kurban, Hasan and Serpedin, Erchin and Kurban, Mustafa},
  journal={arXiv preprint arXiv:2505.12137},
  year={2025}
}
```

## Contact
For questions or contributions, please open an issue or pull request.