

# Multi-Modal Graph Construction and Classification for Autism Spectrum Disorder Using Stacked GCNs

## Overview

This repository contains the implementation of a multi-modal graph-based deep learning framework designed to enhance the diagnosis of Autism Spectrum Disorder (ASD). The proposed methodology utilizes functional MRI (fMRI) data, phenotypic attributes (e.g., gender, site), and advanced Graph Convolutional Network (GCN) architectures—specifically, a dual-stream model that integrates residual DeepGCNs with stacked GCNs employing DeepWalk embeddings.

Through recursive feature elimination, adaptive edge pruning, and population graph construction, the framework effectively models complex topological and hierarchical relationships within the brain. Evaluations on the ABIDE dataset demonstrate significant improvements in classification performance over existing baseline models.

---

## Key Contributions

- **Hybrid Dual-Stream Architecture:** Integration of DeepGCNs and Stacked GCNs captures both raw functional connectivity and population-level embeddings.
- **Multimodal Graph Construction:** Combines fMRI-derived functional connectivity with demographic similarities (age, gender, site) using k-NN and cosine similarity.
- **DeepWalk Embedding Integration:** Employs random walk-based skip-gram embeddings to encode subject-level similarity for global graph representation.
- **Adaptive Edge Pruning:** Dynamically eliminates weak graph connections to enhance generalizability and reduce overfitting.
- **Benchmarking on ABIDE:** Achieves an accuracy of 81.29% and AUC of 0.85 using nested 10-fold cross-validation on AAL and HO atlas-based connectivity matrices.

---

![WhatsApp Image 2025-05-14 at 11 17 40_5bc8b2eb](https://github.com/user-attachments/assets/d6daf4cb-f839-42fa-b083-e3316efeaca7)



---

## Dataset

- **Source**: [ABIDE I Dataset](https://fcon1000.projects.nitrc.org/indi/abide/)
- **Modalities**:
  - **Functional MRI**: Resting-state scans, parcellated using AAL and HO atlases.
  - **Phenotypic**: Age, gender, acquisition site.

---

## Setup Instructions

### A. Using Google Colab

1. Upload project files and ABIDE subset.
2. Install dependencies:
   ```bash
   !pip install torch torchvision torchaudio torch-geometric networkx
   !pip install pandas numpy scikit-learn matplotlib seaborn


