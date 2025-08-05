# HEA-Gibbs-ML-LLM

Hybrid machine learning and large language model (LLM) framework for predicting Gibbs free energy in high-entropy alloys (HEAs) using Pair Distribution Function (PDF) data, compositional features, and fine-tuned GPT models. This repository supports the research paper "[Advancements in Gibbs Energy Prediction: A Machine Learning Approach with High Entropy Alloys]" by providing reproducible code for conventional ML baselines, transformer-based regression, LLM embedding generation, and fine-tuning/response retrieval. Achieves test R² of 0.9635 and MAE of 0.0332 with the hybrid model.

## 📄 Content
This project integrates PCA-reduced PDF data, GPT-4o embeddings, and transformer architectures to predict thermodynamic stability in FeCoNiCuZn HEAs for nitrate adsorption. Key components:
- **Conventional ML**: Random Forest, Gradient Boosting, SVR, Linear Regression baselines.
- **Transformer Model**: Combines PDF and LLM embeddings for regression.
- **LLM Fine-Tuning**: Fine-tunes GPT-4.1-nano for interpretable Gibbs insights.
- **Notebooks**: Data preprocessing, embedding generation, and model training/evaluation.

Results highlight the performance of the hybrid model, with t-SNE visualizations showing enhanced clustering of stable alloys. Code is versioned at https://github.com/huynhtnhut97/HEA-Gibbs-ML-LLM for reproducibility.

## 🏗️ Model Architecture
The core model is a TransformerRegressor (implemented in PyTorch):
- **Inputs**: PCA-reduced PDF features (50 components from 2980 g(r) values) and GPT-4o embeddings (1536 dimensions).
- **Architecture**:
  - PCA embedding: Linear layer (1 -> 64) for each PCA component.
  - Embedding projection: Linear (1536 -> 64).
  - Concatenation: [Projected embedding + PCA tokens] with positional encoding.
  - Transformer Encoder: 2 layers, 8 heads, d_model=64.
  - Regression Head: Linear (64 -> 1) on the first token output.
- **Training**: MSE loss, Adam optimizer, 200 epochs, batch size 32.
- **Classes**:
  - **GPTFineTuner**: Manages dataset formatting (JSONL), upload, fine-tuning job creation, and evaluation (MAE on holdout).
  - **LLMresponseder**: Retrieves responses from fine-tuned models/checkpoints, parsing JSON for metrics/IDs.

See code comments in `.py` files or notebooks for implementation details.

## 💾 Dataset Details
- **Source**: Custom FeCoNiCuZn HEA dataset (1268 microstructures) from DFT calculations.
- **Features**:
  - Compositional: Fe, Co, Ni, Cu, Zn ratios (0-1).
  - Active sites: Two categorical variables (e.g., "Co", "Fe").
  - PDF: 2980 g(r) values (0.2-30 Å, 0.01 Å steps); PCA-reduced to 50 components (~95% variance).
  - Embeddings: GPT-4o-generated from prompts (e.g., "Alloy composition: Fe=0.2, Co=0.2... Active sites: Co and Fe"), 1536 dimensions.
  - Target: Gibbs free energy of nitrate adsorption (eV, range ~[-0.6, 0.6]).
- **Preprocessing**: StandardScaler on PDF features, tiktoken for prompt token limits (max 8000).
- **Generation**: Run `CI-2025-01838n_LLM_embedding.ipynb` to create `HEA_Dataset_with_embeddings.csv`.
- **Note**: Full dataset not included due to size/privacy; structure described in notebooks. Contact for access.
- **Download**: The dataset is available for download at [link](https://example.com/HEA_Dataset.zip). Contact the author if access issues arise.

## 🔧 Installation
To set up and run the experiments:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/huynhtnhut97/HEA-Gibbs-ML-LLM.git
   cd HEA-Gibbs-ML-LLM
   ```

2. **Create a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Requires Python 3.9+ and the following libraries:
   ```bash
   pip install pandas numpy torch scikit-learn openai tiktoken matplotlib seaborn
   ```
   - **GPU Support**: Install PyTorch with CUDA if available (see pytorch.org).
   - **OpenAI API**: Set key as environment variable:
     ```bash
     export OPENAI_API_KEY='your-key-here'  # On Windows: set OPENAI_API_KEY=your-key-here
     ```
   - Create a requirements.txt for reproducibility:
     ```bash
     pip freeze > requirements.txt
     ```

4. **Dataset Setup**:
   - Place `HEA_Dataset.csv` in `data/` folder (not included due to size; structure includes ID, prompt, n_tokens, active_site_1, active_site_2, Fe, Co, Ni, Cu, Zn, Gibbs, and 2980 g(r) columns).
   - Run `CI-2025-01838n_LLM_embedding.ipynb` to generate `HEA_Dataset_with_embeddings.csv`.

5. **Git Authentication** (if pushing changes):
   - Generate a Personal Access Token (PAT) on GitHub:
     - Go to Settings > Developer settings > Personal access tokens > Tokens (classic).
     - Create token with repo scope, copy (e.g., ghp_abc123).
     - Use PAT as password for `git push`.
   - Alternative: Set up SSH (generate key with `ssh-keygen -t ed25519`, add to GitHub, change remote to `git@github.com:huynhtnhut97/HEA-Gibbs-ML-LLM.git`).

## 🚀 Usage
Run experiments via notebooks or Python classes:

1. **Notebooks**:
   - Start Jupyter: `jupyter notebook`.
   - `CI-2025-01838n_Methodologies.ipynb`: Preprocesses PDF data (PCA to 50 components), trains/tests conventional ML and transformer models, saves best model (e.g., model_20250226_141802_loss_0.0014.pth).
   - `CI-2025-01838n_LLM_embedding.ipynb`: Generates GPT-4o embeddings from compositional prompts, saves enhanced dataset.

2. **Python Classes** (save as `gpt_fine_tuner.py` and `llm_responseder.py`):
   - **GPTFineTuner**: Manages LLM fine-tuning workflow (dataset creation, upload, job creation, evaluation).
     ```python
     from gpt_fine_tuner import GPTFineTuner

     tuner = GPTFineTuner(api_key='your-key', dataset_path='data/HEA_Dataset_with_embeddings.csv', num_examples=50)
     mae = tuner.run_all()  # Executes fine-tuning, returns MAE on holdout
     ```
   - **LLMresponseder**: Retrieves responses from fine-tuned models or checkpoints.
     ```python
     from llm_responseder import LLMresponseder

     responder = LLMresponseder(api_key='your-key', job_id='ftjob-abc123', checkpoint_id='ftckpt_zc4Q7MP6XxulcVzj4MZdwsAB')
     response = responder.get_response("Predict Gibbs for HEA with Fe=0.3, explain stability.")
     print(response)
     ```

3. **Training the Transformer**:
   - Run Methodologies notebook to train on PCA-reduced PDF + embeddings.
   - Outputs saved in `models/` (e.g., best model with test loss 0.0014).

## 🧪 Experiments and Notebooks
- **CI-2025-01838n_Methodologies.ipynb**:
  - Loads dataset, applies PCA (2980 g(r) features to 50 components, ~95% variance).
  - Trains conventional models (Random Forest: R² 0.9023, Gradient Boosting: R² 0.9074).
  - Trains transformer (200 epochs, achieves R² 0.9635, MAE 0.0332).
  - Tests: Reproduce by running cells; adjust epochs (default 200) or PCA components (default 50).
- **CI-2025-01838n_LLM_embedding.ipynb**:
  - Generates prompts (e.g., "Alloy composition: Fe=0.2, Co=0.2..."), computes GPT-4o embeddings.
  - Saves dataset with embeddings for downstream modeling.
  - Tests: Run with subset (10 samples) to verify token limits (max 8000).
- Reproduce: Ensure data/ exists, run notebooks sequentially, set API key.
- Experiments: Test PCA components (e.g., m=100), fine-tuning examples (50-100), or checkpoint selection.

## 📚 Classes
- **GPTFineTuner** (gpt_fine_tuner.py): Automates fine-tuning:
  - Builds JSONL from dataset (prompts + Gibbs responses).
  - Uploads to OpenAI, creates job, evaluates (MAE on holdout).
  - Example: Fine-tunes gpt-4.1-nano-2025-04-14 with 50 examples.
- **LLMresponseder** (llm_responseder.py): Retrieves responses:
  - Parses checkpoint JSON (e.g., id: ftckpt_zc4Q7MP6XxulcVzj4MZdwsAB, metrics: valid_loss 0.134).
  - Queries fine-tuned models/checkpoints for stability insights.
  - Example: "Gibbs -0.3 eV, high stability due to Fe."

See code comments for detailed usage.

## 🔗 Citation
This repo supports "[Advancements in Gibbs Energy Prediction: A Machine Learning Approach with High Entropy Alloys]" (under review). Cite as:
```bibtex
@article{huynh2025hea,
  title={Advancements in Gibbs Energy Prediction: A Machine Learning Approach with High Entropy Alloys},
  author={Huynh, Nhut and He, Xiang and Nguyen, Kim-Doang},
  journal={Journal of Chemical Information and Modeling},
  year={2025}
}
```

For questions, contact huynhtnhut97@gmail.com or open an issue.
