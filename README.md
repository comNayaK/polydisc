# Polymer Discovery AI (PolyLab) 🧪🌿

> **Generative AI & Machine Learning Platform for Sustainable Polymer Discovery**

Polymer Discovery AI is a full-stack research platform designed to accelerate the discovery and design of green, sustainable polymer candidates. It combines generative deep learning (GRU-based Variational Autoencoders) with multi-model predictive ensembles (Graph Neural Networks, XGBoost, Random Forests) and an LLM-powered chemistry copilot to generate, score, compare, and report novel polymers against comprehensive sustainability criteria.

---

## 🌟 Key Features

- **Generative Molecular Design (VAE)**: Character-level GRU Variational Autoencoder capable of sampling novel valid polymer SMILES strings with controllable sampling temperature.
- **7-Property Predictive Ensemble**:
  - 🌿 **Biodegradability**
  - ⚙️ **Synthesizability (SA Score)**
  - ☠️ **Non-Toxicity (Tox21 / ToxCast)**
  - ♻️ **Recyclability**
  - 💪 **Mechanical Strength**
  - 🌡️ **Glass Transition Temperature ($T_g$)**
  - 💧 **Solubility ($\log S$)**
- **Unified Green Score**: Multi-criteria weighted scoring algorithm allowing researchers to prioritize target performance metrics.
- **Latent Space Interpolation**: Morph smoothly between two known polymers through the continuous latent representation to discover intermediate candidates.
- **PolyLab AI Copilot**: Integrated Groq LLM (`llama-3.3-70b-versatile`) assistant that recommends hyperparameter configurations, interprets chemical properties, and injects run configurations directly into the discovery workspace.
- **Interactive Web Interface**: Modern, responsive React + Vite dashboard with real-time property radar charts, PCA projections, candidate filtering, and PDF/CSV export.

---

## 🏗️ Architecture Overview

```
new_polymer/
├── api/                   # FastAPI application & REST route definitions
│   ├── routes.py          # API endpoints (/discovery, /compare, /interpolate, etc.)
│   ├── schemas.py         # Pydantic data schemas
│   └── ai_assistant.py    # Groq LLaMA-3.3 integration
├── data/                  # Dataset loaders & preprocessing pipelines
│   └── loader.py          # Automated acquisition of PI1M, Tox21, TSUDALAB datasets
├── features/              # Chemical featurization (Morgan fingerprints, RDKit descriptors)
├── generation/            # Generative models & sampling controllers
├── models/                # GNN, VAE, XGBoost & RF predictors
├── reports/               # Automated multi-page PDF generation (FPDF2)
├── scoring/               # Multi-objective Green Score calculations
├── ui/                    # Visualizations & legacy UI components
├── frontend/              # Modern React + Vite frontend application
├── app.py                 # Backend server entrypoint
├── config.py              # Central hyperparameter & pipeline configuration
├── requirements.txt       # Python dependencies
└── DEPLOY.sh              # Production systemd & Nginx deployment guide
```

---

## 🚀 Quick Start

### 1. Prerequisites

- **Python**: 3.10 or higher
- **Node.js**: 18.0 or higher & `npm`
- **RDKit**: Required for molecular manipulation (included in requirements or installable via Conda)
- **Groq API Key**: (Optional, for AI Assistant) Obtain a free key at [Groq Console](https://console.groq.com/)

---

### 2. Backend Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/polymer-discovery-ai.git
   cd polymer-discovery-ai
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Copy `.env.example` to `.env` and insert your Groq API key:
   ```bash
   cp .env.example .env
   ```
   Edit `.env`:
   ```env
   GROQ_API_KEY=gsk_your_actual_groq_key_here
   ```

5. **Start the Backend Server**:
   ```bash
   python app.py
   ```
   *Note: On the first run, the system automatically fetches required training datasets and initializes model weights in `saved_models/`.*
   The REST API will be live at `http://localhost:8000` (Interactive Swagger docs available at `http://localhost:8000/docs`).

---

### 3. Frontend Setup

1. **Navigate to the frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install Node modules**:
   ```bash
   npm install
   ```

3. **Run the Development Server**:
   ```bash
   npm run dev
   ```
   The web dashboard will be available at `http://localhost:5173`.

---

## 📊 Configuration

All global model configurations, network dimensions, and training flags can be adjusted in [config.py](file:///c:/Users/T.KAVYA%20REDDY/OneDrive/Desktop/new_polymer/config.py):
- `DEFAULT_WEIGHTS`: Customize initial property weightings for Green Score calculation.
- `FORCE_RETRAIN_*`: Toggle retrain flags (`FORCE_RETRAIN_PREDICTORS`, `FORCE_RETRAIN_VAE`, `FORCE_RETRAIN_GNN`) when fine-tuning on custom datasets.
- `VAE_*` and `GNN_*`: Adjust neural network architectures, latent dimensions, and batch sizes.

---

## 🛠️ Production Deployment

For continuous hosting on Linux (systemd + Nginx reverse proxy), refer to the step-by-step instructions provided in [DEPLOY.sh](file:///c:/Users/T.KAVYA%20REDDY/OneDrive/Desktop/new_polymer/DEPLOY.sh).

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
