# YOLO-Based Person Detection, Tracking & Re-Identification

A computer vision project that combines YOLO-based person detection, multi-object tracking, and person re-identification (ReID) to recognize and track enrolled individuals across video frames.

## Overview

This project is designed to detect people through a webcam or video stream, assign tracking identities, and compare detected individuals against an enrolled identity gallery.

It combines person detection, appearance-based re-identification, face recognition features, and tracking logic to improve identity consistency when people move, cross paths, or become temporarily occluded.

## Features

- **Person Detection:** Detects people using a YOLO model.
- **Multi-Object Tracking:** Tracks multiple people across video frames.
- **Person Re-Identification:** Uses appearance embeddings to compare detected people with enrolled identities.
- **Face-Based Identification:** Supports face-based identity matching.
- **Body Enrollment:** Captures body images for identity gallery creation.
- **Face Enrollment:** Captures face images for identity gallery creation.
- **Occlusion Handling:** Includes tracking and recovery logic for temporary occlusions.
- **Identity Management:** Distinguishes enrolled individuals from unidentified people.
- **Metrics and Analytics:** Tracks presence-related information and detection events.
- **Dashboard:** Includes a Streamlit interface for project monitoring and analytics.
- **Database Integration:** Supports MySQL configuration for storing application data.

## Project Structure

```text
reid_tracker_v08/
â”‚
â”œâ”€â”€ app/
â”‚   â”œâ”€â”€ core/             # Detection, tracking and ReID pipeline
â”‚   â”œâ”€â”€ dashboard/        # Streamlit dashboard components
â”‚   â”œâ”€â”€ scripts/          # Enrollment and gallery-building scripts
â”‚   â”œâ”€â”€ storage/          # Database integration
â”‚   â””â”€â”€ utils/             # Utility functions
â”‚
â”œâ”€â”€ tests/                 # Project tests
â”œâ”€â”€ Models/                # Local model weights (not included in Git)
â”œâ”€â”€ known_people/          # Local enrollment data (not included in Git)
â”œâ”€â”€ output/                # Generated outputs (not included in Git)
â”‚
â”œâ”€â”€ enrollment_pipeline.py
â”œâ”€â”€ run_pipeline.py
â”œâ”€â”€ streamlit_app.py
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ requirements-pipeline.txt
â”œâ”€â”€ .env.example
â””â”€â”€ README.md
```

## Technology Stack

- Python
- OpenCV
- YOLO / Ultralytics
- PyTorch
- TorchReID / OSNet
- InsightFace
- ONNX Runtime
- NumPy
- SciPy
- Streamlit
- MySQL

## Requirements

- Python 3.11 (recommended for the current development environment)
- Webcam or compatible video source
- NVIDIA GPU with compatible CUDA dependencies (optional; CPU execution may be slower)
- Git

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/shubhamgote1128/yolo-person-reid-tracker.git
cd yolo-person-reid-tracker
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements-pipeline.txt
```

If you are setting up the dashboard or other optional components, also review `requirements.txt`.

### 4. Configure environment variables

Copy `.env.example` to `.env` and update the required values.

Never commit your actual `.env` file or real database credentials.

### 5. Add model weights

The model weights are not included in this repository. Obtain the required weights from their respective authorized sources and place them in the expected `Models/` directory.

Check the project configuration and documentation for the expected model filenames.

## Usage

### 1. Enroll a person

Run:

```powershell
python -m app.scripts.enroll
```

Enter the person's name when prompted.

Choose an enrollment mode:

- `body` â€” capture body images.
- `face` â€” capture face images.

Only enroll individuals who have given informed consent.

### 2. Build the identity gallery

Run:

```powershell
python enrollment_pipeline.py --build-only
```

This processes the enrolled images and prepares the identity gallery.

### 3. Run the detection and tracking pipeline

Run:

```powershell
python run_pipeline.py
```

The application starts the webcam pipeline using the configured video source.

Press `q` to stop webcam processing.

### 4. Run the Streamlit dashboard

Run:

```powershell
streamlit run streamlit_app.py
```

Configure the database and dashboard settings before using database-dependent features.

## Configuration

The project provides example configuration files:

- `.env.example`
- `app/.env.example`
- `.streamlit/secrets.toml.example`
- `app/ca.pem.example`

Copy and configure only the files required for your environment.

Do not upload actual secrets, private certificates, passwords, or personal enrollment data.

## Privacy and Security

This project may process biometric information, including face images and body appearance data.

- Obtain informed consent before enrolling anyone.
- Keep enrollment images and generated biometric data private.
- Do not publish personal images or identity galleries.
- Never commit passwords, API keys, database credentials, or private certificates.
- Follow applicable privacy and data-protection laws.


