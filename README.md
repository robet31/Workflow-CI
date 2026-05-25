# Workflow-CI

Repository CI/CD Workflow untuk Proyek Akhir "Membangun Sistem Machine Learning" - Dicoding.

## Author
**Ar'raffi Abqori Nur Azizi**

## Deskripsi
Repository ini berisi MLflow Project dan GitHub Actions CI workflow untuk melakukan training model secara otomatis ketika trigger dipantik.

## Struktur Repository
```
Workflow-CI/
├── .github/workflows/
│   └── ci.yml                        # GitHub Actions CI workflow
└── MLProject/
    ├── MLProject                     # MLflow Project definition
    ├── modelling.py                  # Training script
    ├── conda.yaml                    # Conda environment
    ├── Tautan ke Docker Hub.txt      # Docker Hub link
    └── winequality_preprocessing/
        ├── train.csv                 # Preprocessed training data
        └── test.csv                  # Preprocessed test data
```

## CI Workflow Stages
1. ✅ Checkout repository
2. ✅ Setup Python 3.12.7
3. ✅ Install dependencies
4. ✅ Set MLflow Tracking URI
5. ✅ Run MLflow project (training)
6. ✅ Get latest MLflow run_id
7. ✅ Upload artifacts to GitHub
8. ✅ Build Docker Image (mlflow build-docker)
9. ✅ Push Docker Image to Docker Hub

## Docker Hub
Image: `ravnxx/wine-quality-model:latest`

## Cara Menjalankan (Lokal)
```bash
cd MLProject
python modelling.py 100 10
```

## Secrets yang Diperlukan
- `DOCKERHUB_USERNAME`: Username Docker Hub
- `DOCKERHUB_TOKEN`: Docker Hub Personal Access Token
