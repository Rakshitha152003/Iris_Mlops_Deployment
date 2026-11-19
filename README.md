# 🌸 End-to-End Iris MLOps Deployment

### Building, Automating, Deploying and Monitoring a Machine Learning Application

An end-to-end Machine Learning Operations (MLOps) project that demonstrates the complete machine learning lifecycle, from data ingestion and preprocessing to model training, experiment tracking, model evaluation, containerization, and deployment.

The project integrates DVC, MLflow, Scikit-learn, Streamlit, Docker, and GitHub Actions, with upcoming enhancements focused on production monitoring, data drift detection, and automated alerting using Prometheus, Grafana, and Evidently AI.

---

## 📌 1. Project Overview

Developing a machine learning model is only one part of a real-world ML solution. Making that model reproducible, manageable, deployable, and reliable in a production environment requires an organized MLOps workflow.

This project implements an end-to-end Iris flower classification system by integrating machine learning development with MLOps practices.

The application predicts the species of an Iris flower using four numerical measurements:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The project focuses on building a modular and reproducible machine learning pipeline, tracking experiments, selecting an appropriate model, and deploying the trained model through a Dockerized Streamlit application.

### 🎯 Project Objectives

- Develop an end-to-end machine learning pipeline.
- Automate data ingestion, preprocessing, and feature engineering.
- Manage reproducible ML workflows using DVC.
- Track machine learning experiments using MLflow.
- Perform hyperparameter optimization and model comparison.
- Implement automated champion model selection.
- Evaluate model performance and overfitting.
- Serialize and deploy the selected model.
- Build an interactive application using Streamlit.
- Containerize the application using Docker.
- Integrate CI/CD automation using GitHub Actions.
- Extend the project with production monitoring and data drift detection.

---

## 🛠️ 2. Technology Stack

| Category                    | Technologies                      |
| --------------------------- | --------------------------------- |
| Programming Language        | Python 3.11                       |
| Data Manipulation           | Pandas, NumPy                     |
| Machine Learning            | Scikit-learn                      |
| Scientific Computing        | SciPy                             |
| Experiment Tracking         | MLflow                            |
| Data Versioning             | DVC                               |
| Hyperparameter Optimization | Optuna-based search configuration |
| Model Serialization         | Joblib                            |
| Application Framework       | Streamlit                         |
| Containerization            | Docker                            |
| Version Control             | Git, GitHub                       |
| CI/CD                       | GitHub Actions                    |

### Upcoming Technologies

| Category                | Technologies     |
| ----------------------- | ---------------- |
| Metrics Collection      | Prometheus       |
| Monitoring Dashboards   | Grafana          |
| Data Drift Detection    | Evidently AI     |
| Automated Alerting      | Grafana Alerting |
| Container Orchestration | Docker Compose   |

---

## 📊 3. Dataset Description

The project uses the Iris flower classification dataset.

The dataset contains four numerical input features and one target variable.

### Input Features

| Feature       | Description                 |
| ------------- | --------------------------- |
| SepalLengthCm | Sepal length in centimeters |
| SepalWidthCm  | Sepal width in centimeters  |
| PetalLengthCm | Petal length in centimeters |
| PetalWidthCm  | Petal width in centimeters  |

### Target Variable

`Species`

The target variable contains three Iris flower classes:

1. Iris-setosa
2. Iris-versicolor
3. Iris-virginica

The objective is to predict the correct flower species based on its measurements.

---

## 🏗️ 4. Project Architecture

The project follows a modular machine learning pipeline.

```text
                 RAW IRIS DATASET
                        |
                        v
                  DATA INGESTION
                        |
                        v
                 DATA PREPROCESSING
                        |
                        v
                 FEATURE ENGINEERING
                        |
                        v
                  MODEL BUILDING
                        |
                        v
                 MODEL COMPARISON
                        |
                        v
                 MODEL SELECTION
                        |
                        v
                 MODEL EVALUATION
                        |
                        v
                 MODEL SERIALIZATION
                        |
                        v
                    MODEL.PKL
                        |
                        v
                 STREAMLIT APPLICATION
                        |
                        v
                   DOCKER IMAGE
                        |
                        v
                 DOCKER CONTAINER
                        |
                        v
                 DEPLOYED APPLICATION
                        |
                        v
             FUTURE MONITORING LAYER
                        |
             +----------+----------+
             |          |          |
             v          v          v
         PROMETHEUS  EVIDENTLY   GRAFANA
             |          |          |
             +----------+----------+
                        |
                        v
                 AUTOMATED ALERT\
```

---

## 🔄 5. Machine Learning Pipeline

DVC manages the machine learning pipeline through `dvc.yaml`.

The pipeline contains the following stages:

### Stage 1: Data Ingestion

File: `src/data_ingestion.py`

Responsibilities:

- Load the raw Iris dataset.
- Validate the required input data.
- Perform the initial data ingestion.
- Generate the ingested dataset.

Output:

`data/ingested/iris.csv`

### Stage 2: Data Preprocessing

File: `src/data_preprocessing.py`

Responsibilities:

- Load the ingested dataset.
- Prepare the data for model development.
- Split the dataset into training and testing sets.
- Use configurable test size and random state parameters.

Outputs:

- `data/processed/train.csv`
- `data/processed/test.csv`

### Stage 3: Feature Engineering

File: `src/feature_engineering.py`

Responsibilities:

- Select the four Iris measurement features.
- Separate independent and dependent variables.
- Generate training and testing feature datasets.
- Prepare the data for model training.

Outputs:

- `X_train.csv`
- `X_test.csv`
- `y_train.csv`
- `y_test.csv`

### Stage 4: Model Building

File: `src/model_building.py`

Responsibilities:

- Train and compare candidate machine learning models.
- Perform cross-validation.
- Apply hyperparameter optimization.
- Measure model performance.
- Evaluate overfitting.
- Select the champion model.
- Generate model comparison reports.

The selection strategy considers:

- Validation accuracy.
- Training and validation accuracy gap.
- Model size.
- Training time.
- Prediction time.

Outputs:

- Trained model artifacts.
- `reports/model_selection.json`
- `reports/model_comparison.csv`
- `reports/model_comparison.md`

### Stage 5: Model Evaluation

File: `src/model_evaluation.py`

Responsibilities:

- Evaluate the selected model using test data.
- Generate model evaluation metrics.
- Produce evaluation reports.
- Record the final model evaluation results.

---

## 📈 6. Automated Champion Model Selection

The project implements a multi-criteria model selection strategy.

Rather than selecting a model using accuracy alone, the pipeline considers predictive performance and deployment efficiency.

### Model Selection Criteria

| Criterion           | Purpose                         |
| ------------------- | ------------------------------- |
| Validation Accuracy | Measures predictive performance |
| Accuracy Gap        | Helps identify overfitting      |
| Model Size          | Evaluates storage efficiency    |
| Training Time       | Measures training efficiency    |
| Prediction Time     | Evaluates inference efficiency  |

The selection strategy is configurable through `params.yaml`.

Important configuration parameters include:

- `max_accuracy_gap_pct`
- `accuracy_tolerance_pct`
- `model_size_mb`
- `training_time_seconds`
- `prediction_time_ms`
- `fallback_if_none_eligible`

This approach helps select a suitable champion model while considering both model quality and deployment requirements.

---

## 📊 7. Experiment Tracking – MLflow

MLflow is used as part of the machine learning experiment management workflow.

The project configuration includes the experiment name:

`Iris_DVC_Pipeline`

Experiment tracking helps organize model development and compare experimental results.

The MLflow tracking directory can be maintained locally through:

`mlruns/`

---

## 🔗 8. Data Version Control – DVC

DVC is used to manage and reproduce the machine learning pipeline.

The pipeline is defined in:

`dvc.yaml`

Pipeline configuration is maintained in:

`params.yaml`

The generated pipeline state is recorded in:

`dvc.lock`

### Important DVC Commands

Initialize DVC:

```bash
dvc init
```

Execute the complete pipeline:

```bash
dvc repro
```

Check pipeline status:

```bash
dvc status
```

View metrics:

```bash
dvc metrics show
```

Visualize the pipeline:

```bash
dvc dag
```

DVC helps maintain reproducibility by tracking dependencies, parameters, and pipeline outputs.

---

## 🌐 9. Streamlit Application

The project includes an interactive web application developed using Streamlit.

Application file:

`app.py`

### Application Features

- Accepts four Iris flower measurements.
- Loads the trained champion model.
- Predicts the Iris flower species.
- Displays the prediction result.
- Displays class probabilities when supported by the selected model.

### Run the Application Locally

```bash
streamlit run app.py
```

Open the application:

```text
http://localhost:8501
```

---

## 🐳 10. Docker Deployment

Docker is used to containerize the Streamlit application and its trained machine learning model.

The Dockerfile uses Python 3.11 slim as the base image.

### Step 1: Build the Machine Learning Pipeline

Before creating the Docker image, ensure that the trained model exists.

```bash
dvc repro
```

Verify that:

`models/model.pkl`

has been generated.

### Step 2: Build Docker Image

```bash
docker build -t iris-ml-app .
```

### Step 3: Verify Docker Image

```bash
docker images
```

### Step 4: Create and Run Container

```bash
docker run -d -p 8501:8501 --name iris-container iris-ml-app
```

### Step 5: Verify Running Container

```bash
docker ps
```

### Step 6: Access the Application

```text
http://localhost:8501
```

### Useful Docker Commands

Stop container:

```bash
docker stop iris-container
```

Start container:

```bash
docker start iris-container
```

View container logs:

```bash
docker logs iris-container
```

Remove container:

```bash
docker rm -f iris-container
```

---

## ⚙️ 11. CI/CD – GitHub Actions

GitHub Actions is included in the project for workflow automation.

The intended CI/CD workflow is to automate important development and deployment tasks.

The planned workflow includes:

1. Check out the repository.
2. Configure the Python environment.
3. Install project dependencies.
4. Execute the machine learning pipeline.
5. Generate model evaluation reports.
6. Validate model quality.
7. Build the Docker image.
8. Perform deployment-related checks.

The workflow configuration is maintained under the `.github` directory.

---

# 🚀 12. Upcoming Enhancements – Production Monitoring & Observability

The next phase of this project focuses on making the deployed machine learning application more production-oriented.

The objective is to monitor application health, identify changes in incoming data, detect potential model degradation, and generate automated alerts.

The monitoring architecture will integrate Prometheus, Grafana, and Evidently AI.

### 12.1 Model Monitoring Using Grafana

**Planned Tool: Grafana**

Grafana will provide an interactive monitoring dashboard for observing the deployed ML application.

Planned monitoring metrics:

- Number of prediction requests.
- Application availability.
- Prediction response time.
- Request failure rate.
- Application health status.
- Prediction activity over time.
- Data drift indicators.

The dashboard will help visualize application performance and operational health.

### 12.2 Metrics Collection Using Prometheus

**Planned Tool: Prometheus**

Prometheus will collect and store application metrics for monitoring.

Planned implementation:

- Expose application metrics through a metrics endpoint.
- Collect prediction request counts.
- Track application response latency.
- Monitor prediction failures.
- Track application availability.
- Provide metrics to Grafana dashboards.

### 12.3 Data Drift Detection

**Planned Tool: Evidently AI**

Data drift occurs when the distribution of incoming production data changes compared with the reference training data.

The upcoming implementation will compare:

- Reference dataset – Training data.
- Current dataset – Incoming prediction data.

Planned activities:

- Monitor incoming feature distributions.
- Compare production data against training data.
- Identify significant distribution changes.
- Generate data drift reports.
- Track drift indicators over time.
- Identify when model investigation or retraining may be necessary.

### 12.4 Automated Alerting System

**Planned Tool: Grafana Alerting**

An automated alerting mechanism will be introduced to notify users when important monitoring conditions are detected.

Planned alert scenarios:

| Condition                             | Expected Action             |
| ------------------------------------- | --------------------------- |
| Significant data drift                | Generate drift alert        |
| High prediction latency               | Generate performance alert  |
| Application unavailable               | Generate availability alert |
| Increased prediction failures         | Generate error alert        |
| Critical monitoring threshold crossed | Send notification           |

Notification channels will be configured during implementation.

### 12.5 Expected Monitoring Architecture

```text
             STREAMLIT ML APPLICATION
                       |
                       v
               PREDICTION REQUESTS
                       |
             +---------+---------+
             |                   |
             v                   v
        APP METRICS          INPUT DATA
             |                   |
             v                   v
        PROMETHEUS           EVIDENTLY AI
             |                   |
             v                   v
          GRAFANA          DRIFT REPORTS
             |                   |
             +---------+---------+
                       |
                       v
                 ALERT MANAGER
                       |
                       v
                NOTIFICATIONS
```

### Expected Outcome

The monitoring phase aims to establish a continuous observability system that can:

- Monitor application health.
- Track prediction performance.
- Detect changes in incoming data.
- Identify potential data quality problems.
- Generate automated notifications.
- Support informed model maintenance and retraining decisions.

**Project Status:** Monitoring, data drift detection, and alerting are planned enhancements and will be implemented in the next development phase.

---

## 📁 13. Project Structure

```text
Iris_Mlops_Deployment/
│
├── .dvc/
│   └── DVC configuration and cache
│
├── .github/
│   └── work flow/
│       └── main.yaml
│
├── data/
│   └── raw/
│       └── Iris.csv
│
├── src/
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_building.py
│   └── model_evaluation.py
│
├── models/
│   └── model.pkl
│
├── reports/
│   ├── model_selection.json
│   ├── model_comparison.csv
│   ├── model_comparison.md
│   └── evaluation reports
│
├── mlruns/
│
├── app.py
├── Dockerfile
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── requirements.txt
├── .dockerignore
├── .dvcignore
├── .gitignore
└── README.md
```

*Note: Some directories and files are generated during pipeline execution and may not be committed to GitHub. The monitoring directories and configuration files will be added during the upcoming implementation phase.*

---

## 💡 14. Key Learnings

This project provides practical experience in:

- End-to-end machine learning development.
- Data ingestion and preprocessing.
- Feature engineering.
- Hyperparameter optimization.
- Model comparison and selection.
- Model evaluation and overfitting analysis.
- Experiment tracking using MLflow.
- Data and pipeline versioning using DVC.
- Application development using Streamlit.
- Docker image creation and containerization.
- CI/CD workflow development.
- Production monitoring concepts.
- Data drift detection concepts.
- Automated alerting and observability.

---

## 🔮 15. Future Scope

The project will be extended with additional production-oriented MLOps capabilities:

- Prometheus-based metrics collection.
- Grafana monitoring dashboards.
- Evidently AI-based data drift detection.
- Automated drift alerts.
- Model performance monitoring.
- Automated model retraining.
- Cloud-based application deployment.
- Improved application scalability and reliability.

---

## 👩‍💻 16. Author

**Madishetti Rakshitha**

B.Tech Graduate | Data Science & Machine Learning & Gen AI Enthusiast 

**GitHub:** [https://github.com/Rakshitha152003](https://github.com/Rakshitha152003)

**Project Repository:**

[https://github.com/Rakshitha152003/Iris_Mlops_Deployment](https://github.com/Rakshitha152003/Iris_Mlops_Deployment)

---

## ⭐ Acknowledgement

This project represents my practical implementation and learning journey in Machine Learning and MLOps.

It combines machine learning development, pipeline automation, experiment tracking, model evaluation, containerization, and deployment, with plans to extend the system through production monitoring and observability.

If you find this project useful, please consider giving the repository a ⭐.
