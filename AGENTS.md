# General 

I am holding a course on the topics DevOps and MLOps for students of AIS (Artificial Intelligence Sytems) in the second semester. The topics that are handled are 

- Motivation for DevOps / MLOps
- Principles of Flow, Feedback, Continuous Learning and Experimentation
- Version Control using Git
- Zero Configuration, Build Tooling
- Automated Testing
- CI, Github Build Pipelines
- Packaging with Docker
- IaC using Docker Compose (potentially K8S if we have time)
- Observability / Production Telemetry (Logs, Metrics, Traces)
- Monitoring

You help me creating contents and especially creating simplest contents that students can grasp. We only have 9 sessions a 150 minutes time to tackle these topics, therefore everything must be kept as simple as possible. The contents are taught in the form of 

- lectured content (slides)
- exercises that are directly done in pairing sessions in the sessions
- exercises that follow up on the session contexts and that are done as homework

Each session follows this principle outline
- Grasp the problem at hand by experiencing how it is if you don't have the practices in place (e.g. diagnose an app that misbehaves but does not have observability)
- Have a block with theory (lecture style)
- Have an example that students can work on on their own in pairing mode in the session
- Have additional examples in a "Level Up" section available for students that are faster and for homework
- If required, there can be multiple of such blocks (theory + self study) in a single session (e.g. per topic)
The students are not very far in their studies. They only had 1 semester of Python basics.

# Instructions for this Repository

This repository focuses on these MLOps topics
- Organising data using Parquet
- Data Management via Dvc
- Training of a model
- Usage of MLFlow for experiment tracking
- Usage of MLFlow for model registry
- GitHub Actions Workflow for training the model and registering it in the MLFlow model registry
- Usage of the model in a simple FastAPI app

# Tooling 

The project is managed via uv. Therefore run all python scripts via `uv run <script name>`. 
Also make use of `uvx` if further tools, such as dvc are required. 
