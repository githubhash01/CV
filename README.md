# CV Project

This project is structured to work with a dataset and includes a utility script to build a cached version of the dataset.

## Setup Instructions

### 1. Add the Dataset

Place your dataset folder in the main directory of the repository. The folder should be named `Dataset` and be located alongside this README. For example:

/CV
├── Dataset/       # <– Your dataset folder goes here
├── src/
├── .gitignore
├── README.md

### 2. Build the Cached Dataset

The project includes a utility script `utils.py` located in the `src` folder to build a cached version of the dataset. To run the script, follow these steps:

1. Open your terminal and navigate to the project directory:
   ```bash
   cd /path/to/CV

python src/utils.py
