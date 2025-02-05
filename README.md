# RAGS_BUNNY

RAGS_BUNNY is a project for creating SQL queries from natural language questions and is still being developed. 
The project provides text processing and SQL query generation functionality using powerful tools such as Hugging Face models, Langchain, FAISS, PyTorch and Streamlit. In this project, a simple interface with Streamlit was used to generate SQL queries in response to natural language questions given by users, and is still open to development.

## Project Environment

### Required Environment:
This project is designed to run in the Python environment. You can install the required environment by following the steps below:

 1. **Creating a Virtual Environment (Optional):**
   It is recommended that you use a virtual environment to isolate the project's dependencies. You can follow the commands below to create the virtual environment.

   #### Creating an Environment Using Conda:

```bash
   conda create -n rags_bunny python=3.9
   conda activate rags_bunny
```

#### Creating Using Virtualenv:
```bash
  python -m venv rags_bunny
  source rags_bunny/bin/activate  # MacOS/Linux 
  rags_bunny\Scripts\activate     # Windows 
  ```
2. **Installing Dependencies:**
    You can use the requirements.txt file to install the required Python libraries. You can install dependencies by running the following command via terminal:
```bash
pip install -r requirements.txt
```

3.**GPU Support (Device Selection):**
The project uses GPU for model and embedding calculations. However, the MPS device can only be used on MacOS. Windows users require CUDA support. Here's how you can change device settings:

**-For MacOS:** 
If you are using MacOS, the project will use MPS (Metal Performance Shaders) by default.

```bash
device = torch.device("mps")
```
**-For Windows:**
If you are using Windows, a CUDA-supported GPU is required. To work in a CUDA supported environment, you can set the device to CUDA as follows:

```bash
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**Important:** For CUDA to work properly, you must have an NVIDIA GPU and the appropriate CUDA drivers (CUDA Toolkit and cuDNN) installed.

## File structure and contents

```bash
RAGS_BUNNY/complex_rag
│
├── app.py                   
├── sql_generator.py         
├── chunks.csv               
├── embeddings.csv           
├── requirements.txt         
├── README.md                
```

**-app.py**: Streamlit-based interface receives natural language input from the user and generates SQL queries.

**-sql_generator.py**: Python code that performs the operations necessary to generate SQL queries. This file processes natural language texts for database queries and converts them into appropriate SQL queries.

**-chunks.csv**: File containing database texts and associated embeddings.

**-embeddings.csv**: File containing embeddings of various text fragments. It could not be uploaded here due to its large size, but you can run the sql_generator.py file to do automatic chunking and embedding and save it as .csv.

**-requirements.txt**: Contains project dependencies, required during installation.

## Models and Dataset used

**Dataset** : https://huggingface.co/datasets/b-mc2/sql-create-context

**LLM Model**: https://huggingface.co/cssupport/t5-small-awesome-text-to-sql

**Embedding Model**: https://huggingface.co/sentence-transformers/all-mpnet-base-v2

## Usage

### Install Required Libraries:
Before running the project, run the following command to install all the dependencies of the project:

```bash
pip install -r requirements.txt
```

### Launch Streamlit App:
You can launch the Streamlit interface by running app.py:

```bash
streamlit run app.py
```

### Enter Natural Language Question:
When the app opens, enter a question in natural language. Since the project is still in development, it is dependent on FAISS, so it is recommended that you choose your question from the dataset. After entering your question, wait for the system to create the SQL query.

### View SQL Results:
The SQL query generated in response to the natural language question entered by the user will be displayed on the screen.


# WARNING

The project has some deficiencies in Prompt sensitivity, and until these are resolved, it is recommended that you generate queries based on the questions in the https://huggingface.co/datasets/b-mc2/sql-create-context/viewer/default/train dataset. Despite the semantic inputs you enter, the model will still try to return results via vector DB. Thank you for your understanding!
