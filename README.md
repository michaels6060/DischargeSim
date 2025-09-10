# DischargeSim

DischargeSim is a medical discharge simulation framework that creates realistic doctor-patient interactions to simulate hospital discharge scenarios. 

## Overview

This system uses AI agents playing the roles of doctors and patients to simulate medical discharge conversations. The doctor agent guides patients through understanding their diagnosis, treatments, medications, and follow-up care plans. The framework produces structured output including:

- Complete conversation logs
- AHRQ-format after-visit plans
- Discharge summaries in clinical note format

## Installation

### Prerequisites

- Python 3.9+
- [vLLM](https://github.com/vllm-project/vllm) (for running local models)
- OpenAI API key (for using OpenAI models)

### Setup

Create a conda environment and install dependencies:

```
conda create -n discharge python=3.10
conda activate discharge
pip install -r requirements.txt
```

## Usage

### Basic Usage

To run a simulation with default settings:
```
python DischargeSim.py 
--openai_api_key YOUR_API_KEY 
--doctor_llm gpt-4o-mini 
--patient_llm gpt-4o-mini 
--agent_dataset MIMICIV 
--output_file output_directory
```

### Using Local Models with vLLM

1. Start the vLLM server:
```
python -m vllm.entrypoints.openai.api_server --model PATH_TO_YOUR_MODEL
```

2. Run the simulation using the local model:
```
python DischargeSim.py 
--doctor_llm qwen 
--patient_llm gpt-4o-mini 
--agent_dataset MIMICIV 
--output_file output_directory 
--model_file PATH_TO_YOUR_MODEL 
--vllm_port 8000
```

## Output

The simulation produces three main output files in the specified directory:

1. `history.csv` - Complete conversation logs between doctor and patient
2. `plan.jsonl` - Structured after-visit plans in AHRQ format
3. `summary.jsonl` - Discharge summaries in clinical documentation format


## License
CC-BY-NC 4.0
