<h1 align="center">Rank1: Test-Time Compute for Reranking in Information Retrieval</h1>

<h4 align="center">
    <p>
        <a href="#links">Model/Data Links</a> |
        <a href="#installation">Installation</a> |
        <a href="#usage">Usage</a> |
        <a href="#citing">Citing</a>
    <p>
</h4>

Official repository for Rank1, a reasoning reranker model that "thinks". Rank1 leverages test-time compute to generate reasoning chains before making relevance judgments.

## Links
| Resource | Description |
|:---------|:------------|
| [Rank1-7B](https://huggingface.co/jhu-clsp/Rank1-7B) | Trained from Qwen-7B base for document reranking with reasoning |
| [Rank1-14B](https://huggingface.co/jhu-clsp/Rank1-14B) | Trained from Qwen-14B base for document reranking with reasoning |
| [Rank1-32B](https://huggingface.co/jhu-clsp/Rank1-32B) | Trained from Qwen-32B base for document reranking with reasoning |
| [Rank1-Mistral-2501-24B](https://huggingface.co/jhu-clsp/Rank1-Mistral-2501-24B) | Trained from Mistral-Small 2501 24B base for document reranking with reasoning |
| [Rank1-Llama3-8B](https://huggingface.co/jhu-clsp/Rank1-Llama3-8B) | Trained from Llama 3.1 8B base for document reranking with reasoning |
| [Rank1-7B-awq](https://huggingface.co/jhu-clsp/Rank1-7B-awq) | Quantized version of Rank1-7B for efficient inference |
| [Rank1-14B-awq](https://huggingface.co/jhu-clsp/Rank1-14B-awq) | Quantized version of Rank1-14B for efficient inference |
| [Rank1-32B-awq](https://huggingface.co/jhu-clsp/Rank1-32B-awq) | Quantized version of Rank1-32B for efficient inference |
| [Rank1-Mistral-2501-24B-awq](https://huggingface.co/jhu-clsp/Rank1-Mistral-2501-24B-awq) | Quantized version of Rank1-Mistral-24B for efficient inference |
| [Rank1-Llama3-8B-awq](https://huggingface.co/jhu-clsp/Rank1-Llama3-8B-awq) | Quantized version of Rank1-Llama3-8B for efficient inference |
| [Rank1-R1-MSMARCO](https://huggingface.co/datasets/jhu-clsp/Rank1-R1-MSMARCO) | All R1 output examples from MS MARCO |
| [Rank1-training-data](https://huggingface.co/datasets/jhu-clsp/Rank1-training-data) | Training data used for Rank1 models |
| [Rank1-Run-Files](https://huggingface.co/datasets/jhu-clsp/Rank1-Run-Files) | Pre-computed run files for use in top 100 doc reranking |

## Installation 
To reproduce the experiments, you can use the following code with uv for fast, reliable dependency management:

```bash
git clone https://github.com/orionw/rank1.git
cd rank1/

# Install uv if you don't have it already
curl -fsSL https://pkg.uv.dev/install.sh | sh

# Create and activate virtual environment with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies with uv
uv pip install -r requirements.txt
uv pip install -e mteb/

# Recommended: download a flash attention wheel from https://github.com/Dao-AILab/flash-attention/releases and `uv pip install` it

# Download the Rank1-Run-Files repository (required for evaluation)
git lfs install
git clone https://huggingface.co/datasets/jhu-clsp/Rank1-Run-Files
```

## Usage
### Running Evaluations
To run an evaluation with the Rank1 model on a specific dataset:

```bash
bash launch_job.sh jhu-clsp/Rank1-7B BrightRetrieval biology 1
```

Parameters:
- `jhu-clsp/Rank1-7B`: Model name or path
- `BrightRetrieval`: Dataset name
- `biology`: Subtask name (use "default" if no subtask)
- `1`: Number of GPUs to use

For batch evaluation of multiple benchmarks:

```bash
# Example: evaluate on all MTEB instruction-following IR tasks
python run_mteb.py -m jhu-clsp/Rank1-7B -d all -n 1
```

### Using Rank1 in Your Own Code
You can integrate Rank1 into your code:

```python
from rank1 import Rank1

# Initialize the model
model = Rank1(
    model_name_or_path="jhu-clsp/Rank1-7B",
    num_gpus=1,
    device="cuda",
    context_size=16000,
    max_output_tokens=8192,
    fp_options="float16"
)

# Rerank documents
results = model.predict({
    "query": "Your query here",
    "corpus": ["Document 1 content", "Document 2 content", ...],
    "instructions": "Your specific instructions here"
})
```

### MTEB Integration
Rank1 is compatible with the MTEB benchmarking framework. To evaluate your model:

```python
from mteb import MTEB
from rank1 import Rank1

# Initialize your model
model = Rank1(
    model_name_or_path="jhu-clsp/Rank1-7B",
    num_gpus=1,
    device="cuda"
)

# Select tasks (or use specific task names)
evaluation = MTEB(tasks=["BrightRetrieval"])

# Run evaluation
results = evaluation.run(model)
```

## Citing
If you use Rank1 in your research, please cite our work:

```bibtex
@article{weller2023rank1,
  title={Rank1: Test-Time Compute for Reranking in Information Retrieval},
  author={Weller, Orion and Ricci, Kathryn and Yang, Eugene and Yates, Andrew and Lawrie, Dawn and Van Durme, Benjamin},
  journal={arXiv preprint},
  year={2025},
}
```

## License
[MIT](LICENSE)
