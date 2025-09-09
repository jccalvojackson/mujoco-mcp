# MuJoCo Robot Joint Configuration Evaluation Demo

This demo evaluates AI agents' ability to match robot joint configurations from reference images using different language models (Claude and Gemini).

## Overview

The `evaluate_directly.py` script:

- Creates a simulated robot environment using MuJoCo
- Generates reference images from random joint configurations
- Uses AI agents (Claude or Gemini) to predict joint positions from the images
- Measures accuracy by comparing end-effector poses
- Logs results to Weights & Biases (wandb)

## Setup Instructions

### 1. Install Dependencies

Install the demo dependencies using uv:

```bash
uv pip install -e . --group demo
```

### 2. Configure Environment Variables

Create a `.env` file with your API keys. Start by copying the example:

```bash
cp .env.example .env
```

Then edit `.env` to include:

```bash
# Anthropic API key for Claude models
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Google API key for Gemini models
GOOGLE_API_KEY=your_google_api_key_here

# Weights & Biases API key for experiment tracking
WANDB_API_KEY=your_wandb_api_key_here
```

**Getting API Keys:**

- **Anthropic**: Get your API key from [console.anthropic.com](https://console.anthropic.com/)
- **Google**: Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Weights & Biases**: Get your API key from [wandb.ai/settings](https://wandb.ai/settings)

### 3. Install GNU Parallel

GNU Parallel enables efficient concurrent execution:

**macOS:**

```bash
brew install parallel
```

**Ubuntu/Debian:**

```bash
sudo apt-get install parallel
```

**Fedora:**

```bash
sudo dnf install parallel
```

**First-time setup:**

```bash
parallel --record-env  # Run this once in a clean environment
```

## Running the Evaluation

### Single Run

Test with a single configuration:

```bash
# Load environment variables
set -a; source .env; set +a

# Run with Claude
uv run examples/evaluate_directly.py --model-class claude --ground-truth-seed 42

# Run with Gemini
uv run examples/evaluate_directly.py --model-class gemini --ground-truth-seed 42
```

### Parallel Evaluation

Run comprehensive evaluation across both models and multiple seeds:

```bash
# Load environment variables
set -a; source .env; set +a

# Run parallel evaluation (4 concurrent jobs)
parallel -j 4 --env _ "uv run examples/evaluate_directly.py --model-class {1} --ground-truth-seed {2}" ::: claude gemini ::: {1..10}
```

**Command breakdown:**

- `set -a; source .env; set +a`: Export all variables from `.env`
- `parallel -j 4`: Run up to 4 jobs concurrently
- `--env _`: Pass all environment variables to parallel jobs
- `{1}` and `{2}`: Placeholders for parameter combinations
- `::: claude gemini`: Model classes to test
- `::: {1..10}`: Ground truth seeds 1 through 10

This creates **20 total experiments** (2 models × 10 seeds) running 4 at a time.

## Command Line Options

```bash
uv run examples/evaluate_directly.py --help
```

Available options:

- `--robot-name`: Robot model to use (default: "so_arm100_mj_description")
- `--model-class`: AI model to use ("claude" or "gemini", default: "gemini")
- `--ground-truth-source`: Data source ("simulated" or "real", default: "simulated")
- `--ground-truth-seed`: Random seed for reproducible experiments (default: 42)

## Output and Results

### Weights & Biases Dashboard

Results are automatically logged to your wandb project `evaluate_joint_configuration_agent`. View:

- End-effector pose distance metrics
- Model performance comparisons
- Experiment configurations
- Run history and trends

### Metrics

The primary metric is **end-effector pose distance**: the Euclidean distance between ground truth and predicted end-effector positions/orientations.

Lower values indicate better performance.

## Troubleshooting

### Common Issues

1. **Missing API keys**: Ensure all required keys are in `.env`
2. **Parallel not found**: Install GNU Parallel using your package manager
3. **Import errors**: Run `uv pip install -e . --group demo` to install dependencies
4. **Wandb login**: Run `wandb login` if authentication fails

### Debug Mode

Run a single experiment with verbose output:

```bash
set -a; source .env; set +a
uv run examples/evaluate_directly.py --model-class claude --ground-truth-seed 1
```

### Performance Tips

- Adjust `-j` parameter based on your system (e.g., `-j 2` for fewer concurrent jobs)
- Use different seed ranges for different experiments
- Monitor system resources during parallel runs

## Example Results

Typical results show:

- **Claude**: Generally more consistent joint position predictions
- **Gemini**: Faster inference but higher variance in accuracy
- **Seed variation**: Different poses provide varying difficulty levels

Check your wandb dashboard for detailed performance analysis and visualizations.

## Citation

This evaluation uses GNU Parallel for efficient parallel processing:

```bibtex
@software{tange_2025_16944306,
      author       = {Tange, Ole},
      title        = {GNU Parallel 20250822 ('Петропавловск')},
      month        = Aug,
      year         = 2025,
      note         = {{GNU Parallel is a general parallelizer to run
                       multiple serial command line programs in parallel
                       without changing them.}},
      publisher    = {Zenodo},
      doi          = {10.5281/zenodo.16944306},
      url          = {https://doi.org/10.5281/zenodo.16944306}
}
```
