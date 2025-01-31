# SeerCuts

A tool for suggesting useful and semantically meaningful discretization strategies for numerical attributes.

## Overview

SeerCuts is an interactive framework that helps users find optimal ways to discretize (partition) numerical data. It leverages GPT-4 for semantic assessment of partitions and employs efficient search strategies to explore discretization options.

Key features:
- Interactive specification of attributes and utility measures
- Semantic meaningfulness assessment using GPT-4
- Efficient exploration of partition space using hierarchical clustering
- Multi-armed bandit policy for identifying useful partitions
- Support for various downstream tasks including visualization and modeling

## Installation

### Prerequisites
- Python 3.12
- OpenAI API key for GPT-4 access

### Setup
1. Clone the repository:
```bash
git clone [https://github.com/yourusername/seercuts.git](https://github.com/noambitton/Demo.git)
cd seercuts
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key'
```

## Prompt Templates

The repository includes pre-defined prompt templates for semantic measurements using GPT-4. These prompts are designed to assess the meaningfulness of different partition strategies and can be found in the `prompts/` directory.

## Validation Study

We conducted a comprehensive human validation study to evaluate the effectiveness of our semantic meaningfulness measurements [user study form](https://forms.gle/NRju2UoYYBzGANSA7).
