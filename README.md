# MuJoCo MCP: AI-Controlled Robot Pose Matching

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

An MCP (Model Context Protocol) server that enables AI systems to control a simulated robot arm in MuJoCo to match target poses from reference images. 

## 🎯 Problem Statement

...

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jccalvojackson/mujoco-mcp.git
   cd mujoco-mcp
   ```

2. **Set up virtual environment and install dependencies:**
   ```bash
   uv sync
   source .venv/bin/activate
   ```

3. **Run the MCP server:**
   ```bash
   python -m mujoco_mcp.server
   ```

### Usage with MCP-enabled AI Agents

The server exposes a single tool and prompt for AI agents:

**Tool:** `set_robot_state(state: list[float])`
- **Input:** Joint positions as a list of floats (6 values for so_arm100)
- **Output:** WebP image grid showing 4 camera viewpoints of the robot
- **Behavior:** Stateless - robot resets to home position after each call

**Prompt:** "Achieve pose"
- Provides detailed instructions for iterative pose matching
- Guides the AI through analysis, planning, and refinement steps

## 🛠️ Development Roadmap

### 🚨 Critical
- [x] ✅ Enhanced README with comprehensive documentation
- [x] ✅ MIT License added
- [x] ✅ Fixed main function to run server properly
- [x] ✅ Added type hints and comprehensive docstrings
- [x] ✅ Updated project metadata with proper description and keywords
- [ ] 🔄 Create examples/ directory with demo scripts and sample images
- [ ] 🔄 Add basic unit tests for core functionality
- [ ] 🔄 Implement proper error handling and input validation

### ⚠️ Important
- [ ] 🔄 Add pre-commit hooks, ruff formatting
- [ ] 🔄 Set up GitHub Actions for CI/CD (testing, linting, formatting)

### ✨ Nice-to-Have
- [ ] 🔄 Multi-robot support for different models
- [ ] 🔄 Web interface for interactive robot control

## 🤝 Contributing

Contributions are welcome and appreciated! Whether you're fixing bugs, adding features, or improving documentation, your help makes this project better.

### How to Contribute

1. **Fork the repository** and create your feature branch
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. **Make your changes** following the existing code style
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Submit a pull request** with a clear description

### Ways to Contribute

- 🐛 **Report bugs** via [GitHub Issues](https://github.com/jccalvojackson/mujoco-mcp/issues)
- 💡 **Suggest features** or improvements
- 🔧 **Fix issues** from the roadmap above
- 📚 **Improve documentation** and examples
- ⭐ **Star the repository** if you find it useful
- 🗣️ **Share the project** with others who might benefit

### Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/mujoco-mcp.git
cd mujoco-mcp

# Set up development environment
uv sync
source .venv/bin/activate

# Run tests (when available)
python -m pytest

# Run the server for testing
python -m mujoco_mcp.server
```

All contributions, no matter how small, are valued and appreciated!

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
