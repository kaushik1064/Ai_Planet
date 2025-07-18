import os

# Define folder and file structure
structure = {
    "math_agent": {
        "backend": {
            "app": {
                "__init__.py": "",
                "main.py": "# FastAPI main application\n",
                "models": {
                    "__init__.py": "",
                    "schemas.py": "# Pydantic models\n"
                },
                "core": {
                    "__init__.py": "",
                    "config.py": "# Configuration settings\n",
                    "guardrails.py": "# Input/Output guardrails\n",
                    "logging.py": "# Logging configuration\n"
                },
                "agents": {
                    "__init__.py": "",
                    "routing_agent.py": "# Main routing logic\n",
                    "math_solver.py": "# Math problem solving\n",
                    "feedback_agent.py": "# Human-in-the-loop feedback\n",
                    "evaluation_agent.py": "# Solution evaluation\n"
                },
                "services": {
                    "__init__.py": "",
                    "groq_service.py": "# Groq API integration\n",
                    "vector_db.py": "# Qdrant vector database\n",
                    "web_search.py": "# Web search using MCP\n",
                    "knowledge_base.py": "# Knowledge base operations\n"
                },
                "api": {
                    "__init__.py": "",
                    "routes": {
                        "__init__.py": "",
                        "math.py": "# Math endpoints\n",
                        "feedback.py": "# Feedback endpoints\n",
                        "health.py": "# Health check endpoints\n"
                    }
                },
                "utils": {
                    "__init__.py": "",
                    "data_processor.py": "# Process math dataset\n",
                    "embeddings.py": "# Generate embeddings\n",
                    "helpers.py": "# Utility functions\n"
                }
            },
            "data": {
                "raw": {},           # Raw JSON files (238 files)
                "processed": {},     # Processed knowledge base
                "embeddings": {}     # Vector embeddings
            },
            "scripts": {
                "setup_knowledge_base.py": "# Initialize knowledge base\n",
                "benchmark_jee.py": "# JEE benchmark testing\n",
                "data_migration.py": "# Data migration utilities\n"
            },
            "requirements.txt": ""
        },
        "frontend": {
            "src": {
                "components": {
                    "MathInput.jsx": "// Math problem input\n",
                    "SolutionDisplay.jsx": "// Step-by-step solution display\n",
                    "FeedbackForm.jsx": "// Human feedback form\n",
                    "LoadingSpinner.jsx": "// Loading indicators\n"
                },
                "services": {
                    "api.js": "// API service layer\n",
                    "utils.js": "// Utility functions\n"
                },
                "hooks": {
                    "useMathAgent.js": "// Custom hook for math agent\n",
                    "useFeedback.js": "// Custom hook for feedback\n"
                },
                "App.jsx": "",
                "index.js": ""
            },
            "package.json": ""
        },
        "docker": {
            "Dockerfile.backend": "",
            "Dockerfile.frontend": "",
            "docker-compose.yml": "",
            "nginx.conf": ""
        },
        "scripts": {
            "deploy.sh": "#!/bin/bash\n"
        },
        ".env.example": "",
        ".gitignore": "",
        "README.md": "# Math Agent Project\n"
    }
}

def create_structure(base_path, struct):
    for name, content in struct.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        else:
            with open(path, 'w') as f:
                f.write(content)

# Run the script
if __name__ == "__main__":
    create_structure('.', structure)
    print("✅ Project structure created successfully.")
