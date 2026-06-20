from __future__ import annotations

import contextlib
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crewai import Agent, Crew, Process, Task

try:
    from crewai import LLM
except ImportError:  # pragma: no cover - older CrewAI fallback
    LLM = None


DEFAULT_REQUIREMENTS = """A simple account management system for a trading simulation platform.
The system should allow users to create an account, deposit funds, and withdraw funds.
The system should allow users to record that they have bought or sold shares, providing a quantity.
The system should calculate the total value of the user's portfolio, and the profit or loss from the initial deposit.
The system should be able to report the holdings of the user at any point in time.
The system should be able to report the profit or loss of the user at any point in time.
The system should be able to list the transactions that the user has made over time.
The system should prevent the user from withdrawing funds that would leave them with a negative balance, or
from buying more shares than they can afford, or selling shares that they don't have.
The system has access to a function get_share_price(symbol) which returns the current price of a share, and includes a test implementation that returns fixed prices for AAPL, TSLA, GOOGL.
"""


@dataclass(frozen=True)
class AgentModelSettings:
    engineering_lead: str
    backend_engineer: str
    frontend_engineer: str
    test_engineer: str


@dataclass(frozen=True)
class CrewRunSettings:
    provider: str
    agent_models: AgentModelSettings
    requirements: str
    module_name: str
    class_name: str
    output_dir: Path
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.2
    verbose: bool = True
    allow_code_execution: bool = True


def ensure_python_filename(module_name: str) -> str:
    module_name = module_name.strip()
    if not module_name:
        raise ValueError("Module filename is required.")
    if any(separator in module_name for separator in ("/", "\\")):
        raise ValueError("Module filename must not include a path.")
    return module_name if module_name.endswith(".py") else f"{module_name}.py"


def normalize_model_name(provider: str, model_name: str) -> str:
    model_name = model_name.strip()
    if not model_name:
        raise ValueError("Model name is required.")

    provider = provider.lower()
    if provider == "ollama" and not model_name.startswith("ollama/"):
        return f"ollama/{model_name}"
    if provider == "openai" and "/" not in model_name:
        return f"openai/{model_name}"
    return model_name


def build_llm(provider: str, model_name: str, base_url: str, api_key: str, temperature: float) -> Any:
    model = normalize_model_name(provider, model_name)
    provider = provider.lower()

    if provider == "openai" and api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if provider == "openai" and base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
    if provider == "ollama" and base_url:
        os.environ["OLLAMA_API_BASE"] = base_url

    if LLM is None:
        return model

    kwargs: dict[str, Any] = {"model": model, "temperature": temperature}
    if base_url:
        kwargs["base_url"] = base_url
    if api_key and provider == "openai":
        kwargs["api_key"] = api_key
    return LLM(**kwargs)


def agent_config(agent_name: str) -> dict[str, str]:
    configs = {
        "engineering_lead": {
            "role": "Engineering Lead for the engineering team, directing the work of the engineer",
            "goal": (
                "Take the high level requirements described here and prepare a detailed design for the backend developer; "
                "everything should be in 1 python module; describe the function and method signatures in the module. "
                "The python module must be completely self-contained, and ready so that it can be tested or have a simple UI built for it. "
                "Here are the requirements: {requirements} "
                "The module should be named {module_name} and the class should be named {class_name}"
            ),
            "backstory": "You're a seasoned engineering lead with a knack for writing clear and concise designs.",
        },
        "backend_engineer": {
            "role": "Python Engineer who can write code to achieve the design described by the engineering lead",
            "goal": (
                "Write a python module that implements the design described by the engineering lead, in order to achieve the requirements. "
                "The python module must be completely self-contained, and ready so that it can be tested or have a simple UI built for it. "
                "Here are the requirements: {requirements} "
                "The module should be named {module_name} and the class should be named {class_name}"
            ),
            "backstory": (
                "You're a seasoned python engineer with a knack for writing clean, efficient code. "
                "You follow the design instructions carefully. "
                "You produce 1 python module named {module_name} that implements the design and achieves the requirements."
            ),
        },
        "frontend_engineer": {
            "role": "A Gradio expert to who can write a simple frontend to demonstrate a backend",
            "goal": (
                "Write a gradio UI that demonstrates the given backend, all in one file to be in the same directory as the backend module {module_name}. "
                "Here are the requirements: {requirements}"
            ),
            "backstory": (
                "You're a seasoned python engineer highly skilled at writing simple Gradio UIs for a backend class. "
                "You produce a simple gradio UI that demonstrates the given backend class; you write the Gradio UI in a module app.py "
                "that is in the same directory as the backend module {module_name}."
            ),
        },
        "test_engineer": {
            "role": "An engineer with python coding skills who can write unit tests for the given backend module {module_name}",
            "goal": "Write unit tests for the given backend module {module_name} and create a test module in the same directory as the backend module.",
            "backstory": "You're a seasoned QA engineer and software developer who writes great unit tests for python code.",
        },
    }
    return configs[agent_name]


def make_agent(agent_name: str, llm: Any, settings: CrewRunSettings, code_agent: bool = False) -> Agent:
    kwargs: dict[str, Any] = {
        "config": agent_config(agent_name),
        "llm": llm,
        "verbose": settings.verbose,
    }
    if code_agent and settings.allow_code_execution:
        kwargs.update(
            {
                "allow_code_execution": True,
                "code_execution_mode": "safe",
                "max_execution_time": 500,
                "max_retry_limit": 3,
            }
        )
    return Agent(**kwargs)


def build_crew(settings: CrewRunSettings) -> Crew:
    module_name = ensure_python_filename(settings.module_name)
    module_stem = Path(module_name).stem
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    lead = make_agent(
        "engineering_lead",
        build_llm(settings.provider, settings.agent_models.engineering_lead, settings.base_url, settings.api_key, settings.temperature),
        settings,
    )
    backend = make_agent(
        "backend_engineer",
        build_llm(settings.provider, settings.agent_models.backend_engineer, settings.base_url, settings.api_key, settings.temperature),
        settings,
        code_agent=True,
    )
    frontend = make_agent(
        "frontend_engineer",
        build_llm(settings.provider, settings.agent_models.frontend_engineer, settings.base_url, settings.api_key, settings.temperature),
        settings,
    )
    tester = make_agent(
        "test_engineer",
        build_llm(settings.provider, settings.agent_models.test_engineer, settings.base_url, settings.api_key, settings.temperature),
        settings,
        code_agent=True,
    )

    design_task = Task(
        description=(
            "Take the high level requirements described here and prepare a detailed design for the engineer; "
            "everything should be in 1 python module, but outline the classes and methods in the module. "
            "Here are the requirements: {requirements} "
            "IMPORTANT: Only output the design in markdown format, laying out in detail the classes and functions in the module, describing the functionality."
        ),
        expected_output="A detailed design for the engineer, identifying the classes and functions in the module.",
        agent=lead,
        output_file=str(settings.output_dir / f"{module_stem}_design.md"),
    )
    code_task = Task(
        description=(
            "Write a python module that implements the design described by the engineering lead, in order to achieve the requirements. "
            "Here are the requirements: {requirements}"
        ),
        expected_output=(
            "A python module that implements the design and achieves the requirements. "
            "IMPORTANT: Output ONLY the raw Python code without any markdown formatting, code block delimiters, or backticks. "
            "The output should be valid Python code that can be directly saved to a file and executed."
        ),
        agent=backend,
        context=[design_task],
        output_file=str(settings.output_dir / module_name),
    )
    frontend_task = Task(
        description=(
            "Write a gradio UI in a module app.py that demonstrates the given backend class in {module_name}. "
            "Assume there is only 1 user, and keep the UI very simple indeed - just a prototype or demo. "
            "Here are the requirements: {requirements}"
        ),
        expected_output=(
            "A gradio UI in module app.py that demonstrates the given backend class. "
            "The file should be ready so that it can be run as-is, in the same directory as the backend module, and it should import the backend class from {module_name}. "
            "IMPORTANT: Output ONLY the raw Python code without any markdown formatting, code block delimiters, or backticks. "
            "The output should be valid Python code that can be directly saved to a file and executed."
        ),
        agent=frontend,
        context=[code_task],
        output_file=str(settings.output_dir / "app.py"),
    )
    test_task = Task(
        description="Write unit tests for the given backend module {module_name} and create a test module in the same directory as the backend module.",
        expected_output=(
            "A test module that tests the given backend module. "
            "IMPORTANT: Output ONLY the raw Python code without any markdown formatting, code block delimiters, or backticks. "
            "The output should be valid Python code that can be directly saved to a file and executed."
        ),
        agent=tester,
        context=[code_task],
        output_file=str(settings.output_dir / f"test_{module_stem}.py"),
    )

    return Crew(
        agents=[lead, backend, frontend, tester],
        tasks=[design_task, code_task, frontend_task, test_task],
        process=Process.sequential,
        verbose=settings.verbose,
    )


def run_engineering_team(settings: CrewRunSettings) -> tuple[Any, str]:
    module_name = ensure_python_filename(settings.module_name)
    inputs = {
        "requirements": settings.requirements,
        "module_name": module_name,
        "class_name": settings.class_name.strip(),
    }
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        result = build_crew(settings).kickoff(inputs=inputs)
    return result, buffer.getvalue()

