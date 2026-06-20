from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import streamlit as st

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from EngTeamCrewSt.crew import (  # noqa: E402
    DEFAULT_REQUIREMENTS,
    AgentModelSettings,
    CrewRunSettings,
    ensure_python_filename,
    run_engineering_team,
)


DEFAULT_OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]


def model_picker(label: str, options: list[str], default: str, key: str) -> str:
    if not options:
        return st.text_input(label, value=default, placeholder="Model name", key=f"{key}_custom_only").strip()

    selected = st.selectbox(label, options + ["Custom"], index=options.index(default) if default in options else 0, key=f"{key}_select")
    if selected == "Custom":
        return st.text_input("Custom model name", value=default, key=f"{key}_custom").strip()
    return selected


@st.cache_data(ttl=30, show_spinner=False)
def get_ollama_models(base_url: str) -> tuple[list[str], str]:
    endpoint = f"{base_url.rstrip('/')}" + "/api/tags"
    try:
        with urllib.request.urlopen(endpoint, timeout=3) as response:
            data = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        return [], str(exc)

    try:
        payload = json.loads(data)
    except ValueError as exc:
        return [], f"Could not parse Ollama response: {exc}"

    models = sorted(
        model["name"]
        for model in payload.get("models", [])
        if isinstance(model, dict) and model.get("name")
    )
    return models, ""


def output_files(output_dir: Path, module_name: str) -> list[Path]:
    module_name = ensure_python_filename(module_name)
    stem = Path(module_name).stem
    return [
        output_dir / f"{stem}_design.md",
        output_dir / module_name,
        output_dir / "app.py",
        output_dir / f"test_{stem}.py",
    ]


st.set_page_config(page_title="Engineering Team Crew", layout="wide")

st.title("Engineering Team Crew")

with st.sidebar:
    st.header("Agent Parameters")
    provider_label = st.radio("LLM provider", ["Ollama", "OpenAI"], horizontal=True)
    provider = provider_label.lower()

    if provider == "ollama":
        base_url = st.text_input("Ollama base URL", value=os.getenv("OLLAMA_API_BASE", "http://localhost:11434"))
        api_key = ""
        if st.button("Refresh Ollama models", use_container_width=True):
            get_ollama_models.clear()
        model_options, ollama_error = get_ollama_models(base_url)
        default_model = model_options[0] if model_options else ""
        if ollama_error:
            st.warning(f"Could not load Ollama models from {base_url}. Enter a model name manually.")
        elif not model_options:
            st.info("No Ollama models found. Pull a model with `ollama pull <model>` or enter a model name manually.")
    else:
        base_url = st.text_input("OpenAI base URL", value=os.getenv("OPENAI_BASE_URL", ""))
        api_key = st.text_input("OpenAI API key", value="", type="password")
        model_options = DEFAULT_OPENAI_MODELS
        default_model = DEFAULT_OPENAI_MODELS[0]

    same_model = st.toggle("Use one model for all agents", value=True)
    if same_model:
        model = model_picker("Model", model_options, default_model, "single_model")
        agent_models = AgentModelSettings(model, model, model, model)
    else:
        with st.expander("Per-agent models", expanded=True):
            agent_models = AgentModelSettings(
                engineering_lead=model_picker("Engineering lead", model_options, default_model, "lead_model"),
                backend_engineer=model_picker("Backend engineer", model_options, default_model, "backend_model"),
                frontend_engineer=model_picker("Frontend engineer", model_options, default_model, "frontend_model"),
                test_engineer=model_picker("Test engineer", model_options, default_model, "test_model"),
            )

    temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.2, step=0.1)
    verbose = st.toggle("Verbose CrewAI logs", value=True)
    allow_code_execution = st.toggle("Allow backend/test code execution", value=True)

left, right = st.columns([2, 1])

with left:
    requirements = st.text_area("Requirements", value=DEFAULT_REQUIREMENTS, height=360)

with right:
    module_name = st.text_input("Module filename", value="accounts.py")
    class_name = st.text_input("Primary class name", value="Account")
    output_dir_text = st.text_input("Output directory", value="output_streamlit")
    run_button = st.button("Run engineering crew", type="primary", use_container_width=True)

    st.caption("Generated files are written under the selected output directory.")

if run_button:
    try:
        module_name = ensure_python_filename(module_name)
        output_dir = Path(output_dir_text).expanduser()
        if provider == "openai" and not api_key and not os.getenv("OPENAI_API_KEY"):
            st.error("Provide an OpenAI API key or set OPENAI_API_KEY before running.")
            st.stop()
        if not requirements.strip():
            st.error("Requirements cannot be empty.")
            st.stop()
        if not class_name.strip():
            st.error("Primary class name cannot be empty.")
            st.stop()

        settings = CrewRunSettings(
            provider=provider,
            agent_models=agent_models,
            requirements=requirements.strip(),
            module_name=module_name,
            class_name=class_name.strip(),
            output_dir=output_dir,
            api_key=api_key,
            base_url=base_url.strip(),
            temperature=temperature,
            verbose=verbose,
            allow_code_execution=allow_code_execution,
        )

        with st.spinner("Running the engineering crew..."):
            result, logs = run_engineering_team(settings)
        st.session_state["last_result"] = result
        st.session_state["last_logs"] = logs
        st.session_state["last_output_dir"] = str(output_dir)
        st.session_state["last_module_name"] = module_name
        st.success("Crew run completed.")
    except Exception as exc:
        st.exception(exc)

if "last_output_dir" in st.session_state:
    output_dir = Path(st.session_state["last_output_dir"])
    module_name = st.session_state["last_module_name"]
    files = output_files(output_dir, module_name)
    existing_files = [path for path in files if path.exists()]

    if existing_files:
        st.subheader("Generated Files")
        tabs = st.tabs([path.name for path in existing_files])
        for tab, path in zip(tabs, existing_files):
            with tab:
                text = path.read_text(encoding="utf-8", errors="replace")
                language = "python" if path.suffix == ".py" else "markdown"
                st.code(text, language=language)
                st.download_button(
                    "Download",
                    data=text,
                    file_name=path.name,
                    mime="text/plain",
                    key=f"download_{path.name}",
                )

    if st.session_state.get("last_result") is not None:
        with st.expander("Crew result"):
            st.write(st.session_state["last_result"])

    logs = st.session_state.get("last_logs")
    if logs:
        with st.expander("CrewAI logs"):
            st.code(logs)

