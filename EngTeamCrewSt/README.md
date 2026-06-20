# Engineering Team Crew Streamlit App

This is a Streamlit version of the original `src/engineering_team` CrewAI app. It keeps the same four-agent workflow:

- engineering lead
- backend engineer
- frontend engineer
- test engineer

The UI lets users enter requirements, choose the target Python module and class names, pick Ollama or OpenAI as the LLM provider, and configure model/runtime settings before running the crew.

## Run

From the repository root:

```powershell
streamlit run .\src\EngTeamCrewSt\app.py
```

For Ollama, start Ollama locally first. The sidebar loads installed models from the configured Ollama base URL, and also lets you enter a model name manually if the local API is unavailable.

For OpenAI, enter an API key in the sidebar or set `OPENAI_API_KEY` before launching Streamlit.

Generated files are written to `output_streamlit` by default.
