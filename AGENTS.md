# AI Coding Configuration

This project was developed with an AI coding workflow that uses the OpenAI Developer Documentation MCP server for current OpenAI API, Codex, plugin, and tool guidance.

When an AI coding agent needs OpenAI product or API information, it should use the `openaiDeveloperDocs` MCP server before relying on general web sources. The server is read-only and does not require a project secret.

The application itself does not depend on MCP or Codex at runtime. It runs with Docker using only the variables documented in `.env.example`.
