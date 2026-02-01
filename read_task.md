Focus on fix UI Performance Issue. Explore whole codebase, not just app.py.
Issue:
  - The UI still feels slow because it waits for the full liaison response (11s) before showing the user message
  - This is a frontend design issue: The UI uses HTMX form submission that blocks until the backend completes
  - Assistant messages arrive via SSE but user message appears only after backend response

Solution requirements:
use FastHTML + HTMX not streamlit.
SSE handles the "Teammate's Voice" (the stream of thoughts).
WebSockets handle the "Canvas" (the shared document state).
This allows you to see the Agent's reasoning on the left (SSE) while it simultaneously edits a shared report on the right (WebSocket/CRDT).

Suggestion for reference:
HTMX form submissions naturally wait for a server response to update the DOM, creating a synchronous feel. To prevent the UI from appearing frozen during long backend processes, use the hx-indicator attribute to show loading spinners and htmx.config.defaultSwapStyle to manage content replacement, ensuring a responsive user experience. 


update CLAUDE.md
always add log when add code