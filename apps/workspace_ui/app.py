"""
Teammate Workspace UI - FastHTML Collaborative Canvas
The "Desk" - A three-panel interface for human-AI collaboration.
"""
# ruff: noqa: F403, F405
import os
import uuid
import httpx
from fasthtml.common import *

# Advanced 2026 Enterprise CSS Styling
hdrs = (
    # Tailwind CSS for rapid, modern styling
    Script(src="https://cdn.tailwindcss.com"),
    # Lucide Icons for clean, professional iconography
    Script(src="https://unpkg.com/lucide@latest"),
    # HTMX for reactive updates
    Script(src="https://unpkg.com/htmx.org@1.9.10"),
    Script(src="https://unpkg.com/htmx-ext-sse@2.2.0"),
    # Custom CSS for the "Glassmorphism" and sidebar effects
    Style("""
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Fira+Code:wght@400;500&display=swap');

        :root { --bg-dark: #0a0a0c; --panel-bg: #111114; --accent-blue: #3b82f6; --text-muted: #94a3b8; }
        body { background-color: var(--bg-dark); color: #f8fafc; font-family: 'Inter', sans-serif; }
        .glass-panel { background: var(--panel-bg); border: 1px solid #1f1f23; border-radius: 12px; }
        .thought-trace { font-family: 'Fira Code', monospace; font-size: 0.85rem; color: #10b981; }
        .pulse-dot { height: 8px; width: 8px; background-color: var(--accent-blue); border-radius: 50%; display: inline-block; box-shadow: 0 0 8px var(--accent-blue); animation: pulse 2s infinite; }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(59, 130, 246, 0); }
            100% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0); }
        }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 10px; }
        .input-glow:focus-within { box-shadow: 0 0 20px rgba(59, 130, 246, 0.15); }
        /* HTMX loading indicators */
        .htmx-indicator { opacity: 0; transition: opacity 200ms ease-in; }
        .htmx-request .htmx-indicator { opacity: 1; }
    """)
)

# Configuration
BRAIN_URL = os.getenv("BRAIN_URL", "http://liaison:8000")

def card(title: str, content, id=None):
    """Create a styled card component."""
    return Div(
        Div(title, cls="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2"),
        content,
        cls="bg-slate-800 border border-slate-700 rounded-lg p-4",
        id=id
    )

def stat_value(value: str, label: str, color: str = "text-cyan-400"):
    """Create a stat display."""
    return Div(
        Div(value, cls=f"text-2xl font-bold {color}"),
        Div(label, cls="text-xs text-gray-500"),
        cls="bg-slate-900/50 rounded-lg p-3"
    )

def left_panel(thread_id: str, cost: float, budget_status: str):
    """Left panel - The Stats Dashboard."""
    budget_color = "text-emerald-400" if budget_status == "Healthy" else "text-amber-400" if budget_status == "Warning" else "text-rose-400"
    return card(
        "Session Stats",
        Div(
            stat_value(thread_id[:8], "Thread ID"),
            Div(cls="h-3"),
            stat_value(f"${cost:.4f}", "Total Cost", "text-emerald-400"),
            Div(cls="h-3"),
            stat_value(budget_status, "Budget", budget_color),
            cls="space-y-3"
        )
    )

def center_panel(messages: list, thread_id: str):
    """Center panel - The Canvas (final scrubbed output)."""
    message_items = []
    for msg in messages:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        timestamp = msg.get("timestamp", "")
        is_user = role == "user"
        message_items.append(
            Div(
                Div(
                    Div("You" if is_user else "Teammate", cls=f"font-semibold { 'text-emerald-400' if is_user else 'text-cyan-400'}"),
                    Div(timestamp, cls="text-xs text-gray-500 ml-2"),
                    cls="flex items-center"
                ),
                Div(content, cls="mt-1 text-gray-300 text-sm leading-relaxed"),
                cls=f"p-4 rounded-lg {'bg-slate-800/50 border border-slate-700/50' if not is_user else 'bg-emerald-900/20 border border-emerald-700/30'}"
            )
        )

    return card(
        "The Canvas",
        Div(
            *message_items,
            id="canvas-messages",
            cls="space-y-3 max-h-[600px] overflow-y-auto pr-2"
        ),
        id="canvas-container"
    )

def right_panel(thoughts: list):
    """Right panel - The Thought Trace (INTERNAL_MONOLOGUE)."""
    thought_items = [
        Div(
            Div(
                Span(t.get("node", "Thinking"), cls="text-xs font-medium text-purple-400"),
                Span(f"{t.get('cost', 0):.4f}", cls="text-xs text-gray-500 ml-auto"),
            ),
            Div(t.get("thought", ""), cls="mt-1 text-sm text-gray-400"),
            cls="border-l-2 border-purple-500/50 pl-3 py-1"
        )
        for t in thoughts
    ]

    return card(
        "Thought Trace",
        Div(
            *thought_items,
            id="thought-trace",
            cls="space-y-2 max-h-[600px] overflow-y-auto pr-2"
        )
    )

def intervention_panel(visible: bool = False, thread_id: str = ""):
    """Human-in-the-loop intervention panel."""
    if not visible:
        return Div(id="intervention-panel")

    return Div(
        Div(
            Div("Human Intervention Required", cls="text-amber-400 font-semibold text-sm"),
            Div("An action requires your approval before proceeding.", cls="text-gray-400 text-xs mt-1"),
        ),
        Form(
            Button("Approve", hx_post=f"{BRAIN_URL}/chat/{thread_id}/approve", hx_target="#intervention-panel", cls="bg-emerald-600 hover:bg-emerald-500 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors mr-2"),
            Button("Deny", hx_post=f"{BRAIN_URL}/chat/{thread_id}/deny", hx_target="#intervention-panel", cls="bg-rose-600 hover:bg-rose-500 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"),
            method="post",
            cls="mt-3 flex"
        ),
        id="intervention-panel",
        cls="fixed bottom-6 left-1/2 transform -translate-x-1/2 bg-slate-800 border border-amber-500/50 rounded-xl shadow-2xl shadow-amber-500/20 p-4 z-50 max-w-md w-full mx-4"
    )

def main_page():
    """Main application layout - 3-Pane Canvas Design."""
    thread_id = str(uuid.uuid4())
    return Title("Teammate Workspace"), Body(
        Main(cls="flex h-screen w-full p-4 gap-4 bg-[#0a0a0c]")(
            # PANEL 1: THE STATS (Left) - 20% width
            Div(cls="w-1/5 glass-panel p-4 flex flex-col gap-6")(
                # Header with pulse dot
                Div(cls="flex items-center gap-2")(
                    Span(cls="pulse-dot"),
                    H2("Liaison Active", cls="text-sm font-semibold text-white")
                ),
                # Stats content
                Div(cls="space-y-4")(
                    Div()(P("Budget Usage", cls="text-xs text-muted uppercase tracking-wider"), P("$0.42 / $1.00", cls="text-lg font-mono text-emerald-400")),
                    Div()(P("Current Specialist", cls="text-xs text-muted uppercase tracking-wider"), P("Research Agent", cls="text-md font-medium text-blue-400")),
                    Div()(P("Session ID", cls="text-xs text-muted uppercase tracking-wider"), P(thread_id[:8], cls="text-sm font-mono text-gray-400"))
                ),
                # Connection status
                Div(cls="mt-auto pt-4 border-t border-gray-800")(
                    Div(cls="flex items-center gap-2")(
                        Span(cls="w-2 h-2 bg-emerald-500 rounded-full"),
                        Span("Connected", cls="text-xs text-emerald-400")
                    )
                )
            ),

            # PANEL 2: THE CANVAS (Center) - 60% width
            Div(cls="w-3/5 flex flex-col gap-4")(
                # Canvas area
                Div(id="canvas-area", cls="flex-grow glass-panel p-6 overflow-y-auto")(
                    P("Awaiting task...", cls="text-muted italic")
                ),
                # Input Area
                Form(
                    hx_post="/chat",
                    hx_target="#canvas-area",
                    hx_swap="none",
                    hx_indicator="#loading-spinner",
                    cls="input-glow glass-panel p-2 relative"
                )(
                    Input(type="hidden", name="thread_id", value=thread_id),
                    Input(
                        name="msg",
                        placeholder="Describe the task...",
                        cls="w-full bg-transparent border-none rounded-lg p-3 focus:outline-none text-gray-300 placeholder-gray-600"
                    ),
                    # Loading spinner
                    Div(id="loading-spinner", cls="htmx-indicator absolute right-3 top-1/2 transform -translate-y-1/2")(
                        Div(cls="animate-spin rounded-full h-5 w-5 border-b-2 border-emerald-400")
                    )
                )
            ),

            # PANEL 3: THE THOUGHT TRACE (Right) - 20% width
            Div(cls="w-1/5 glass-panel p-4 flex flex-col")(
                H3("Thought Trace", cls="text-xs font-bold uppercase tracking-widest text-muted mb-4"),
                Div(id="thought-stream",
                    cls="thought-trace flex-grow overflow-y-auto space-y-2",
                    hx_ext="sse",
                    sse_connect=f"/chat/stream/{thread_id}",
                    sse_swap="thought")(
                    P("Thinking...", cls="text-gray-500 text-xs")
                )
            )
        ),

        # SSE Script for real-time updates
        Script("""
            // Optimistic UI: Handle HTMX form submission for immediate feedback
            const messageForm = document.querySelector('form[hx-post="/chat"]');
            if (messageForm) {
                messageForm.addEventListener('htmx:beforeRequest', (e) => {
                    const formData = new FormData(messageForm);
                    const msg = formData.get('msg');
                    if (!msg?.trim()) return;

                    const canvas = document.getElementById('canvas-area');
                    const msgId = 'msg-' + Date.now();
                    const html = `
                        <div id="${msgId}" class="p-4 rounded-lg bg-emerald-900/20 border border-emerald-700/30 mb-4">
                            <div class="flex items-center">
                                <span class="font-semibold text-emerald-400">You</span>
                                <span class="ml-2 text-xs text-gray-500">Sending...</span>
                                <div class="ml-auto htmx-indicator">
                                    <div class="animate-spin rounded-full h-4 w-4 border-b-2 border-emerald-400"></div>
                                </div>
                            </div>
                            <div class="mt-1 text-gray-300 text-sm">${msg}</div>
                        </div>
                    `;
                    canvas.insertAdjacentHTML('beforeend', html);
                    canvas.scrollTop = canvas.scrollHeight;

                    e.detail.requestConfig.headers['X-Message-Id'] = msgId;
                });

                messageForm.addEventListener('htmx:afterRequest', (e) => {
                    if (e.detail.successful) {
                        const msgId = e.detail.requestConfig.headers['X-Message-Id'];
                        const elem = document.getElementById(msgId);
                        if (elem) {
                            elem.querySelector('.text-gray-500').remove();
                            elem.querySelector('.htmx-indicator').remove();
                        }
                    }
                });

                messageForm.addEventListener('htmx:responseError', (e) => {
                    const msgId = e.detail.requestConfig.headers['X-Message-Id'];
                    const elem = document.getElementById(msgId);
                    if (elem) {
                        elem.querySelector('.text-emerald-400').textContent = 'Error';
                        elem.querySelector('.text-gray-500').textContent = 'Failed - try again';
                        elem.className = 'p-4 rounded-lg bg-rose-900/20 border border-rose-700/30 mb-4';
                    }
                });
            }

            const BRAIN_URL = "%s";
            const threadId = "%s";
            const eventSource = new EventSource('/chat/stream/' + threadId);
            let currentThreadId = threadId;

            eventSource.addEventListener('thread_id', (e) => {
                currentThreadId = e.data;
            });

            // Listen for 'thought' events (internal monologue from LLM)
            eventSource.addEventListener('thought', (e) => {
                const data = JSON.parse(e.data);
                const thoughtTrace = document.getElementById('thought-stream');
                const newThought = document.createElement('div');
                newThought.className = 'border-l-2 border-purple-500/50 pl-3 py-1';
                newThought.innerHTML = `<div class="flex items-center"><span class="text-xs font-medium text-purple-400">${data.node || 'Thinking'}</span></div><div class="mt-1 text-sm text-gray-400">${data.content || ''}</div>`;
                thoughtTrace.insertBefore(newThought, thoughtTrace.firstChild);
            });

            // Listen for 'message' events (assistant responses)
            eventSource.addEventListener('message', (e) => {
                const data = JSON.parse(e.data);
                const canvas = document.getElementById('canvas-area');
                const newMsg = document.createElement('div');
                newMsg.className = 'mb-4 p-4 rounded-lg bg-slate-800/50 border border-slate-700/50';
                newMsg.innerHTML = `<div class="flex items-center"><span class="font-semibold text-cyan-400">Teammate</span></div><div class="mt-1 text-gray-300 text-sm leading-relaxed">${data.content || ''}</div>`;
                canvas.appendChild(newMsg);
                canvas.scrollTop = canvas.scrollHeight;
            });

            // Listen for 'cost' events (cost updates)
            eventSource.addEventListener('cost', (e) => {
                const data = JSON.parse(e.data);
                console.log('Cost update:', data);
                // Update cost display if element exists
                const costEl = document.getElementById('cost-display');
                if (costEl) {
                    costEl.textContent = `$${data.total_cost.toFixed(4)}`;
                }
            });

            // Listen for 'intervene' events (budget/PII approval requests)
            eventSource.addEventListener('intervene', (e) => {
                const data = JSON.parse(e.data);
                console.log('Intervention required:', data);
                // Show intervention modal or notification
                alert(`Intervention required: ${data.reason}. Current cost: $${data.current_cost}`);
            });

            // Listen for 'error' events
            eventSource.addEventListener('error', (e) => {
                const data = JSON.parse(e.data);
                console.error('Error from liaison:', data);
                const canvas = document.getElementById('canvas-area');
                const errorMsg = document.createElement('div');
                errorMsg.className = 'mb-4 p-4 rounded-lg bg-rose-900/20 border border-rose-700/30';
                errorMsg.innerHTML = `<div class="flex items-center"><span class="font-semibold text-rose-400">Error</span></div><div class="mt-1 text-gray-300 text-sm leading-relaxed">${data.message || 'Unknown error'}</div>`;
                canvas.appendChild(errorMsg);
                canvas.scrollTop = canvas.scrollHeight;
            });

            // Listen for 'result' events (final output) - kept for backward compatibility
            eventSource.addEventListener('result', (e) => {
                const data = JSON.parse(e.data);
                const outputContent = data.output ? (data.output.content || JSON.stringify(data.output)) : '';
                const canvas = document.getElementById('canvas-area');
                const newMsg = document.createElement('div');
                newMsg.className = 'mb-4 p-4 rounded-lg bg-slate-800/50 border border-slate-700/50';
                newMsg.innerHTML = `<div class="flex items-center"><span class="font-semibold text-cyan-400">Teammate</span></div><div class="mt-1 text-gray-300 text-sm leading-relaxed">${formatContent(outputContent)}</div>`;
                canvas.appendChild(newMsg);
                canvas.scrollTop = canvas.scrollHeight;
            });

            eventSource.onerror = (e) => {
                console.log('SSE connection error', e);
            };
        """ % (BRAIN_URL, thread_id))
    )

# Application entry
app, rt = fast_app(
    hdrs=hdrs,
    pico=False,
    exit_on_close=False
)

@rt("/")
def get():
    return main_page()

@rt("/chat", methods=["POST"])
async def chat_post(msg: str, thread_id: str = None):
    """Forward user message to the Liaison Brain asynchronously.

    Returns immediately (< 200ms) while starting graph execution in background.
    Optimistic UI already shows user message; errors are communicated via SSE.
    """
    from starlette.responses import HTMLResponse
    import asyncio
    import socket

    # Use thread_id from form or generate new one
    thread_id = thread_id or str(uuid.uuid4())

    # Determine brain host (try Docker service name, fallback to bridge IP)
    brain_host = "liaison"
    try:
        socket.gethostbyname("liaison")
    except socket.gaierror:
        brain_host = "172.17.0.1"  # Default Docker bridge IP

    brain_url = f"http://{brain_host}:8000"

    async def forward_to_brain():
        """Background task to forward message to liaison brain."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{brain_url}/chat/input",
                    json={
                        "message": msg,
                        "thread_id": thread_id,
                        "task_description": None,
                        "task_budget": 1000,
                        "cost_limit": 0.50
                    }
                )
                response.raise_for_status()
                data = response.json()
                # thread_id could be updated by brain (though unlikely)
                updated_thread_id = data.get("thread_id", thread_id)
                if updated_thread_id != thread_id:
                    # Log thread_id change (rare)
                    print(f"[chat_post] Brain returned new thread_id: {updated_thread_id}")
        except Exception as e:
            # Log error but don't surface to user (SSE will handle streaming errors)
            print(f"[chat_post] Background task failed to forward to brain: {type(e).__name__}: {e}")
            # Note: SSE stream may fail separately; error will be shown via SSE error events

    # Quick connectivity check (non-blocking, but still synchronous)
    # This catches immediate network errors while keeping response time low
    try:
        # Try to resolve host and create a socket connection (quick check)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)  # 1 second timeout for connectivity check
        host_ip = socket.gethostbyname(brain_host)
        sock.connect((host_ip, 8000))
        sock.close()
    except (socket.gaierror, socket.timeout, ConnectionRefusedError, OSError) as e:
        # Immediate connectivity issue - return error to update optimistic UI
        return HTMLResponse(f'''
        <div class="p-4 rounded-lg bg-rose-900/20 border border-rose-700/30">
          <div class="flex items-center">
            <div class="font-semibold text-rose-400">Error</div>
          </div>
          <div class="mt-1 text-gray-300 text-sm leading-relaxed">Cannot connect to teammate service: {str(e)}</div>
        </div>
        ''')

    # Start background task for actual brain communication
    asyncio.create_task(forward_to_brain())

    # Return empty response immediately - UI already shows message optimistically
    return HTMLResponse("")

@rt("/chat/stream/{thread_id}")
def chat_stream(thread_id: str):
    """Proxy SSE stream from Liaison Brain to the UI."""
    import asyncio

    # Use Docker IP or localhost
    brain_host = "liaison"
    try:
        import socket
        socket.gethostbyname("liaison")
    except socket.gaierror:
        brain_host = "172.17.0.1"  # Default Docker bridge IP

    brain_url = f"http://{brain_host}:8000"

    async def event_generator():
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "GET",
                    f"{brain_url}/chat/stream/{thread_id}",
                    timeout=None
                ) as response:
                    # SSE requires exact byte passthrough - don't split lines
                    async for chunk in response.aiter_bytes():
                        if chunk:
                            yield chunk
        except asyncio.CancelledError:
            # Client disconnected gracefully
            yield None
        except Exception as e:
            yield f"event: error\ndata: {str(e)}\n\n".encode()

    # Return SSE response with proper streaming headers
    from starlette.responses import StreamingResponse
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

@rt("/chat/{thread_id}/approve", methods=["POST"])
def approve(thread_id: str):
    return Div("Approved", cls="text-emerald-400 text-sm font-medium")

@rt("/chat/{thread_id}/deny", methods=["POST"])
def deny(thread_id: str):
    return Div("Denied", cls="text-rose-400 text-sm font-medium")

serve(port=5001)
