"""
Teammate Workspace UI - FastHTML Collaborative Canvas
The "Desk" - A three-panel interface for human-AI collaboration.
"""
# ruff: noqa: F403, F405
import os
from fasthtml.common import *
import uuid

# Tailwind CSS via CDN
head = Title("Teammate Workspace"), Script(src="https://cdn.tailwindcss.com"), Script(src="https://unpkg.com/htmx.org@1.9.10")

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
    """Main application layout."""
    thread_id = str(uuid.uuid4())
    return Body(
        # Header
        Div(
            Div(
                Span("Teammate", cls="font-bold text-white text-lg"),
                Span("Workspace", cls="font-light text-gray-400 text-lg"),
            ),
            Div(
                Span("Connected", cls="text-xs text-emerald-400 bg-emerald-400/10 px-2 py-1 rounded-full"),
            ),
            cls="flex items-center justify-between px-6 py-4 border-b border-slate-700 bg-slate-800/50"
        ),

        # Main content - Three panel layout
        Div(
            # Left Panel - Stats
            Div(
                left_panel(thread_id, 0.0, "Healthy"),
                cls="w-64 flex-shrink-0"
            ),

            # Center Panel - Canvas
            Div(
                center_panel([], thread_id),
                cls="flex-1 min-w-0"
            ),

            # Right Panel - Thought Trace
            Div(
                right_panel([]),
                cls="w-80 flex-shrink-0"
            ),

            cls="flex gap-4 p-6 min-h-[calc(100vh-73px)] bg-slate-900"
        ),

        # Intervention Panel (initially hidden)
        intervention_panel(False, thread_id),

        # SSE Script for real-time updates
        Script("""
            const BRAIN_URL = "%s";
            const eventSource = new EventSource(BRAIN_URL + '/chat/stream');
            let currentThreadId = null;

            eventSource.addEventListener('thread_id', (e) => {
                currentThreadId = e.data;
                const statValues = document.querySelectorAll('[class*="text-2xl font-bold"]');
                if (statValues[0]) statValues[0].textContent = currentThreadId.slice(0, 8);
            });

            eventSource.addEventListener('internal_monologue', (e) => {
                const data = JSON.parse(e.data);
                const thoughtTrace = document.getElementById('thought-trace');
                const newThought = document.createElement('div');
                newThought.className = 'border-l-2 border-purple-500/50 pl-3 py-1';
                newThought.innerHTML = `
                    <div class="flex items-center">
                        <span class="text-xs font-medium text-purple-400">${data.node || 'Thinking'}</span>
                        <span class="text-xs text-gray-500 ml-auto">${data.cost || 0}</span>
                    </div>
                    <div class="mt-1 text-sm text-gray-400">${data.thought || ''}</div>
                `;
                thoughtTrace.insertBefore(newThought, thoughtTrace.firstChild);
            });

            eventSource.addEventListener('message', (e) => {
                const data = JSON.parse(e.data);
                const canvas = document.getElementById('canvas-messages');
                const newMsg = document.createElement('div');
                newMsg.className = 'p-4 rounded-lg ' + (data.role === 'user' ? 'bg-emerald-900/20 border border-emerald-700/30' : 'bg-slate-800/50 border border-slate-700/50');
                newMsg.innerHTML = `
                    <div class="flex items-center">
                        <span class="font-semibold ${data.role === 'user' ? 'text-emerald-400' : 'text-cyan-400'}">${data.role === 'user' ? 'You' : 'Teammate'}</span>
                        <span class="text-xs text-gray-500 ml-2">${data.timestamp || ''}</span>
                    </div>
                    <div class="mt-1 text-gray-300 text-sm leading-relaxed">${data.content}</div>
                `;
                canvas.appendChild(newMsg);
                canvas.scrollTop = canvas.scrollHeight;
            });

            eventSource.addEventListener('cost_update', (e) => {
                const data = JSON.parse(e.data);
                const statValues = document.querySelectorAll('[class*="text-2xl font-bold"]');
                if (statValues[1]) statValues[1].textContent = '$' + data.total_cost.toFixed(4);
            });

            eventSource.addEventListener('budget_warning', (e) => {
                const data = JSON.parse(e.data);
                const statValues = document.querySelectorAll('[class*="text-2xl font-bold"]');
                if (statValues[2]) {
                    statValues[2].textContent = data.status;
                    statValues[2].className = 'text-2xl font-bold ' + (data.status === 'Healthy' ? 'text-emerald-400' : data.status === 'Warning' ? 'text-amber-400' : 'text-rose-400');
                }
            });

            eventSource.addEventListener('interrupt', (e) => {
                const data = JSON.parse(e.data);
                const panel = document.getElementById('intervention-panel');
                panel.innerHTML = `
                    <div class="flex items-start">
                        <div class="flex-1">
                            <div class="text-amber-400 font-semibold text-sm">Human Intervention Required</div>
                            <div class="text-gray-400 text-xs mt-1">${data.reason || 'An action requires your approval before proceeding.'}</div>
                        </div>
                    </div>
                    <form class="mt-3 flex" hx-post="` + BRAIN_URL + `/chat/${data.thread_id}/approve" hx-target="#intervention-panel">
                        <button type="submit" class="bg-emerald-600 hover:bg-emerald-500 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors mr-2">Approve</button>
                        <button type="button" class="bg-rose-600 hover:bg-rose-500 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors">Deny</button>
                    </form>
                `;
                panel.classList.remove('hidden');
            });

            eventSource.addEventListener('resume', (e) => {
                document.getElementById('intervention-panel').classList.add('hidden');
            });

            eventSource.onerror = () => {
                console.log('SSE connection error - ensure brain is running on port 8000');
            };
        """ % BRAIN_URL),

        # Dark theme base styles
        cls="bg-slate-900 min-h-screen text-gray-300 font-sans antialiased"
    )

# Application entry
app, rt = fast_app(
    hdrs=head,
    pico=False,
    exit_on_close=False
)

@rt("/")
def get():
    return main_page()

@rt("/chat/{thread_id}/approve", methods=["POST"])
def approve(thread_id: str):
    return Div("Approved", cls="text-emerald-400 text-sm font-medium")

@rt("/chat/{thread_id}/deny", methods=["POST"])
def deny(thread_id: str):
    return Div("Denied", cls="text-rose-400 text-sm font-medium")

serve()
