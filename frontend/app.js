const STORAGE_KEY = "rag_chat_session";
const HISTORY_KEY = "rag_chat_history";

const chatDisplay = document.getElementById("chatDisplay");
const chatForm = document.getElementById("chatForm");
const messageInput = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const loadingIndicator = document.getElementById("loadingIndicator");
const newChatBtn = document.getElementById("newChatBtn");

function getSessionId() {
  let id = localStorage.getItem(STORAGE_KEY);
  if (!id) {
    id = "sess_" + Math.random().toString(36).slice(2, 12);
    localStorage.setItem(STORAGE_KEY, id);
  }
  return id;
}

function loadHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveHistory(messages) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(messages));
}

function formatTime(date) {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function appendMessage(role, text, meta = "") {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.textContent = text;
  if (meta) {
    const span = document.createElement("span");
    span.className = "message-meta";
    span.textContent = meta;
    div.appendChild(span);
  }
  chatDisplay.appendChild(div);
  chatDisplay.scrollTop = chatDisplay.scrollHeight;
}

function renderHistory() {
  chatDisplay.innerHTML = "";
  const history = loadHistory();
  if (history.length === 0) {
    const welcome = document.createElement("p");
    welcome.className = "welcome";
    welcome.textContent =
      "Ask anything covered in the knowledge base — e.g. “How can I reset my password?”";
    chatDisplay.appendChild(welcome);
    return;
  }
  history.forEach((msg) => {
    appendMessage(msg.role, msg.text, msg.meta || "");
  });
}

function setLoading(active) {
  loadingIndicator.classList.toggle("hidden", !active);
  loadingIndicator.setAttribute("aria-hidden", String(!active));
  sendBtn.disabled = active;
  messageInput.disabled = active;
}

async function sendMessage(text) {
  const sessionId = getSessionId();
  const history = loadHistory();
  const now = formatTime(new Date());

  history.push({ role: "user", text, meta: now });
  saveHistory(history);
  appendMessage("user", text, now);

  setLoading(true);

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sessionId, message: text }),
    });

    const data = await res.json();

    if (!res.ok) {
      const errText = data.error || "Something went wrong.";
      appendMessage("error", errText);
      return;
    }

    const meta = [
      formatTime(new Date()),
      data.retrievedChunks > 0 ? `${data.retrievedChunks} chunks · ${data.tokensUsed} tokens` : "no retrieval",
    ].join(" · ");

    history.push({ role: "assistant", text: data.reply, meta });
    saveHistory(history);
    appendMessage("assistant", data.reply, meta);
  } catch (err) {
    appendMessage("error", "Network error. Is the backend running?");
  } finally {
    setLoading(false);
    messageInput.focus();
  }
}

chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = messageInput.value.trim();
  if (!text) return;
  messageInput.value = "";
  sendMessage(text);
});

messageInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    chatForm.requestSubmit();
  }
});

newChatBtn.addEventListener("click", () => {
  localStorage.removeItem(STORAGE_KEY);
  localStorage.removeItem(HISTORY_KEY);
  getSessionId();
  renderHistory();
  messageInput.focus();
});

renderHistory();
messageInput.focus();
