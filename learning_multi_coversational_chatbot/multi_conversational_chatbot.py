import sqlite3
# datetime and timezone create timestamps in UTC for chats and messages.
from datetime import datetime, timezone
# uuid4 creates a unique random ID for each new conversation.
from uuid import uuid4

# Streamlit creates the web interface and chat controls.
import streamlit as st
# load_dotenv reads secrets and configuration from a .env file.
from dotenv import load_dotenv
# ChatOpenAI is LangChain's wrapper for an OpenAI chat model.
from langchain_openai import ChatOpenAI
# HumanMessage and AIMessage identify user and assistant messages for LangChain.
from langchain_core.messages import AIMessage, HumanMessage
# ChatPromptTemplate builds the prompt and MessagesPlaceholder inserts history.
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Store all conversations and messages in this SQLite file.
DB_PATH = "chatbot.db"

# ---------- Database layer ----------
def get_connection():
    # Open chatbot.db, creating it automatically if it does not exist.
    conn = sqlite3.connect(DB_PATH)
    # Return rows by column name, such as chat["title"], instead of tuple indexes.
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    # The with block commits changes and closes the connection afterward.
    with get_connection() as conn:
        # Create the chats table; IF NOT EXISTS prevents duplicate-table errors.
        conn.execute("""CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )""")
        # Create the messages table and restrict role to user or assistant.
        conn.execute("""CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(chat_id) REFERENCES chats(id)
        )""")
        # Index chat_id and message id so a conversation's transcript loads quickly.
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id, id)")


def create_chat():
    # Generate a unique UUID string as the new chat's primary key.
    chat_id = str(uuid4())
    # Generate one UTC timestamp in ISO format for both chat timestamp columns.
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        # Insert the chat with the temporary title "New chat".
        conn.execute(
            "INSERT INTO chats(id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (chat_id, "New chat", now, now),
        )
    # Return the ID so the caller can make this chat active.
    return chat_id


def list_chats():
    with get_connection() as conn:
        # Get chat IDs and titles, ordered by most recently updated first.
        return conn.execute(
            "SELECT id, title FROM chats ORDER BY updated_at DESC"
        ).fetchall()


def get_messages(chat_id):
    with get_connection() as conn:
        # Get only role and content for this chat, oldest message first.
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id",
            (chat_id,),
        ).fetchall()
    # Convert SQLite Row objects into dictionaries for simpler Streamlit rendering.
    return [dict(row) for row in rows]


def save_message(chat_id, role, content):
    # Record the exact UTC time when this message is saved.
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        # Insert the message; placeholders keep the SQL safe and bind values correctly.
        conn.execute(
            "INSERT INTO messages(chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (chat_id, role, content, now),
        )
        # Updating updated_at makes this chat appear at the top of recent chats.
        conn.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (now, chat_id))


def update_title_from_first_message(chat_id, text):
    # Remove outer whitespace and line breaks, then keep at most 40 characters.
    title = text.strip().replace("\n", " ")[:40] or "New chat"
    with get_connection() as conn:
        # Replace the default title only once, never changing titles on later messages.
        conn.execute(
            "UPDATE chats SET title = ? WHERE id = ? AND title = 'New chat'",
            (title, chat_id),
        )

# ---------- Model and prompt ----------
# Load API credentials and settings from the .env file.
load_dotenv()
# Select gpt-5.5 and use temperature 0 for consistent, low-randomness replies.
llm = ChatOpenAI(model="gpt-5.5", temperature=0)
# Build the prompt: system behavior, recent history, then the new human message.
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer clearly and accurately."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])
# The pipe sends the formatted prompt through the model and produces a response.
chain = prompt | llm

# ---------- Streamlit app ----------
# Configure the browser title, icon, and page width.
st.set_page_config(page_title="Multi Chat AI", page_icon="💬", layout="wide")
# Display the app heading.
st.title("💬 Multi-conversation AI Chatbot")
# Initialize the database tables before reading or writing chat data.
init_db()
# Store the selected chat ID for this Streamlit session; None means "not selected".
st.session_state.setdefault("active_chat_id", None)

# All controls inside this block appear in the sidebar.
with st.sidebar:
    st.header("Conversations")
    # Create a chat, activate it, and rerun immediately when this button is clicked.
    if st.button("＋ New chat", use_container_width=True):
        st.session_state.active_chat_id = create_chat()
        st.rerun()

    st.caption("Recent chats")
    # Draw one button per saved conversation.
    for chat in list_chats():
        # Detect the currently open chat so a dot can mark it.
        is_active = chat["id"] == st.session_state.active_chat_id
        label = ("● " if is_active else "") + chat["title"]
        # Selecting a chat changes active_chat_id and reruns the page.
        if st.button(label, key=f"open_{chat['id']}", use_container_width=True):
            st.session_state.active_chat_id = chat["id"]
            st.rerun()

    st.divider()
    # Limit only the history sent to the model; all messages remain in SQLite.
    max_turns = st.selectbox(
        "Context turns sent to AI",
        options=[3, 5, 10, 15, 20],
        index=1,
        help="Only the latest N user/assistant turns are sent to the model. The full transcript remains saved.",
    )

# Create the first conversation automatically for a first-time visitor.
if st.session_state.active_chat_id is None:
    # Prefer the newest existing chat, or create one if the database is empty.
    chats = list_chats()
    st.session_state.active_chat_id = chats[0]["id"] if chats else create_chat()

# Store the active chat ID locally and load its complete saved transcript.
active_chat_id = st.session_state.active_chat_id
all_messages = get_messages(active_chat_id)

# Render the complete transcript, not just the limited LLM context.
for message in all_messages:
    # Choose the message style from the stored role and display Markdown content.
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# The walrus operator stores submitted text and also tests whether text was entered.
if user_text := st.chat_input("Message the assistant..."):
    # Convert only the latest N turns into LangChain messages.
    # Each normal turn consists of one user message and one assistant message.
    # Slice the list to keep at most max_turns × 2 individual messages.
    recent_context = all_messages[-(max_turns * 2):]
    # Convert database role/content dictionaries into LangChain message objects.
    history = [
        HumanMessage(content=m["content"]) if m["role"] == "user"
        else AIMessage(content=m["content"])
        for m in recent_context
    ]

    # Persist the new user message before calling the model.
    # This ensures the user message remains saved even if the model call fails.
    save_message(active_chat_id, "user", user_text)
    # A later message changes the title only if "New chat" is still the title.
    update_title_from_first_message(active_chat_id, user_text)
    # Immediately show the submitted user message in the chat UI.
    with st.chat_message("user"):
        st.markdown(user_text)

    try:
        # Render the response inside an assistant chat bubble.
        with st.chat_message("assistant"):
            # Show "Thinking..." until the model returns a response.
            with st.spinner("Thinking..."):
                # Send the current input and limited history through the prompt-and-model chain.
                response = chain.invoke({"input": user_text, "history": history})
            # Extract the plain text response from the LangChain response object.
            answer = response.content
            # Render the answer as Markdown.
            st.markdown(answer)
        # Save the successful assistant response to the active chat.
        save_message(active_chat_id, "assistant", answer)
    except Exception as exc:
        # Display the model error and clarify that the user message was already saved.
        st.error(f"The model request failed: {exc}")
        st.info("Your message is saved. You can try sending another message.")

    # Refresh the sidebar title/order and redraw the persisted transcript.
    st.rerun()