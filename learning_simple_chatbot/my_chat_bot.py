import os
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

load_dotenv()

# Configure the page before displaying any Streamlit elements.
st.set_page_config(page_title="Groq Chat", page_icon="💬", layout="wide")

# Use Streamlit theme tokens so text stays readable in light and dark mode.
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at top, color-mix(in srgb, var(--primary-color) 8%, transparent), transparent 55%),
            var(--background-color);
    }
    .block-container {
        max-width: 1100px;
        padding: 2.5rem 2rem 6rem;
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid color-mix(in srgb, var(--text-color) 10%, transparent);
    }
    [data-testid="stSidebar"] > div:first-child {
        background: var(--secondary-background-color);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] *,
    [data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"],
    [data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"] * {
        color: var(--text-color) !important;
    }
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
        border-color: color-mix(in srgb, var(--text-color) 14%, transparent);
        background: transparent;
        text-align: left;
    }
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
        border-color: var(--primary-color);
        color: var(--primary-color);
    }
    [data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"] {
        color: var(--text-color) !important;
        opacity: 1 !important;
        border: 1px solid color-mix(in srgb, var(--text-color) 20%, transparent);
        background: var(--background-color);
    }
    [data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"]:hover {
        color: var(--text-color) !important;
        border-color: var(--primary-color);
        background: var(--secondary-background-color);
    }
    [data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"] svg {
        color: var(--text-color) !important;
        fill: currentColor;
        opacity: 1 !important;
    }
    [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
        color: #fff;
        border-color: #3346b8;
        background: #3346b8;
    }
    [data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover {
        border-color: #273894;
        background: #273894;
    }
    [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] *,
    [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] [data-testid="stMarkdownContainer"] * {
        color: #fff !important;
    }
    [data-testid="stSidebar"] input {
        background: var(--background-color);
        border-color: color-mix(in srgb, var(--text-color) 20%, transparent);
    }
    [data-testid="stChatMessage"] {
        border: 1px solid color-mix(in srgb, var(--text-color) 12%, transparent);
        border-radius: 18px;
        background: var(--secondary-background-color);
        padding: 1rem 1.15rem;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li {
        color: var(--text-color);
    }
    [data-testid="stChatInput"] textarea {
        color: var(--text-color);
        border-color: color-mix(in srgb, var(--text-color) 18%, transparent);
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: color-mix(in srgb, var(--text-color) 58%, transparent);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def create_chat():
    """Create a conversation and discard an unused blank draft."""
    active_id = st.session_state.get("active_chat_id")
    if active_id in st.session_state.chats:
        active_chat = st.session_state.chats[active_id]
        if not active_chat["messages"] and active_chat["title"] == "New chat":
            del st.session_state.chats[active_id]

    chat_id = str(uuid4())
    st.session_state.chats[chat_id] = {"title": "New chat", "messages": []}
    st.session_state.active_chat_id = chat_id
    st.session_state.pop("renaming_chat_id", None)


# Keep all conversations between Streamlit reruns and preserve older session data.
if "chats" not in st.session_state:
    old_messages = st.session_state.pop("messages", [])
    chat_id = str(uuid4())
    st.session_state.chats = {
        chat_id: {"title": "New chat", "messages": old_messages}
    }
    st.session_state.active_chat_id = chat_id
elif st.session_state.active_chat_id not in st.session_state.chats:
    create_chat()

with st.sidebar:
    st.title("💬 Groq Chat")
    st.caption("Your AI workspace")
    if st.button(
        "＋  New chat",
        key="new_chat",
        type="primary",
        use_container_width=True,
    ):
        create_chat()
        st.rerun()

    if st.button(
        "Clear chat",
        key="clear_chat",
        icon=":material/delete_sweep:",
        type="secondary",
        use_container_width=True,
        help="Remove all messages from the active chat to reduce its context.",
    ):
        active_chat = st.session_state.chats[st.session_state.active_chat_id]
        active_chat["messages"] = []
        active_chat["title"] = "New chat"
        active_chat["pinned"] = False
        st.session_state.pop("renaming_chat_id", None)
        st.rerun()

    st.subheader("Recent conversations")
    recent_chats = [
        (chat_id, chat)
        for chat_id, chat in reversed(list(st.session_state.chats.items()))
        if chat["messages"] or chat["title"] != "New chat"
    ]
    recent_chats.sort(key=lambda item: not item[1].get("pinned", False))
    for chat_id, chat in recent_chats:
        select_col, menu_col = st.columns([7, 1], vertical_alignment="center")
        with select_col:
            if st.button(
                f"{'📌 ' if chat.get('pinned') else ''}{chat['title']}",
                key=f"select_{chat_id}",
                type="primary" if chat_id == st.session_state.active_chat_id else "secondary",
                use_container_width=True,
            ):
                st.session_state.active_chat_id = chat_id
                st.session_state.pop("renaming_chat_id", None)
                st.rerun()
        with menu_col:
            with st.popover(
                " ",
                icon=":material/more_vert:",
                type="tertiary",
                key=f"chat_menu_{chat_id}",
                help="Chat options",
                use_container_width=True,
            ):
                if st.button(
                    "Rename",
                    icon=":material/edit:",
                    key=f"rename_{chat_id}",
                    use_container_width=True,
                ):
                    st.session_state.active_chat_id = chat_id
                    st.session_state.renaming_chat_id = chat_id
                    st.rerun()
                if st.button(
                    "Unpin" if chat.get("pinned", False) else "Pin",
                    icon=":material/push_pin:",
                    key=f"pin_{chat_id}",
                    use_container_width=True,
                ):
                    chat["pinned"] = not chat.get("pinned", False)
                    st.rerun()
                if st.button(
                    "Delete",
                    icon=":material/delete_outline:",
                    key=f"delete_{chat_id}",
                    use_container_width=True,
                ):
                    del st.session_state.chats[chat_id]
                    if st.session_state.get("renaming_chat_id") == chat_id:
                        st.session_state.pop("renaming_chat_id")
                    if chat_id == st.session_state.active_chat_id:
                        if st.session_state.chats:
                            st.session_state.active_chat_id = next(
                                reversed(st.session_state.chats)
                            )
                        else:
                            create_chat()
                    st.rerun()

    if not recent_chats:
        st.caption("Your saved conversations will appear here.")

    renaming_chat_id = st.session_state.get("renaming_chat_id")
    if renaming_chat_id in st.session_state.chats:
        renaming_chat = st.session_state.chats[renaming_chat_id]
        st.caption(f"Rename: {renaming_chat['title']}")
        with st.form(f"rename_form_{renaming_chat_id}", enter_to_submit=True):
            new_title = st.text_input(
                "Chat name",
                value=renaming_chat["title"],
                label_visibility="collapsed",
                placeholder="Enter a chat name",
            )
            save_name = st.form_submit_button("Save", use_container_width=True)
        if save_name:
            new_title = new_title.strip()
            if new_title:
                renaming_chat["title"] = new_title
                st.session_state.pop("renaming_chat_id", None)
                st.rerun()

active_chat = st.session_state.chats[st.session_state.active_chat_id]
if active_chat["messages"]:
    st.title(active_chat["title"])
st.caption("Ask anything. Your conversations stay in the sidebar for this session.")

# Read the API key from your local .env file.
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY is missing/expired. Add it to your .env file to continue.")
    st.stop()

# Show the saved conversation.
for message in active_chat["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Show the chat input at the bottom of the page.
prompt = st.chat_input("Message the assistant...")
if prompt:
    if not active_chat["messages"]:
        active_chat["title"] = " ".join(prompt.split())[:40] or "New chat"
    active_chat["messages"].append({"role": "user", "content": prompt})
    # Move the active conversation to the top of the recent list.
    st.session_state.chats[st.session_state.active_chat_id] = (
        st.session_state.chats.pop(st.session_state.active_chat_id)
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    # Convert saved messages into the format expected by LangChain.
    history = [SystemMessage(content="You are a helpful, clear assistant.")]
    for message in active_chat["messages"][:-1]:
        if message["role"] == "user":
            history.append(HumanMessage(content=message["content"]))
        else:
            history.append(AIMessage(content=message["content"]))
    history.append(HumanMessage(content=prompt))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            llm = ChatGroq(
                model="openai/gpt-oss-120b",
                api_key=api_key,
                temperature=0.7,
            )
            answer = llm.invoke(history).content
            st.markdown(answer)

    active_chat["messages"].append({"role": "assistant", "content": answer})