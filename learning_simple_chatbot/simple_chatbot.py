import streamlit as st

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


st.title("Chatbot AI")

# Load the API settings from the environment, then create the model client.
load_dotenv()  # Load environment variables from .env file
llm = ChatOpenAI(model_name="gpt-5.5")

# Build the conversation prompt from its fixed instructions, earlier turns,
# and the new user input supplied for each request.
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | llm

# Keep messages in Streamlit session state so they survive reruns in this tab
# without being shared with other users or browser tabs.
st.session_state.setdefault("chat_history", [])

# Replay every saved turn so the complete conversation remains visible.
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# A chat input returns a value only when the user submits a new message.
if user_input := st.chat_input("Enter your message:"):
    # Convert saved display messages back into LangChain message objects so
    # the model receives the prior conversation as context.
    history = [
        HumanMessage(content=message["content"])
        if message["role"] == "user"
        else AIMessage(content=message["content"])
        for message in st.session_state.chat_history
    ]

    # Save and display the new user turn before requesting the model reply.
    st.session_state.chat_history.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate a reply using the previous turns, display it, and save it for
    # the next request and for display after the next Streamlit rerun.
    with st.chat_message("assistant"):
        # Show a loading indicator while waiting for the model response.
        with st.spinner("Preparing an answer..."):
            output = chain.invoke({"input": user_input, "history": history})
        st.markdown(output.content)
    st.session_state.chat_history.append(
        {"role": "assistant", "content": output.content}
    )

