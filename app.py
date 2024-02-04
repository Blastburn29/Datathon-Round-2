import streamlit as st
from logic import logicOfLLM

def main():
    st.set_page_config("Air India AI")
    st.header("Airlines Rules and Regulations.")
    st.write("Ask your doubts below.")
    user_question = st.text_input("Ask a Question")

    with st.spinner("Processing..."):
        chain = logicOfLLM()
    
    if user_question:
        with st.spinner("Processing..."):
            result = chain.invoke({"query": user_question})
            st.write("Reply: ", result["result"])


if __name__ == "__main__":
    main()