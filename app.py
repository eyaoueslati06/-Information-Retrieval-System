import streamlit as st

from src.helper import (
    get_pdf_text,
    get_text_chunks,
    get_vector_store,
    answer_question
)


# --------------------------------------------------
# MAIN APP
# --------------------------------------------------

def main():

    st.set_page_config(
        page_title="Information Retrieval",
        page_icon="🔎",
        layout="wide"
    )

    st.header(
        "Information Retrieval System 💁‍♀️"
    )


    # --------------------------------------------------
    # SESSION STATE
    # --------------------------------------------------

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "messages" not in st.session_state:
        st.session_state.messages = []


    # --------------------------------------------------
    # DISPLAY FULL CHAT HISTORY
    # --------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    # --------------------------------------------------
    # USER QUESTION
    # --------------------------------------------------

    user_question = st.chat_input(
        "Ask a question from the PDF files"
    )


    if user_question:

        if st.session_state.vector_store is None:

            st.warning(
                "Please upload and process a PDF first."
            )

        else:

            # Save user message
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": user_question
                }
            )


            try:

                with st.spinner(
                    "Thinking..."
                ):

                    answer = answer_question(
                        st.session_state.vector_store,
                        user_question,
                        st.session_state.messages[:-1]
                    )


                # Save assistant answer
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )


            except Exception as e:

                print(e)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content":
                        "A temporary Google API error occurred. "
                        "Please try your question again."
                    }
                )


            # Rerun so the full history appears
            st.rerun()


    # --------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------

    with st.sidebar:

        st.title(
            "Menu:"
        )


        pdf_docs = st.file_uploader(
            "Upload your PDF files",
            accept_multiple_files=True,
            type=["pdf"],
            key="pdf_uploader"
        )


        if st.button(
            "Submit & Process",
            key="process_button"
        ):

            if not pdf_docs:

                st.warning(
                    "Please upload at least one PDF."
                )

            else:

                try:

                    with st.spinner(
                        "Processing..."
                    ):

                        # ----------------------------------
                        # Extract text
                        # ----------------------------------

                        raw_text = get_pdf_text(
                            pdf_docs
                        )


                        if not raw_text.strip():

                            st.error(
                                "No readable text was found "
                                "in the PDF."
                            )

                            return


                        # ----------------------------------
                        # Split text
                        # ----------------------------------

                        text_chunks = (
                            get_text_chunks(
                                raw_text
                            )
                        )


                        # ----------------------------------
                        # Create FAISS
                        # ----------------------------------

                        vector_store = (
                            get_vector_store(
                                text_chunks
                            )
                        )


                        # Save vector store
                        st.session_state.vector_store = (
                            vector_store
                        )


                        # New PDF = new chat
                        st.session_state.messages = []


                    st.success(
                        "Processing completed!"
                    )


                except Exception as e:

                    st.error(
                        f"Processing error: {e}"
                    )


# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":
    main()