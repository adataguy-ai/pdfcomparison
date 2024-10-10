import streamlit as st
import tempfile
import os
import base64
from pdf_compare import EnhancedPDFComparer


def main():
    st.set_page_config(layout="wide")
    st.title("Enhanced PDF Comparison Tool")

    # Initialize session state
    if "comparison_done" not in st.session_state:
        st.session_state.comparison_done = False
    if "annotated_pdf" not in st.session_state:
        st.session_state.annotated_pdf = None
    if "change_log" not in st.session_state:
        st.session_state.change_log = None
    if "original_pdf" not in st.session_state:
        st.session_state.original_pdf = None

    # Check if comparison has been done
    if not st.session_state.comparison_done:
        st.write("Upload two PDF files to compare and analyze the differences.")

        # File uploaders
        original_pdf = st.file_uploader("Upload the original PDF", type="pdf")
        modified_pdf = st.file_uploader("Upload the modified PDF", type="pdf")

        if original_pdf and modified_pdf:
            if st.button("Compare PDFs"):
                with st.spinner("Comparing PDFs... This may take a moment."):
                    # Perform comparison
                    compare_pdfs(original_pdf, modified_pdf)
                    st.session_state.comparison_done = True
                st.rerun()

    else:
        # Display results
        st.subheader("Comparison Results")

        # Create two columns for side-by-side display
        col1, col2 = st.columns(2)

        with col1:
            st.write("Original PDF:")
            st.download_button(
                label="Download Original PDF",
                data=st.session_state.original_pdf,
                file_name="original.pdf",
                mime="application/pdf",
            )
            display_pdf(st.session_state.original_pdf, "Original PDF Viewer")

        with col2:
            st.write("Annotated PDF:")

            # Create two columns for download buttons
            button_col1, button_col2 = st.columns(2)

            with button_col1:
                st.download_button(
                    label="Download Annotated PDF",
                    data=st.session_state.annotated_pdf,
                    file_name="annotated_comparison.pdf",
                    mime="application/pdf",
                )

            with button_col2:
                st.download_button(
                    label="Download Change Log",
                    data=st.session_state.change_log,
                    file_name="change_log.txt",
                    mime="text/plain",
                )

            display_pdf(st.session_state.annotated_pdf, "Annotated PDF Viewer")

        # Display change summary
        st.write("Change Summary:")
        st.text_area("Change Log", value=st.session_state.change_log, height=300)

        # Option to start over
        if st.button("Start Over"):
            st.session_state.comparison_done = False
            st.session_state.annotated_pdf = None
            st.session_state.change_log = None
            st.session_state.original_pdf = None
            st.rerun()


def compare_pdfs(original_pdf, modified_pdf):
    # Create temporary files
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".pdf"
    ) as tmp1, tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp2:
        tmp1.write(original_pdf.getvalue())
        tmp2.write(modified_pdf.getvalue())
        tmp1_path, tmp2_path = tmp1.name, tmp2.name

    # Create temporary output files
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".pdf"
    ) as tmp_output, tempfile.NamedTemporaryFile(
        delete=False, suffix=".txt"
    ) as tmp_summary:
        output_path, summary_path = tmp_output.name, tmp_summary.name

    # Perform comparison
    comparer = EnhancedPDFComparer()
    comparer.compare_pdfs(tmp1_path, tmp2_path, output_path, summary_path)

    # Read results into session state
    with open(output_path, "rb") as file:
        st.session_state.annotated_pdf = file.read()
    with open(summary_path, "r") as file:
        st.session_state.change_log = file.read()
    with open(tmp1_path, "rb") as file:
        st.session_state.original_pdf = file.read()

    # Clean up temporary files
    os.unlink(tmp1_path)
    os.unlink(tmp2_path)
    os.unlink(output_path)
    os.unlink(summary_path)


def display_pdf(pdf_bytes, label):
    # Encode PDF to base64
    base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

    # Embedding PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(f"<p>{label}</p>{pdf_display}", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
