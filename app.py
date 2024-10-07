import streamlit as st
import tempfile
import os
from pdf_compare import PDFComparer
import base64


def main():
    st.title("PDF Comparison Tool")

    # Initialize session state
    if "comparison_done" not in st.session_state:
        st.session_state.comparison_done = False
    if "annotated_pdf" not in st.session_state:
        st.session_state.annotated_pdf = None
    if "change_log" not in st.session_state:
        st.session_state.change_log = None

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

        # Display annotated PDF
        st.write("Annotated PDF:")
        display_pdf(st.session_state.annotated_pdf)

        # Display change summary
        st.write("Change Summary:")
        st.text_area("", st.session_state.change_log, height=300)

        # Download options
        st.subheader("Download Options")

        # Download annotated PDF
        st.download_button(
            label="Download Annotated PDF",
            data=st.session_state.annotated_pdf,
            file_name="annotated_comparison.pdf",
            mime="application/pdf",
        )

        # Download change log
        st.download_button(
            label="Download Change Log",
            data=st.session_state.change_log,
            file_name="change_log.txt",
            mime="text/plain",
        )

        # Option to start over
        if st.button("Start Over"):
            st.session_state.comparison_done = False
            st.session_state.annotated_pdf = None
            st.session_state.change_log = None
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
    comparer = PDFComparer()
    original_blocks = comparer.extract_text_blocks(tmp1_path)
    modified_blocks = comparer.extract_text_blocks(tmp2_path)
    changes = comparer.analyze_semantic_changes(original_blocks, modified_blocks)
    comparer.highlight_changes_in_pdf(tmp2_path, changes, output_path)
    comparer.generate_change_summary(changes, summary_path)

    # Read results into session state
    with open(output_path, "rb") as file:
        st.session_state.annotated_pdf = file.read()
    with open(summary_path, "r") as file:
        st.session_state.change_log = file.read()

    # Clean up temporary files
    os.unlink(tmp1_path)
    os.unlink(tmp2_path)
    os.unlink(output_path)
    os.unlink(summary_path)


def display_pdf(pdf_bytes):
    base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
