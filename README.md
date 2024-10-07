# PDF Comparison Tool

This PDF Comparison Tool is a Streamlit web application that allows users to compare two PDF files, highlighting the differences and providing a detailed change log. It's designed to make document comparison easy and accessible through a user-friendly interface.

## Features

- Upload and compare two PDF files
- Visual highlighting of differences in an annotated PDF
- Detailed change log summarizing all modifications
- Download options for both the annotated PDF and change log
- Persistent results until the user chooses to start a new comparison

## Installation

1. Clone this repository:

   ```
   git clone https://github.com/adataguy-ai/pdfcomparison.git
   cd pdf-comparison-tool
   ```

2. Create a virtual environment (optional but recommended):

   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:

   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Use the interface to:
   - Upload two PDF files for comparison
   - Initiate the comparison process
   - View the annotated PDF and change log
   - Download the results

## How it Works

The PDF Comparison Tool uses the `PDFComparer` class (implemented in `pdf_compare.py`) to analyze and compare the uploaded PDF files. The comparison process involves:

1. Extracting text blocks from both PDFs
2. Analyzing semantic changes between the documents
3. Generating an annotated PDF highlighting the differences
4. Creating a detailed change log

The Streamlit app provides a user-friendly interface for this process and allows users to easily view and download the results.

## Dependencies

- streamlit
- PyMuPDF (fitz)
- Other dependencies as specified in `requirements.txt`

## Contributing

Contributions to improve the PDF Comparison Tool are welcome. Please feel free to submit issues or pull requests.

## Contact

Saurabh Tripathi
