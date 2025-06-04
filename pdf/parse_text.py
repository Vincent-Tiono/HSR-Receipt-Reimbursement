# pip install 'markitdown[all]~=0.1.0a1'

from markitdown import MarkItDown

# Initialize the MarkItDown converter
md_converter = MarkItDown()

# Convert the PDF file
result = md_converter.convert("pdf\docs\pdf_1.pdf")

# Access the converted Markdown content
markdown_content = result.text_content

print(markdown_content)
