from xhtml2pdf import pisa
import os

def convert_html_to_pdf(source_html, output_filename):
    # check if file exists
    if not os.path.isfile(source_html):
        print(f"File not found: {source_html}")
        return

    # open output file for writing (truncated binary)
    with open(output_filename, "wb") as result_file:
        # convert HTML to PDF
        with open(source_html, "r") as f:
            source_html_content = f.read()
            
        # Create PDF
        pisa_status = pisa.CreatePDF(
            source_html_content,                # the HTML to convert
            dest=result_file,                   # the file handle to recieve result
            link_callback=link_callback         # custom handler for loading images
        )

    # return True on success and False on errors
    if pisa_status.err:
        print(f"Error extracting PDF: {pisa_status.err}")
    else:
        print(f"Successfully created {output_filename}")

def link_callback(uri, rel):
    """
    Convert HTML URIs to absolute system paths so xhtml2pdf can verify text resources
    """
    # Use the directory of the script/HTML file as the base
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # If the URI is relative, resolve it relative to the base directory
    if not uri.startswith('http') and not uri.startswith('/'):
        path = os.path.join(base_dir, uri)
    else:
        path = uri
        
    return path

if __name__ == "__main__":
    html_file = "research_paper.html"
    pdf_file = "research_paper.pdf"
    convert_html_to_pdf(html_file, pdf_file)
