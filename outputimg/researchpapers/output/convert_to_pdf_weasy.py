from weasyprint import HTML, CSS
import sys
import os

def convert_to_pdf(html_path, pdf_path):
    print(f"Converting {html_path} to {pdf_path}...")
    
    # Custom CSS for better PDF rendering
    css_string = """
    @page {
        size: A4;
        margin: 2.5cm;
        @bottom-right {
            content: counter(page);
        }
    }
    body {
        font-family: Arial, serif;
        font-size: 11pt;
        line-height: 1.6;
        text-align: justify;
    }
    h1 { font-size: 20pt; text-align: center; margin-bottom: 20px; }
    h2 { font-size: 16pt; margin-top: 20px; border-bottom: 1px solid #ccc; padding-bottom: 5px; }
    h3 { font-size: 14pt; margin-top: 15px; }
    p { margin-bottom: 15px; }
    img { max-width: 100%; height: auto; display: block; margin: 20px auto; }
    .caption { font-size: 10pt; font-style: italic; text-align: center; color: #555; margin-top: 5px; }
    table { width: 100%; border-collapse: collapse; margin: 20px 0; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
    th { background-color: #f2f2f2; font-weight: bold; }
    """
    
    try:
        html = HTML(filename=html_path)
        css = CSS(string=css_string)
        html.write_pdf(pdf_path, stylesheets=[css])
        print(f"Successfully created {pdf_path}")
    except Exception as e:
        print(f"Error generating PDF: {e}")

if __name__ == "__main__":
    html_file = "/home/sumit/Downloads/ChestXRayModel/outputimg/researchpapers/output/research_paper.html"
    pdf_file = "/home/sumit/Downloads/ChestXRayModel/outputimg/researchpapers/output/research_paper.pdf"
    convert_to_pdf(html_file, pdf_file)
