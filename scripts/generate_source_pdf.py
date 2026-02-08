import os

def create_source_code_html():
    main_dart_path = "/home/sumit/Downloads/ChestXRayModel/geminicxr/flutter_application_2/lib/main.dart"
    inference_service_path = "/home/sumit/Downloads/ChestXRayModel/geminicxr/flutter_application_2/lib/services/inference_service_io.dart"
    
    with open(main_dart_path, 'r') as f:
        main_dart_content = f.read()
        
    with open(inference_service_path, 'r') as f:
        inference_service_content = f.read()

    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Source Code - TB-Ray Analyzer</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Fira+Code&display=swap');
            body {{ font-family: 'Fira Code', monospace; padding: 40px; font-size: 9pt; line-height: 1.4; color: #333; }}
            .file-header {{ background: #f0f4f8; padding: 15px; margin: 30px 0 10px 0; font-weight: bold; border-left: 6px solid #2c5282; color: #2c5282; font-size: 12pt; }}
            pre {{ background: #fafafa; padding: 20px; border: 1px solid #e2e8f0; border-radius: 4px; white-space: pre-wrap; word-wrap: break-word; overflow-x: auto; }}
            .page-header {{ text-align: center; border-bottom: 2px solid #2c5282; padding-bottom: 20px; margin-bottom: 40px; }}
            h1 {{ color: #2c5282; margin-bottom: 5px; }}
            .metadata {{ color: #4a5568; font-size: 11pt; }}
            @media print {{
                body {{ padding: 0; }}
                .file-header {{ page-break-after: avoid; }}
                pre {{ border: none; background: transparent; padding: 0; }}
            }}
        </style>
    </head>
    <body>
        <div class="page-header">
            <h1>SOFTWARE SOURCE CODE REPOSITORY</h1>
            <div class="metadata">
                <p><strong>Title of Work:</strong> TB-Ray: AI-Assisted Chest X-ray Analysis and Reporting System</p>
                <p><strong>Author & Owner:</strong> GHUTKE SUMEET PRABHU</p>
                <p><strong>Classification:</strong> Computer Software (Literary Work)</p>
                <p><strong>Filing Date:</strong> 07-02-2026</p>
            </div>
        </div>

        <div class="file-header">FILE 1: lib/main.dart (Main Application logic & UI)</div>
        <pre>{main_dart_content}</pre>

        <div class="file-header">FILE 2: lib/services/inference_service_io.dart (On-Device ML Inference Engine)</div>
        <pre>{inference_service_content}</pre>
        
        <div style="margin-top: 50px; text-align: center; font-style: italic; color: #718096;">
            --- End of Source Code Submission ---
        </div>
    </body>
    </html>
    """
    
    output_path = "/home/sumit/Downloads/ChestXRayModel/Source_Code_Submission.html"
    with open(output_path, 'w') as f:
        f.write(html_template)
    print(f"Generated {output_path}")

if __name__ == "__main__":
    create_source_code_html()
