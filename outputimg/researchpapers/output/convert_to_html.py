import sys
import os

def md_to_html(md_path, html_path):
    with open(md_path, 'r') as f:
        lines = f.readlines()
    
    html = ["<html><head><style>body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; } h1, h2, h3 { color: #333; } p { margin-bottom: 15px; } li { margin-bottom: 5px; } img { max-width: 100%; height: auto; display: block; margin: 10px auto; } .caption { text-align: center; font-style: italic; font-size: 0.9em; color: #666; margin-bottom: 20px; }</style></head><body>"]
    
    in_list = False
    
    in_table = False
    
    for line in lines:
        line = line.strip()
        if not line:
            if in_list:
                html.append("</ul>")
                in_list = False
            if in_table:
                html.append("</table></div>")
                in_table = False
            continue
            
        if line.startswith('# '):
            html.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith('## '):
            html.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith('### '):
            html.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith('* '):
            if not in_list:
                html.append("<ul>")
                in_list = True
            content = line[2:]
            if '**' in content:
                parts = content.split('**')
                new_content = ""
                for i, part in enumerate(parts):
                    if i % 2 == 1:
                        new_content += f"<b>{part}</b>"
                    else:
                        new_content += part
                content = new_content
            html.append(f"<li>{content}</li>")
        elif line.startswith('|'):
            # Table handling
            if not in_table:
                html.append('<div style="overflow-x:auto;"><table border="1" style="border-collapse: collapse; width: 100%; margin: 20px 0;">')
                in_table = True
                # Assume first row is header
                cols = [c.strip() for c in line.split('|') if c.strip()]
                html.append("<tr>")
                for col in cols:
                     html.append(f'<th style="padding: 12px; background-color: #f2f2f2; text-align: left;">{col}</th>')
                html.append("</tr>")
            else:
                if '---' in line:
                    continue
                cols = [c.strip() for c in line.split('|') if c.strip()]
                html.append("<tr>")
                for col in cols:
                     html.append(f'<td style="padding: 8px; border: 1px solid #ddd;">{col}</td>')
                html.append("</tr>")
        elif line.startswith('!['):
            # Image handling: ![alt](path)
            alt_end = line.find(']')
            if alt_end != -1:
                alt_text = line[2:alt_end]
                path_start = line.find('(', alt_end)
                path_end = line.find(')', path_start)
                if path_start != -1 and path_end != -1:
                    image_path_rel = line[path_start+1:path_end]
                    
                    # Convert to absolute path for Weasyprint
                    # script is in outputimg/researchpapers/output/
                    # images are relative like ../../methodology_flowchart.png
                    
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    image_path_abs = os.path.abspath(os.path.join(script_dir, image_path_rel))
                    
                    if os.path.exists(image_path_abs):
                        image_src = f"file://{image_path_abs}"
                    else:
                        print(f"Warning: Image not found at {image_path_abs}")
                        image_src = image_path_rel # Fallback

                    html.append(f'<div style="text-align:center; margin: 20px;"><img src="{image_src}" alt="{alt_text}" /><p class="caption">{alt_text}</p></div>')
        else:
            if in_list:
                html.append("</ul>")
                in_list = False
            
            # Bold handling
            content = line
            if '**' in content:
                parts = content.split('**')
                new_content = ""
                for i, part in enumerate(parts):
                    if i % 2 == 1:
                        new_content += f"<b>{part}</b>"
                    else:
                        new_content += part
                content = new_content
            html.append(f"<p>{content}</p>")
            
    if in_list:
        html.append("</ul>")
        
    html.append("</body></html>")
    
    with open(html_path, 'w') as f:
        f.write("\n".join(html))
    print(f"Created {html_path}")

if __name__ == "__main__":
    md_file = "/home/sumit/Downloads/ChestXRayModel/outputimg/researchpapers/output/research_paper.md"
    html_file = "/home/sumit/Downloads/ChestXRayModel/outputimg/researchpapers/output/research_paper.html"
    md_to_html(md_file, html_file)
