
from PIL import Image, ImageDraw, ImageFont
import os

def create_flowchart():
    # Setup
    width = 800
    height = 1000
    bg_color = (255, 255, 255)
    box_color = (230, 240, 250)
    border_color = (0, 0, 0)
    text_color = (0, 0, 0)
    
    img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fallback to default
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
        title_font = font
        small_font = font

    # Helper to draw box
    def draw_box(text, x, y, w, h, subtext=None):
        draw.rectangle([x, y, x+w, y+h], fill=box_color, outline=border_color, width=2)
        
        # Center text
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        draw.text((x + (w - text_w)/2, y + 20), text, fill=text_color, font=font)
        
        if subtext:
            bbox_s = draw.textbbox((0, 0), subtext, font=small_font)
            text_sw = bbox_s[2] - bbox_s[0]
            draw.text((x + (w - text_sw)/2, y + 50), subtext, fill=(50, 50, 50), font=small_font)
            
        return (x + w/2, y + h) # Return bottom center for arrow

    # Helper to draw arrow
    def draw_arrow(start_x, start_y, end_x, end_y):
        draw.line([start_x, start_y, end_x, end_y], fill=border_color, width=2)
        # Arrowhead
        draw.polygon([(end_x, end_y), (end_x-10, end_y-10), (end_x+10, end_y-10)], fill=border_color)

    # Title
    draw.text((250, 30), "Chest X-Ray Model Pipeline", fill=text_color, font=title_font)

    # Nodes
    cx = width / 2
    box_w = 300
    box_h = 80
    gap = 60
    
    y = 80
    
    # Node 1
    p1 = draw_box("Input Chest X-Ray", cx - box_w/2, y, box_w, box_h, "(RGB Image)")
    
    y += box_h + gap
    # Node 2
    p2 = draw_box("Preprocessing", cx - box_w/2, y, box_w, box_h, "Resize 256 -> Crop 224 -> Norm")
    draw_arrow(p1[0], p1[1], p2[0], y)
    
    y += box_h + gap
    # Node 3
    p3 = draw_box("DenseNet121 Model", cx - box_w/2, y, box_w, box_h, "Feature Extraction (ImageNet Weights)")
    draw_arrow(p2[0], p2[1], p3[0], y)
    
    y += box_h + gap
    # Node 4
    p4 = draw_box("Classification Head", cx - box_w/2, y, box_w, box_h, "Global Pool -> Dropout (0.5) -> Linear")
    draw_arrow(p3[0], p3[1], p4[0], y)
    
    y += box_h + gap
    # Node 5
    p5 = draw_box("Output Probabilities", cx - box_w/2, y, box_w, box_h, "Softmax")
    draw_arrow(p4[0], p4[1], p5[0], y)
    
    y += box_h + gap
    
    # Leaf Nodes
    draw_arrow(p5[0], p5[1], cx - 100, y)
    draw_arrow(p5[0], p5[1], cx + 100, y)
    
    draw.rectangle([cx - 200, y, cx - 20, y+60], fill=(220, 255, 220), outline=border_color, width=2)
    draw.text((cx - 160, y+20), "Normal", fill=(0, 100, 0), font=font)
    
    draw.rectangle([cx + 20, y, cx + 200, y+60], fill=(255, 220, 220), outline=border_color, width=2)
    draw.text((cx + 60, y+20), "Tuberculosis", fill=(150, 0, 0), font=font)
    
    # Save
    output_dir = "outputimg"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, "methodology_flowchart.png")
    img.save(output_path)
    print(f"Flowchart saved to {output_path}")

if __name__ == "__main__":
    create_flowchart()
