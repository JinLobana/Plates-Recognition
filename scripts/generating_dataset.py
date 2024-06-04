from PIL import Image, ImageDraw, ImageFont
import os

def generate_images(font_path, output_dir, characters, img_size=(100, 150)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for char in characters:
        img = Image.new('L', img_size, color=0)  # 'L' mode for greyscale, 0 for black background
        draw = ImageDraw.Draw(img)
        
        # Load a font
        font_size = 175  # You can adjust this size based on your needs
        font = ImageFont.truetype(font_path, font_size)
        
        # Get the bounding box of the text
        bbox = draw.textbbox((0, 0), char, font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        
        # Calculate X, Y position of the text
        x = (img_size[0] - text_width) // 2
        y = (img_size[1] - text_height) // 2
        
        # Draw the character
        draw.text((x, y), char, fill=255, font=font)  # 255 for white color
        
        # Save the image
        img.save(os.path.join(output_dir, f"{char}.png"))

if __name__ == "__main__":
    # Define the characters to include in the dataset
    characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    
    # Path to the font file (You can use any .ttf file available on your system)
    font_path = "font/din-1451-std/DINEngschriftStd.otf"  # Update this path to your font file
    
    # Output directory to save the images
    output_dir = "dane/letters_digits"
    
    # Generate the images
    generate_images(font_path, output_dir, characters)
