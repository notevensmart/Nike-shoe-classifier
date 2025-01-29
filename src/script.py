import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from PIL import Image
from io import BytesIO

# Define the website URL and the folder to save images
website_url = "https://www.kickclub.net/Air-Force-1-c221625/"  # Replace with the actual website URL
output_folder = "C:/Users/parth/OneDrive/Documents/ML project/Nike-shoe-classifier/Data/Fake"

os.makedirs(output_folder, exist_ok=True)

# Function to download and save an image
def download_image(img_url, output_folder):
    try:
        response = requests.get(img_url)
        response.raise_for_status()

        # Convert image to PNG and save it
        img = Image.open(BytesIO(response.content))
        img_name = os.path.splitext(os.path.basename(img_url))[0]
        img_path = os.path.join(output_folder, f"{img_name}.png")
        img.convert("RGB").save(img_path, "PNG")
        print(f"Downloaded and saved: {img_name}.png")
    except Exception as e:
        print(f"Failed to download or process image {img_url}: {e}")

# Step 1: Fetch the website page content
response = requests.get(website_url)
if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')

    # Step 2: Find all <a> tags with "nike" in the href
    nike_links = [a['href'] for a in soup.find_all('a', href=True) if 'air' in a['href'].lower()]
    print(f"Found {len(nike_links)} Nike-related links.")

    for link in nike_links:
        full_url = urljoin(website_url, link)

        try:
            # Step 3: Visit the Nike-related page
            print(f"Visiting: {full_url}")
            link_response = requests.get(full_url)
            link_response.raise_for_status()

            # Parse the linked page
            link_soup = BeautifulSoup(link_response.text, 'html.parser')

            # Step 4: Look for the class `swipebox_img swipebox` and its href (first image only)
            image_element = link_soup.find('a', class_="swipebox_img swipebox")
            if image_element:
                img_href = image_element.get('href')
                if img_href:
                    img_url = urljoin(full_url, img_href)
                    print(f"Downloading first image from: {img_url}")
                    download_image(img_url, output_folder)

        except Exception as e:
            print(f"Failed to process link {full_url}: {e}")

else:
    print(f"Failed to access the website. Status code: {response.status_code}")