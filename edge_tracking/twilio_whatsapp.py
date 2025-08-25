import os
from twilio.rest import Client
from datetime import datetime
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def upload_image_to_freeimage(image_path):
    """Upload image to FreeImage.host and return the URL"""
    try:
        url = "https://freeimage.host/api/1/upload"
        with open(image_path, 'rb') as file:
            files = {'source': file}
            data = {'key': os.getenv('FREEIMAGE_API_KEY')}  # Load from .env
            response = requests.post(url, files=files, data=data)

        if response.status_code == 200:
            result = response.json()
            if result['status_code'] == 200:
                image_url = result['image']['url']
                print(f"Image uploaded: {image_url}")
                return image_url

        print(f"Upload failed: {response.text}")
        return None
    except Exception as e:
        print(f"Error uploading image: {str(e)}")
        return None

def send_whatsapp_alert_with_image(image_path=None, timestamp=None):
    """Send WhatsApp alert with image when falling is detected"""

    # Load from environment variables
    account_sid = os.getenv('TWILIO_ACCOUNT_SID')
    auth_token = os.getenv('TWILIO_AUTH_TOKEN')
    twilio_whatsapp_number = os.getenv('TWILIO_WHATSAPP_NUMBER')
    your_whatsapp_number = os.getenv('YOUR_WHATSAPP_NUMBER')

    # Validate that all required environment variables are present
    if not all([account_sid, auth_token, twilio_whatsapp_number, your_whatsapp_number]):
        print("Error: Missing required environment variables. Please check your .env file.")
        return None

    try:
        client = Client(account_sid, auth_token)

        message_body = f"⚠️ FALL DETECTED! Time: {timestamp}"

        message = client.messages.create(
            body=message_body,
            from_=twilio_whatsapp_number,
            to=your_whatsapp_number
        )

        print(f"Fall alert sent! Message SID: {message.sid}")

        # Upload and send image link separately
        if image_path and os.path.exists(image_path):
            image_url = upload_image_to_freeimage(image_path)
            if image_url:
                # Send follow-up message with image link
                follow_up = client.messages.create(
                    media_url=[image_url],
                    from_=twilio_whatsapp_number,
                    to=your_whatsapp_number
                )
                print(f"Image link sent! Message SID: {follow_up.sid}")

        return message.sid

    except Exception as e:
        print(f"Error sending WhatsApp alert: {str(e)}")
        return None