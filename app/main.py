from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from torchvision import transforms
from PIL import Image
import torch
import uvicorn
from app.model.generator import Generator

app = FastAPI()


# Load your GAN model
gen_BA = Generator(3, 3)
gen_BA.load_state_dict(torch.load('model/models.pth', map_location=torch.device('cpu')))
gen_BA.eval()

# Define image transformations
my_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


class ImageUpload(BaseModel):
    file: UploadFile


@app.get("/")
def home():
    return {"health_check": "OK", "model_version": "Your GAN Model Version"}


@app.post("/predict", response_model=str)
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    # Save the uploaded image temporarily
    with open("temp_image.jpg", "wb") as f:
        f.write(contents)

    # Open and transform the image
    img = Image.open("temp_image.jpg").convert("RGB")
    transformed_img = my_transforms(img)

    # Add an extra batch dimension as the model expects batches
    transformed_img = transformed_img.unsqueeze(0)

    # Generate fake image using the loaded GAN model
    fake_B = gen_BA(transformed_img)

    # Save the generated image temporarily
    fake_B_path = "fake_B.jpg"
    fake_B_image = transforms.ToPILImage()(fake_B.squeeze(0).detach().cpu())
    fake_B_image.save(fake_B_path)

    # Return the path to the generated image
    return FileResponse(fake_B_path, media_type="image/jpeg", filename="fake_B.jpg")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
