import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import os
import re
import random
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

from torchviz import make_dot


# Configurações
DATASET_PATH = "dataset/gado/Vector/B2/Side/data/Side/images"
NEW_IMAGE_PATH = "Foto nova/imagens/side.jpg"
RESULTS_FILE = "B2 - RGB - Side.xlsx"
BATCH_SIZE = 16
EPOCHS = 10
K_FOLDS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(16)
print(f"Usando {torch.get_num_threads()} threads para processamento.")

# Função para extrair informações do nome do arquivo
def parse_filename(filename):
    match = re.match(r"([\d.]+)_([sr])_(\d+)_([\d.]+)_([MF])", filename)
    if match:
        return {
            "id": float(match.group(1)),  # ID agora é float (ex: 86.0)
            "position": "Side" if match.group(2) == "s" else "Rear",
            "weight": float(match.group(3)),  # Peso real do animal
            "sex": match.group(5)
        }
    return None

# Dataset personalizado
class CattleDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
        self.data = [parse_filename(img) for img in self.images if parse_filename(img)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")  
        #image = Image.open(img_path).convert("L") tons cinza
        if self.transform:
            image = self.transform(image)

        metadata = parse_filename(img_name)  # Retorna um dicionário com {'id', 'weight', 'position', 'sex'}
        return image, metadata  # Certifique-se de que está retornando dois valores!

# Transformações
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #Normalize para treino em RGB
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    #Normalize para treino em cinza
    #transforms.Normalize(mean=[0.5], std=[0.5])
])

# Carregar dataset
dataset = CattleDataset(DATASET_PATH, transform=transform)

# Modelo CNN
class CattleWeightPredictor(nn.Module):
    def __init__(self):
        super(CattleWeightPredictor, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)
    def forward(self, x):
        return self.model(x)

# Inicializar modelo
model = CattleWeightPredictor().to(DEVICE)

# Carregar pesos se existirem
MODEL_PATH = "B2 - Side - RGB.pth"
OPTIMIZER_PATH = "optimizer-b2-rgb-side.pth"
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    print("Pesos do modelo carregados!")

# Função de treinamento
def train_fold(train_loader, model, criterion, optimizer):
    model.train()
    for batch in train_loader:
        images, labels = batch  # batch é uma tupla de (imagens, metadados)
        
        # As imagens estão na primeira posição da tupla
        images = images.to(DEVICE)
        
        # Agora extrai o peso real do dicionário de labels
        weights = torch.tensor([label["weight"] for label in labels], dtype=torch.float32).to(DEVICE).unsqueeze(1)
    
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, weights)
        loss.backward()
        optimizer.step()

# Função de validação
def validate_fold(val_loader, model, criterion, fold, results_file):
    model.eval()
    mae = 0
    results = []
    
    with torch.no_grad():
        for images, labels in val_loader:  # Já recebe `images` no formato correto
            images = torch.stack(images).to(DEVICE) if isinstance(images, tuple) else images
            
            weights = torch.tensor([label["weight"] for label in labels], dtype=torch.float32).to(DEVICE).unsqueeze(1)
            
            # Fazer predição
            outputs = model(images)
            
            mae += torch.abs(outputs - weights).sum().item()

            for i, label in enumerate(labels):
                results.append({
                    "Fold": fold + 1,
                    "ID": label['id'],
                    "Peso Real (kg)": label['weight'],
                    "Peso Predito (kg)": round(outputs[i].item(), 2),
                    "Posição": label['position'],
                    "Sexo": label['sex'],
                    "MAE (kg)": round(torch.abs(outputs[i] - weights[i]).item(), 2)
                })

    # Salvar os resultados no arquivo
    #    with open(results_file, "a") as f:
    #        f.write("Fold    ID  Peso Real (kg)  Peso Predito (kg)  Sexo  MAE (kg)\n")
    #        for res in results:
    #            result_str = f"{res['Fold']:>4} {res['ID']:>5} {res['Peso Real (kg)']:>15.2f} {res['Peso Predito (kg)']:>18.2f} {res['Posição']:>8} {res['Sexo']:>4} {res['MAE (kg)']:>14.2f}\n"
    #            print(result_str.strip())  # Exibir no terminal
    #            f.write(result_str)  # Escrever no arquivo

    # Exibir resultados formatados como tabela no console
    df = pd.DataFrame(results)
    print(df.to_string(index=False))  # Print sem o índice numérico

    if os.path.exists(results_file):
        df_existing = pd.read_excel(results_file)
        df = pd.concat([df_existing, df], ignore_index=True)
    # Salvar no arquivo Excel sem sobrescrever os dados anteriores
    df.to_excel(results_file, index=False)

    return mae / len(val_loader.dataset)

    # Criar ou limpar arquivo de resultados
if os.path.exists(RESULTS_FILE):
    os.remove(RESULTS_FILE)

    # Criar ou limpar arquivo de resultados
    # open(RESULTS_FILE, "w").close()

# Cross-validation
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
data_indices = list(range(len(dataset)))

mae_folds = []  # Lista para armazenar os MAEs de cada Fold

for fold, (train_idx, val_idx) in enumerate(kf.split(data_indices)):
    print(f"Treinando fold {fold+1}/{K_FOLDS}")
    train_sampler = torch.utils.data.Subset(dataset, train_idx)
    val_sampler = torch.utils.data.Subset(dataset, val_idx)
    
    # Criação dos DataLoaders
    def collate_fn(batch):
        images, metadata = zip(*batch)
        images = [img if isinstance(img, torch.Tensor) else transforms.ToTensor()(img) for img in images]  # Garante que todos os elementos são tensores
        images = torch.stack(list(images))  # Converte lista de tensores para um único tensor
        return images.to(DEVICE), metadata  # Move as imagens para o DEVICE aqui
    
    train_loader = DataLoader(train_sampler, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_sampler, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    if os.path.exists(OPTIMIZER_PATH):
        optimizer.load_state_dict(torch.load(OPTIMIZER_PATH, weights_only=True))
        print("Otimizador carregado!")
    
    # 🚀 Adicionar este print para ver quantas imagens foram analisadas por fold
    print(f"Fold {fold+1}: {len(train_sampler)} imagens usadas para treino e {len(val_sampler)} para validação.")

    for epoch in range(EPOCHS):
        train_fold(train_loader, model, criterion, optimizer)
    
    mae = validate_fold(val_loader, model, criterion, fold, RESULTS_FILE)
    print(f"MAE Fold {fold+1}: {mae:.2f}")

    mae_folds.append(mae)  # Adicionando o MAE do fold à lista


# Calcular e exibir o MAE médio após todos os Folds
mae_medio = sum(mae_folds) / len(mae_folds)
print(f"\nMAE Médio de todos os Folds: {mae_medio:.2f}")

# Salvar modelo e otimizador
torch.save(model.state_dict(), MODEL_PATH)
torch.save(optimizer.state_dict(), OPTIMIZER_PATH)

# Função para exibir imagens aleatórias com pesos reais e previstos
def mostrar_exemplos():
    amostra_indices = random.sample(range(len(dataset)), 5)
    model.eval()
    with torch.no_grad():
        for idx in amostra_indices:
            image, metadata = dataset[idx]  # Retorna (imagem, dicionário de metadados)
            
            # Extraindo os valores corretamente do dicionário
            real_weight = metadata["weight"]
            sex = metadata["sex"]
            img_id = metadata["id"]
            position = metadata["position"]

            image_tensor = image.unsqueeze(0).to(DEVICE)
            predicted_weight = model(image_tensor).item()

            # Exibir a imagem com informações
            plt.imshow(image.permute(1, 2, 0).numpy() * 0.5 + 0.5)  # Desnormaliza para exibir corretamente
            plt.title(f"ID: {img_id}, Peso Real: {real_weight:.2f} kg, Previsto: {predicted_weight:.2f} kg\nPosição: {position}, Sexo: {sex}")
            plt.axis('off')
            plt.show()

# Chamar a função corrigida
mostrar_exemplos()

# Predição para nova imagem sem rótulo
def predict_new_image(image_path, model):
    model.eval()
    #image = Image.open(image_path).convert("L")
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        predicted_weight = model(image).item()
    print(f"Peso previsto para a nova imagem: {predicted_weight:.2f} kg")

predict_new_image(NEW_IMAGE_PATH, model)

#
# Criar um tensor de entrada simulado (batch de uma imagem RGB 224x224)
dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)

# Gerar a saída do modelo
output = model(dummy_input)

# Criar o diagrama da rede
dot = make_dot(output, params=dict(model.named_parameters()))

# Exibir a topologia da rede
dot.render("network_topology-b2-rgb-side", format="png", cleanup=True)  # Salva como PNG
dot.view()  # Exibe a imagem

